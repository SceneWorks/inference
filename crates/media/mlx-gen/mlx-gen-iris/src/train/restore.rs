//! Native MLX **Iris-3B restoration training** (story sc-25688): the paper's one-step restoration
//! recipe (arXiv 2610.09450v1 §5.3, following HYPIR) over the backend-neutral contract
//! [`gen_core::iris::restoration_train`].
//!
//! Each micro-batch builds its training pairs host-side (a seeded second-order Real-ESRGAN
//! degradation of a clean crop, or an aligned degraded/clean pair), then trains **one** network:
//!
//! * a **generator window** (`(batch_count // accum)` even): `x̂ = x_lq − σ·v(x_lq, σ·N, ∅)` —
//!   the `iris_3b_restore` forward — and the gradient of
//!   `(λ₂·mse(x̂, y) + λ_lpips·LPIPS(x̂, y) + λ_gan·D_G(x̂)) / accum` over the generator's trainable
//!   tensors (critic in eval mode: no power iteration); closing the window clips (1.0), steps AdamW,
//!   then updates the EMA from the post-step weights (`HYPIR/trainer/base.py::run`);
//! * a **critic window**: `x̂` regenerated under no-grad, one spectral power iteration before each
//!   of `D(y, real)` and `D(x̂, fake)`, the gradient of `(mean D_real + mean D_fake) / accum` over the
//!   head; closing the window clips (1.0) and steps the critic's AdamW. One generator window + one
//!   critic window is one **step**.
//!
//! Durability reuses the generation trainer's machinery (atomic checkpoint publish, retention,
//! `DataWalk` cursor) and adds the critic head, its spectral buffers and its optimizer to the
//! checkpoint; the degradation and crop draws are positional (`(seed, epoch, item)`), so cancel +
//! resume reproduces the uninterrupted run bit-exactly. The run exports an `upscaler/`-layout
//! restoration model (full fine-tune, or the adapter merged into the base) that loads in
//! `iris_3b_restore` with no training-only dependency.
//!
//! Every hyper-parameter comes from the request. The documented default request is
//! [`gen_core::iris::restoration_train::recipe_training_config`] — the paper's **full** fine-tune,
//! AdamW at lr 1e-5 with weight decay 0.01 for generator and critic, constant LR; a LoRA/LoKr
//! adapter (HYPIR's own arm) trains only when the request asks for one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use gen_core::iris::restoration_train::{
    self as contract, base_identity, check_resume, critic_head_init, export_metadata, phase_of,
    read_checkpoint_state, training_pair, validate_item, write_export_sidecars, ConvNextConfig,
    LossWindow, Phase, RestoreArtifact, RestoreCheckpointState, RestoreTrainPlan,
    RestoreTrainResources, TrainPair, WindowClose, CKPT_CRITIC, CKPT_CRITIC_OPTIMIZER,
    CRITIC_HEAD_WIDTH, TRAINER_ID,
};
use gen_core::iris::train::lineage::{publish_export_dir, retire_foreign_checkpoints};
use gen_core::iris::train::{
    adapter_param_route, checkpoint_root, checkpoint_staging_dir, file_stem, full_param_route,
    latest_checkpoint, linear_paths_from_keys, lr_factor, mix_seed, prune_checkpoints,
    publish_checkpoint, DataWalk, MixedPrecision, MuonAdjustLr, OptimizerKind, OptimizerPlan,
    ParamRoute, ResumeDataPolicy, WeightsSelect, CKPT_EMA, CKPT_OPTIMIZER, CKPT_STATE,
    CKPT_TRAINABLE, TECHNIQUES,
};
use gen_core::iris::{IrisConfig, FAMILY};
use gen_core::train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques, Trainer,
    TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_gen::adapters::reconstruct_lokr_delta;
use mlx_gen::gen_core;
use mlx_gen::{Error, LoadSpec, Modality, Result};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use super::critic::{
    head_from_arrays, head_from_host, multilevel_loss, power_iteration, ConvNextTrunk, Critic,
    SpectralState,
};
use super::lpips::Lpips;
use super::model::{
    init_adapter, provider_dtype, save_safetensors_atomic, AdapterKind, AdapterTarget, TrainModel,
    Trainable,
};
use super::optim::{clip_grads, ema_update, IrisOptimizer, Params};
use super::{check_backbone_keys, load_source, load_tensors, rc_map, save_params};
use crate::dit::{IrisDiT, TextBatch};

/// Identity + capabilities of the MLX Iris restoration trainer.
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

/// The frozen training-only networks of a run.
pub struct FrozenNets {
    pub lpips: Lpips,
    pub trunk: ConvNextTrunk,
    pub trunk_cfg: ConvNextConfig,
    /// The critic head's width (`MultiLevelDConv(out_ch)`).
    pub head_width: usize,
}

/// Builds the frozen networks for a critic compute dtype.
pub type NetsProvider = Box<dyn Fn(Dtype) -> Result<FrozenNets>>;

/// The resolved resources of a restoration training load.
pub struct IrisRestoreTrainer {
    descriptor: TrainerDescriptor,
    res: RestoreTrainResources,
    nets: NetsProvider,
}

impl IrisRestoreTrainer {
    /// A trainer over already-resolved resources whose frozen networks come from `nets` instead of
    /// the pinned component files (the miniature lifecycle tests; the recipe networks are what
    /// [`load_trainer`] reads).
    pub fn with_networks(res: RestoreTrainResources, nets: NetsProvider) -> Self {
        Self {
            descriptor: trainer_descriptor(),
            res,
            nets,
        }
    }
}

/// The recipe's frozen networks from the staged pinned files: OpenCLIP ConvNeXt-XXL, LPIPS-VGG.
fn pinned_networks(res: &RestoreTrainResources) -> NetsProvider {
    let (trunk, vgg, lin) = (
        res.critic_trunk.clone(),
        res.vgg16.clone(),
        res.lpips_lin.clone(),
    );
    Box::new(move |compute| {
        let trunk_cfg = ConvNextConfig::xxlarge();
        Ok(FrozenNets {
            trunk: ConvNextTrunk::load(&trunk, &trunk_cfg, compute)?,
            lpips: Lpips::load(&vgg, &lin)?,
            trunk_cfg,
            head_width: CRITIC_HEAD_WIDTH,
        })
    })
}

/// Construct the trainer: `spec.weights` = the restoration export directory, plus the training-only
/// components (see [`RestoreTrainResources::from_spec`]). Nothing heavy loads here.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    refuse_trainer_load_overlays(TRAINER_ID, spec)?;
    for (field, set) in [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("pid", spec.pid.is_some()),
    ] {
        if set {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: LoadSpec::{field} is not supported"
            )));
        }
    }
    let res = RestoreTrainResources::from_spec(spec)?;
    let nets = pinned_networks(&res);
    Ok(Box::new(IrisRestoreTrainer::with_networks(res, nets)))
}

mlx_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

impl IrisRestoreTrainer {
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

    fn plan(&self, req: &TrainingRequest, linear_paths: &[String]) -> Result<RestoreTrainPlan> {
        Ok(RestoreTrainPlan::resolve(
            req,
            &self.res.export,
            linear_paths,
            self.res.parent.is_some(),
        )?)
    }
}

impl Trainer for IrisRestoreTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.floors(req)?;
        self.plan(req, &[])?;
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
// The run
// =============================================================================================

fn adamw_plan(lr: f64, betas: (f64, f64), weight_decay: f64) -> OptimizerPlan {
    OptimizerPlan {
        kind: OptimizerKind::AdamW,
        base_lr: lr,
        lr,
        betas,
        weight_decay,
        muon_momentum: 0.0,
        muon_nesterov: false,
        muon_adjust_lr: MuonAdjustLr::None,
        auto_lr: "none".into(),
        base_batch_size: 1,
    }
}

fn to_exc(e: impl std::fmt::Display) -> mlx_rs::error::Exception {
    mlx_rs::error::Exception::custom(e.to_string())
}

/// The frozen networks and fixed tensors of a run.
pub struct Frozen {
    pub lpips: Lpips,
    pub critic: Critic,
    /// `[1, T, L, D]` empty-prompt states and the mask.
    pub states: Array,
    pub mask: Vec<i32>,
    pub model_time: f32,
    pub sigma: f32,
}

/// One micro-batch's arrays.
pub struct PairBatch {
    /// `[B, 3, S, S]` generator input / target in `[−1, 1]`.
    pub lq: Array,
    pub gt: Array,
}

impl PairBatch {
    pub fn from_pairs(pairs: &[TrainPair]) -> Self {
        let s = pairs[0].size as i32;
        let b = pairs.len() as i32;
        let lq: Vec<f32> = pairs.iter().flat_map(|p| p.lq.iter().copied()).collect();
        let gt: Vec<f32> = pairs.iter().flat_map(|p| p.gt.iter().copied()).collect();
        Self {
            lq: Array::from_slice(&lq, &[b, 3, s, s]),
            gt: Array::from_slice(&gt, &[b, 3, s, s]),
        }
    }
}

impl Frozen {
    /// `x̂ = x_lq − σ·v(x_lq, σ·N, ∅)` (f32) — exactly the restorer's per-tile forward.
    pub fn restore(&self, dit: &IrisDiT, lq: &Array) -> Result<Array> {
        let b = lq.shape()[0];
        let states = if b == 1 {
            self.states.clone()
        } else {
            concatenate_axis(&vec![self.states.clone(); b as usize], 0)?
        };
        let masks = vec![self.mask.clone(); b as usize];
        let t = Array::from_slice(&vec![self.model_time; b as usize], &[b]);
        let v = dit
            .forward(
                lq,
                &t,
                &TextBatch {
                    states: &states,
                    mask: &masks,
                },
            )?
            .as_dtype(Dtype::Float32)?;
        Ok(lq.subtract(&v.multiply(Array::from_f32(self.sigma))?)?)
    }
}

/// The Eq. 6 terms of one generator micro-batch (already weighted), each `[]`.
pub struct GeneratorTerms {
    pub l2: Array,
    pub lpips: Array,
    pub gan: Array,
}

/// The generator's weighted loss terms on `x̂` (critic in eval mode). A zero-weighted term is
/// still a term of the objective, so it is skipped only in the sense that it contributes zero.
pub fn generator_terms(
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Params,
    sn: &SpectralState,
    xhat: &Array,
    gt: &Array,
) -> Result<GeneratorTerms> {
    let w = plan.weights;
    let l2 = xhat
        .subtract(gt)?
        .square()?
        .mean(None)?
        .multiply(Array::from_f32(w.l2 as f32))?;
    let zero = Array::from_f32(0.0);
    let lpips = if w.lpips != 0.0 {
        frozen
            .lpips
            .distance(xhat, gt)?
            .mean(None)?
            .multiply(Array::from_f32(w.lpips as f32))?
    } else {
        zero.clone()
    };
    let gan = if w.gan != 0.0 {
        frozen
            .critic
            .generator_term(head, sn, xhat, plan.gan_real_label)?
            .mean(None)?
            .multiply(Array::from_f32(w.gan as f32))?
    } else {
        zero
    };
    Ok(GeneratorTerms { l2, lpips, gan })
}

/// One generator micro-batch: the loss terms (unscaled, host) and the gradient of
/// `total · grad_scale` over the trainable tensors.
#[allow(clippy::too_many_arguments)]
pub fn generator_grads(
    model: &TrainModel,
    params: &Params,
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Params,
    sn: &SpectralState,
    batch: &PairBatch,
    grad_scale: f32,
) -> Result<([f64; 4], Params)> {
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        let dit = model.dit(&p).map_err(to_exc)?;
        let xhat = frozen.restore(&dit, &batch.lq).map_err(to_exc)?;
        let t = generator_terms(frozen, plan, head, sn, &xhat, &batch.gt).map_err(to_exc)?;
        let total = t.l2.add(&t.lpips)?.add(&t.gan)?;
        let scaled = total.multiply(Array::from_f32(grad_scale))?;
        Ok(vec![scaled, total, t.l2, t.lpips, t.gan])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (vals, grads) =
        vg(params.clone(), 0).map_err(|e| Error::Msg(format!("iris restore G step: {e}")))?;
    let mut refs: Vec<&Array> = vals.iter().collect();
    refs.extend(grads.values());
    for chunk in refs.chunks(256) {
        eval(chunk.iter().copied())?;
    }
    let host = |i: usize| super::scalar_f32(&vals[i]).map(f64::from);
    Ok(([host(1)?, host(2)?, host(3)?, host(4)?], grads))
}

/// One critic micro-batch: two power iterations (`D(y)` then `D(x̂)`), the critic loss and the
/// gradient of `loss · grad_scale` over the head. Returns the advanced spectral state.
pub fn critic_grads(
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Params,
    sn: &SpectralState,
    xhat: &Array,
    gt: &Array,
    grad_scale: f32,
) -> Result<(f64, Params, SpectralState)> {
    let cfg = &frozen.critic.head;
    let sn1 = power_iteration(cfg, head, sn)?;
    let sn2 = power_iteration(cfg, head, &sn1)?;
    let label = plan.gan_real_label;
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        let real = multilevel_loss(&frozen.critic.logits(&p, &sn1, gt).map_err(to_exc)?, label)
            .map_err(to_exc)?;
        let fake = multilevel_loss(&frozen.critic.logits(&p, &sn2, xhat).map_err(to_exc)?, 0.0)
            .map_err(to_exc)?;
        let loss = real.mean(None)?.add(&fake.mean(None)?)?;
        Ok(vec![loss.multiply(Array::from_f32(grad_scale))?, loss])
    };
    let (vals, grads) = {
        let mut vg = keyed_value_and_grad(loss_fn);
        vg(head.clone(), 0).map_err(|e| Error::Msg(format!("iris restore D step: {e}")))?
    };
    let mut refs: Vec<&Array> = vals.iter().collect();
    refs.extend(grads.values());
    eval(refs)?;
    Ok((super::scalar_f32(&vals[1])? as f64, grads, sn2))
}

/// Accumulated gradients of one window.
#[derive(Default)]
struct Accum {
    grads: Option<Params>,
}

impl Accum {
    fn add(&mut self, g: Params) -> Result<()> {
        self.grads = Some(match self.grads.take() {
            None => g,
            Some(mut acc) => {
                for (k, v) in g {
                    let s = acc[&k].add(&v)?;
                    acc.insert(k, s);
                }
                acc
            }
        });
        Ok(())
    }
}

struct Run {
    plan: RestoreTrainPlan,
    model: TrainModel,
    params: Params,
    ema: Option<Params>,
    opt: IrisOptimizer,
    head: Params,
    sn: SpectralState,
    critic_opt: IrisOptimizer,
    frozen: Frozen,
    walk: DataWalk,
    total_steps: u64,
    ckpt_root: PathBuf,
    base_identity: String,
    dataset_fingerprint: String,
    export: gen_core::iris::restoration::RestorationResources,
    init_weights: PathBuf,
    // position / counters
    step: u64,
    sched_step: u64,
    lr: f64,
    lr_critic: f64,
    nan_count: u32,
    epoch: u32,
    pos: usize,
    losses: [f64; 5],
}

fn critic_routes(_: &str, _: usize) -> ParamRoute {
    ParamRoute::AdamW
}

/// PEFT's `init_lora_weights="gaussian"` (HYPIR's LoRA init): `A ~ N(0, (1/r)²)`, `B = 0`, drawn
/// from MLX's RNG keyed on `mix_seed(seed, fnv(key))`.
fn gaussian_lora(
    params: &mut Params,
    targets: &[AdapterTarget],
    rank: usize,
    seed: u64,
) -> Result<()> {
    for t in targets {
        let k = format!("{}.lora_A.weight", t.path);
        let key = random::key(mix_seed(&[seed, super::model::fnv(&k)]))?;
        let a = random::normal::<f32>(
            &[rank as i32, t.in_f][..],
            None,
            Some(1.0 / rank as f32),
            Some(&key),
        )?;
        eval([&a])?;
        params.insert(Rc::from(k.as_str()), a);
    }
    Ok(())
}

/// The frozen networks, the empty-prompt conditioning and the fresh critic head of a run.
fn build_frozen(
    res: &RestoreTrainResources,
    plan: &RestoreTrainPlan,
    nets: &dyn Fn(Dtype) -> Result<FrozenNets>,
) -> Result<(Frozen, Params, SpectralState)> {
    let critic_compute = if plan.critic_bf16 {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    };
    let FrozenNets {
        lpips,
        trunk,
        trunk_cfg,
        head_width,
    } = nets(critic_compute)?;
    let head_cfg = trunk_cfg.head(head_width);
    let prompt = res.export.empty_prompt()?;
    let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
    let states = Array::from_slice(&prompt.embeddings, &shape);
    eval([&states])?;
    let frozen = Frozen {
        lpips,
        critic: Critic {
            trunk,
            head: head_cfg,
            compute: critic_compute,
        },
        states,
        mask: prompt.mask,
        model_time: plan.model_time,
        sigma: plan.sigma as f32,
    };
    let (head, sn) = head_from_host(&head_cfg, &critic_head_init(&head_cfg, plan.seed))?;
    Ok((frozen, head, sn))
}

/// The pieces of a **full-fine-tune** run at its initial state (generator model + weights, frozen
/// networks, fresh critic head) — what the step functions take; for tests driving
/// [`generator_grads`] / [`critic_grads`] directly.
pub fn miniature_parts(
    res: &RestoreTrainResources,
    plan: &RestoreTrainPlan,
    nets: NetsProvider,
) -> Result<(TrainModel, Params, Frozen, Params, SpectralState)> {
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
    let config = res.export.config();
    let source = load_source(&res.init_weights(plan.init), None)?;
    check_backbone_keys(&config.model, &source, "the generator weights")?;
    let (frozen, head, sn) = build_frozen(res, plan, nets.as_ref())?;
    Ok((
        TrainModel {
            cfg: config.model.clone(),
            compute,
            trainable: Trainable::Full,
        },
        rc_map(source),
        frozen,
        head,
        sn,
    ))
}

impl Run {
    fn prepare(
        trainer: &IrisRestoreTrainer,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let res = &trainer.res;
        let config: IrisConfig = res.export.config().clone();
        let pre = trainer.plan(req, &[])?;
        let compute = match pre.mixed_precision {
            MixedPrecision::Bf16 => Dtype::Bfloat16,
            MixedPrecision::Fp32 => Dtype::Float32,
            // Only the generation plan resolves fp16 (sc-25689); this task's plan never does.
            MixedPrecision::Fp16 => {
                return Err(Error::Unsupported(
                    "iris training: fp16 mixed precision is the generation trainer's".into(),
                ))
            }
        };
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            validate_item(item, &pre)?;
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total: req.items.len() as u32,
            });
        }
        on_progress(TrainingProgress::LoadingModel);
        let init_weights = res.init_weights(pre.init);
        let adapter_compute = (!req.config.full_finetune).then_some(compute);
        let source = load_source(&init_weights, adapter_compute)?;
        check_backbone_keys(&config.model, &source, "the generator weights")?;
        let shapes = gen_core::iris::train::backbone_tensor_shapes(&config.model);
        let linear_paths =
            linear_paths_from_keys(shapes.iter().map(|(k, s)| (k.as_str(), s.len())));
        let plan = trainer.plan(req, &linear_paths)?;
        let base_identity = base_identity(&res.export, &init_weights)?;

        let (model, mut params) = match &plan.artifact {
            RestoreArtifact::Full => (
                TrainModel {
                    cfg: config.model.clone(),
                    compute,
                    trainable: Trainable::Full,
                },
                rc_map(source),
            ),
            art => {
                let (kind, paths) = match art {
                    RestoreArtifact::Lora {
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
                    RestoreArtifact::Lokr {
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
                    RestoreArtifact::Full => unreachable!(),
                };
                let targets: Vec<AdapterTarget> = paths
                    .iter()
                    .map(|p| {
                        let w = &source[&format!("{p}.weight")];
                        AdapterTarget {
                            path: p.clone(),
                            out_f: w.shape()[0],
                            in_f: w.shape()[1],
                        }
                    })
                    .collect();
                let mut base = HashMap::with_capacity(source.len());
                for (k, a) in source {
                    let a = provider_dtype(&k, &a, compute)?;
                    eval([&a])?;
                    base.insert(k, a);
                }
                let mut params = init_adapter(kind, &targets, plan.seed)?;
                if let (AdapterKind::Lora { rank, .. }, true) = (kind, plan.lora_gaussian_init) {
                    gaussian_lora(&mut params, &targets, rank, plan.seed)?;
                }
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
        let adapter_targets: HashMap<String, String> = match &model.trainable {
            Trainable::Adapter { kind, targets, .. } => targets
                .iter()
                .flat_map(|t| {
                    t.factor_keys(*kind)
                        .into_iter()
                        .map(|k| (k, t.path.clone()))
                })
                .collect(),
            Trainable::Full => HashMap::new(),
        };
        let g = &plan.generator;
        let cfg_model = config.model.clone();
        let mut opt = IrisOptimizer::new(
            &adamw_plan(g.lr, g.betas, g.weight_decay),
            &params,
            |k, nd| match adapter_targets.get(k) {
                Some(t) => adapter_param_route(t, nd, OptimizerKind::AdamW),
                None => full_param_route(k, nd, OptimizerKind::AdamW, &cfg_model),
            },
        )?;
        let mut ema = plan.ema.map(|_| params.clone());

        let (frozen, mut head, mut sn) = build_frozen(res, &plan, trainer.nets.as_ref())?;
        let head_cfg = frozen.critic.head;
        let c = &plan.critic;
        let mut critic_opt = IrisOptimizer::new(
            &adamw_plan(c.lr, c.betas, c.weight_decay),
            &head,
            critic_routes,
        )?;

        let walk = DataWalk::new(req.items.len(), plan.batch_size)?;
        if let Some(w) = walk.unused_tail_warning() {
            eprintln!("{w}");
        }
        let total_steps = gen_core::iris::train::total_optimizer_steps(
            walk.batches_per_epoch(),
            plan.micro_per_step(),
            plan.num_epochs,
            plan.max_steps,
        );
        let ckpt_root = checkpoint_root(&req.output_dir, &req.file_name);
        let dataset_fingerprint = contract::dataset_fingerprint(req)?;

        let (mut step, mut sched_step, mut lr, mut lr_critic, mut nan_count, mut epoch, mut pos) = (
            0u64,
            0u64,
            plan.generator.lr,
            plan.critic.lr,
            0u32,
            1u32,
            0usize,
        );
        let mut losses = [f64::NAN; 5];
        let resume_dir = match (&plan.resume_from, plan.resume) {
            (Some(p), _) => Some(if p.join(CKPT_STATE).is_file() {
                p.clone()
            } else {
                latest_checkpoint(p).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris restore resume: {} holds no complete checkpoint",
                        p.display()
                    ))
                })?
            }),
            (None, true) => latest_checkpoint(&ckpt_root),
            (None, false) => None,
        };
        // sc-25690: the checkpoints under the root that are not this run's lineage move aside
        // before its first publish (retention and `latest` see only this run's).
        retire_foreign_checkpoints(&ckpt_root, resume_dir.as_deref())?;
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let restored = rc_map(load_tensors(&dir.join(CKPT_TRAINABLE))?);
            if restored.len() != params.len() || params.keys().any(|k| !restored.contains_key(k)) {
                return Err(Error::Msg(
                    "iris restore resume: the checkpoint's trainable tensors do not match this run"
                        .into(),
                ));
            }
            params = restored;
            if let Some(e) = ema.as_mut() {
                *e = rc_map(load_tensors(&dir.join(CKPT_EMA))?);
            }
            opt.load_state(&load_tensors(&dir.join(CKPT_OPTIMIZER))?)?;
            let critic = load_tensors(&dir.join(CKPT_CRITIC))?;
            (head, sn) = head_from_arrays(&head_cfg, |k| critic.get(k).cloned())?;
            critic_opt.load_state(&load_tensors(&dir.join(CKPT_CRITIC_OPTIMIZER))?)?;
            step = saved.step;
            sched_step = saved.scheduler_step;
            lr = saved.lr;
            lr_critic = saved.lr_critic;
            nan_count = saved.nan_count;
            losses = saved.losses;
            if plan.resume_data_policy == ResumeDataPolicy::Exact {
                epoch = saved.epoch;
                pos = saved.batches_consumed;
            }
        }
        Ok(Self {
            plan,
            model,
            params,
            ema,
            opt,
            head,
            sn,
            critic_opt,
            frozen,
            walk,
            total_steps,
            ckpt_root,
            base_identity,
            dataset_fingerprint,
            export: res.export.clone(),
            init_weights,
            step,
            sched_step,
            lr,
            lr_critic,
            nan_count,
            epoch,
            pos,
            losses,
        })
    }

    fn batch(&self, req: &TrainingRequest, epoch: u32, pos: usize) -> Result<PairBatch> {
        let pairs = self
            .walk
            .batch(pos)
            .map(|i| training_pair(&req.items[i], &self.plan, epoch, i))
            .collect::<gen_core::Result<Vec<_>>>()?;
        Ok(PairBatch::from_pairs(&pairs))
    }

    fn execute(
        mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let plan = self.plan.clone();
        let accum = plan.grad_accum;
        let per_step = plan.micro_per_step() as u64;
        let bpe = self.walk.batches_per_epoch();
        let max_epoch = plan.num_epochs.unwrap_or(u32::MAX);
        let first_step = self.step;
        let mut last_saved = (self.step > 0).then_some(self.step);
        let grad_scale = 1.0 / accum as f32;
        let mut batch_count: u64 = 0;
        let mut g_acc = Accum::default();
        let mut d_acc = Accum::default();
        let mut g_sums = [0f64; 4];
        let mut d_win: LossWindow<SpectralState> = LossWindow::default();
        let mut window_bad = false;
        let mut boundary = (self.epoch, self.pos);
        let mut cancelled = false;
        let mut finished = self.step >= plan.max_steps as u64;
        let mut epoch = self.epoch;
        let mut start_pos = self.pos;
        while !finished && !cancelled && epoch <= max_epoch {
            for pos in start_pos..bpe {
                // Cancellation lands on step boundaries only (a half-trained step is discarded).
                if req.cancel.is_cancelled() && batch_count.is_multiple_of(per_step) {
                    cancelled = true;
                    break;
                }
                let batch = self.batch(req, epoch, pos)?;
                let (phase, sync) = phase_of(batch_count, accum);
                batch_count += 1;
                match phase {
                    Phase::Generator => {
                        let (terms, grads) = generator_grads(
                            &self.model,
                            &self.params,
                            &self.frozen,
                            &plan,
                            &self.head,
                            &self.sn,
                            &batch,
                            grad_scale,
                        )?;
                        if terms.iter().all(|t| t.is_finite()) {
                            g_acc.add(grads)?;
                            for (s, t) in g_sums.iter_mut().zip(terms) {
                                *s += t / accum as f64;
                            }
                        } else {
                            window_bad = true;
                        }
                        if sync {
                            if window_bad {
                                contract::count_non_finite_window(
                                    "generator",
                                    self.step,
                                    &mut self.nan_count,
                                    plan.nan_loss_tolerance,
                                )?;
                                g_acc = Accum::default();
                                window_bad = false;
                            } else if let Some(g) = g_acc.grads.take() {
                                let lr = self.lr * self.factor();
                                let g: Params = g
                                    .into_iter()
                                    .filter(|(k, _)| self.opt.is_trained(k))
                                    .collect();
                                let (_, g) = clip_grads(g, plan.generator.clip)?;
                                self.opt.step(&mut self.params, &g, lr)?;
                                if let (Some(e), Some(d)) = (self.ema.as_mut(), plan.ema) {
                                    ema_update(e, &self.params, d)?;
                                }
                                self.losses[..4].copy_from_slice(&g_sums);
                            }
                            g_sums = [0.0; 4];
                        }
                    }
                    Phase::Critic => {
                        let dit = self.model.dit(&self.params)?;
                        let xhat = mlx_rs::stop_gradient(&self.frozen.restore(&dit, &batch.lq)?)?;
                        eval([&xhat])?;
                        drop(dit);
                        d_win.begin(|| self.sn.clone());
                        let (loss, grads, sn) = critic_grads(
                            &self.frozen,
                            &plan,
                            &self.head,
                            &self.sn,
                            &xhat,
                            &batch.gt,
                            grad_scale,
                        )?;
                        self.sn = sn;
                        if d_win.record(loss, accum) {
                            d_acc.add(grads)?;
                        }
                        if sync {
                            match d_win.close(
                                "critic",
                                self.step,
                                &mut self.nan_count,
                                plan.nan_loss_tolerance,
                            )? {
                                WindowClose::Step { loss } => {
                                    if let Some(g) = d_acc.grads.take() {
                                        let lr = self.lr_critic * self.factor();
                                        let (_, g) = clip_grads(g, plan.critic.clip)?;
                                        self.critic_opt.step(&mut self.head, &g, lr)?;
                                    }
                                    self.losses[4] = loss;
                                }
                                WindowClose::Skip { restore } => {
                                    // A skipped window leaves no trace: no critic step, and the
                                    // spectral u/v go back to where the window started.
                                    d_acc = Accum::default();
                                    if let Some(sn) = restore {
                                        self.sn = sn;
                                    }
                                }
                            }
                            self.sched_step += 1;
                            self.step += 1;
                            boundary = (epoch, pos + 1);
                            on_progress(TrainingProgress::Training {
                                step: self.micro_step(),
                                total: (self.total_steps * per_step) as u32,
                                loss: self.losses[0] as f32,
                            });
                            if plan.save_every > 0
                                && self.step.is_multiple_of(plan.save_every as u64)
                            {
                                self.save_checkpoint(boundary)?;
                                last_saved = Some(self.step);
                                on_progress(TrainingProgress::Checkpoint {
                                    step: self.micro_step(),
                                });
                            }
                            if plan.preview_every > 0
                                && self.step.is_multiple_of(plan.preview_every as u64)
                            {
                                self.render_preview(req, on_progress)?;
                            }
                            if self.step >= plan.max_steps as u64 {
                                finished = true;
                                break;
                            }
                        }
                    }
                }
            }
            if finished || cancelled {
                break;
            }
            epoch += 1;
            start_pos = 0;
        }
        mlx_rs::memory::clear_cache();
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
            final_loss: self.losses[0] as f32,
        })
    }

    fn factor(&self) -> f64 {
        lr_factor(
            self.plan.schedule,
            self.sched_step,
            self.plan.warmup_steps,
            self.total_steps,
        )
    }

    fn micro_step(&self) -> u32 {
        (self.step * self.plan.micro_per_step() as u64) as u32
    }

    fn state(&self, boundary: (u32, usize)) -> RestoreCheckpointState {
        RestoreCheckpointState {
            step: self.step,
            epoch: boundary.0,
            batches_consumed: boundary.1,
            scheduler_step: self.sched_step,
            lr: self.lr,
            lr_critic: self.lr_critic,
            nan_count: self.nan_count,
            losses: self.losses.map(|v| if v.is_finite() { v } else { 0.0 }),
            artifact: self.plan.artifact.kind().into(),
            ema_decay: self.plan.ema,
            seed: self.plan.seed,
            backend: "mlx".into(),
            state_identity: self.plan.state_identity(),
            data_identity: self.plan.data_identity(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            base_identity: self.base_identity.clone(),
        }
    }

    fn critic_tensors(&self) -> Params {
        let mut out = self.head.clone();
        for (prefix, (u, v)) in &self.sn {
            out.insert(Rc::from(format!("{prefix}.weight_u").as_str()), u.clone());
            out.insert(Rc::from(format!("{prefix}.weight_v").as_str()), v.clone());
        }
        out
    }

    fn save_checkpoint(&self, boundary: (u32, usize)) -> Result<()> {
        let io = |p: &std::path::Path, e: std::io::Error| {
            Error::Msg(format!("iris restore checkpoint: {}: {e}", p.display()))
        };
        std::fs::create_dir_all(&self.ckpt_root).map_err(|e| io(&self.ckpt_root, e))?;
        let staging = checkpoint_staging_dir(&self.ckpt_root, self.step);
        if staging.exists() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        std::fs::create_dir_all(&staging).map_err(|e| io(&staging, e))?;
        save_params(&staging.join(CKPT_TRAINABLE), &self.params)?;
        if let Some(e) = &self.ema {
            save_params(&staging.join(CKPT_EMA), e)?;
        }
        self.opt.save(&staging.join(CKPT_OPTIMIZER))?;
        save_params(&staging.join(CKPT_CRITIC), &self.critic_tensors())?;
        self.critic_opt.save(&staging.join(CKPT_CRITIC_OPTIMIZER))?;
        gen_core::iris::train::atomic_write(
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

    /// The full backbone of `which` (f32): the trained tensors of a full run, or the base with
    /// every adapter delta merged in.
    fn merged(&self, which: WeightsSelect) -> Result<Vec<(String, Array)>> {
        let tensors = self.selected(which);
        let mut out: Vec<(String, Array)> = match &self.model.trainable {
            Trainable::Full => tensors
                .iter()
                .map(|(k, v)| Ok((k.to_string(), v.as_dtype(Dtype::Float32)?)))
                .collect::<Result<_>>()?,
            Trainable::Adapter { kind, targets, .. } => {
                let mut base = load_source(&self.init_weights, None)?;
                for t in targets {
                    let key = format!("{}.weight", t.path);
                    let w = base.get(&key).ok_or_else(|| {
                        Error::Msg(format!("iris restore export: base lacks {key}"))
                    })?;
                    let delta = adapter_delta(*kind, t, tensors)?;
                    let merged = w.add(&delta)?;
                    eval([&merged])?;
                    base.insert(key, merged);
                }
                base.into_iter().collect()
            }
        };
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Write the `upscaler/`-layout restoration export; returns its `model.safetensors`.
    fn export(&self, req: &TrainingRequest) -> Result<PathBuf> {
        let dir = req.output_dir.join(file_stem(&req.file_name));
        let which = self.plan.export_weights;
        let dtype = if self.plan.export_bf16 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        let list = self
            .merged(which)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.as_dtype(dtype)?)))
            .collect::<Result<Vec<_>>>()?;
        let meta = export_metadata(which, self.step, &self.base_identity);
        // sc-25690: the sidecars and the merged weights appear together or not at all.
        publish_export_dir(&dir, |stage: &Path| -> Result<()> {
            write_export_sidecars(&self.export, stage)?;
            save_safetensors_atomic(
                &stage.join(gen_core::iris::BACKBONE_WEIGHTS_FILE),
                &list,
                &meta,
            )
        })?;
        Ok(dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE))
    }

    /// Restore the first item's (epoch-1) degraded input with the preview weights.
    fn render_preview(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pair = training_pair(&req.items[0], &self.plan, 1, 0)?;
        let batch = PairBatch::from_pairs(std::slice::from_ref(&pair));
        let dit = self.model.dit(self.selected(self.plan.preview_weights))?;
        let x = self.frozen.restore(&dit, &batch.lq)?;
        let x = mlx_rs::ops::clip(&x, (-1.0f32, 1.0f32))?;
        let image = crate::pipeline::to_image(&x)?;
        on_progress(TrainingProgress::Sample {
            step: self.micro_step(),
            index: 1,
            total: 1,
            prompt: format!("restoration preview: {}", req.items[0].image_path.display()),
            image,
        });
        mlx_rs::memory::clear_cache();
        Ok(())
    }
}

/// The `[out, in]` f32 delta of one adapter target (`(alpha/rank)·B·A`, or the LoKr Kronecker
/// reconstruction) — what the exported merge adds.
pub fn adapter_delta(kind: AdapterKind, t: &AdapterTarget, p: &Params) -> Result<Array> {
    let get = |n: &str| p.get(format!("{}.{n}", t.path).as_str());
    match kind {
        AdapterKind::Lora { rank, alpha } => {
            let a = get("lora_A.weight").ok_or_else(|| Error::Msg("lora_A missing".into()))?;
            let b = get("lora_B.weight").ok_or_else(|| Error::Msg("lora_B missing".into()))?;
            Ok(
                mlx_rs::ops::matmul(&b.as_dtype(Dtype::Float32)?, &a.as_dtype(Dtype::Float32)?)?
                    .multiply(Array::from_f32(alpha / rank as f32))?,
            )
        }
        AdapterKind::Lokr { rank, alpha, .. } => Ok(reconstruct_lokr_delta(
            alpha,
            rank as f32,
            &[t.out_f, t.in_f],
            get("lokr_w1"),
            None,
            None,
            get("lokr_w2"),
            get("lokr_w2_a"),
            get("lokr_w2_b"),
            Dtype::Float32,
        )?),
    }
}
