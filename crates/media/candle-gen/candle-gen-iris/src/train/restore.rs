//! Native Candle **Iris-3B restoration training** (story sc-25688) — the twin of
//! `mlx_gen_iris::train::restore`: the paper's one-step restoration recipe (following HYPIR) over
//! the backend-neutral contract [`gen_core::iris::restoration_train`]. Generator windows train
//! `x̂ = x_lq − σ·v` on `λ₂·mse + λ_lpips·LPIPS + λ_gan·D_G`, critic windows train the
//! spectral-normalized head on the frozen OpenCLIP ConvNeXt-XXL trunk; one of each is a step. The
//! checkpoint, resume, export (the `upscaler/` layout, adapters merged) and positional randomness
//! are the MLX twin's, key for key.
//!
//! Every hyper-parameter comes from the request. The documented default request is
//! [`recipe_training_config`] — the paper's **full** fine-tune, AdamW at lr 1e-5 with weight decay
//! 0.01 for generator and critic, constant LR; a LoRA/LoKr adapter (HYPIR's own arm) trains only
//! when the request asks for one.
//!
//! [`gen_core::iris::restoration_train`]: candle_gen::gen_core::iris::restoration_train
//! [`recipe_training_config`]: candle_gen::gen_core::iris::restoration_train::recipe_training_config

use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::iris::restoration::RestorationResources;
use candle_gen::gen_core::iris::restoration_train::{
    self as contract, base_identity, check_resume, critic_head_init, export_metadata, phase_of,
    read_checkpoint_state, training_pair, validate_item, write_export_sidecars, ConvNextConfig,
    LossWindow, Phase, RestoreArtifact, RestoreCheckpointState, RestoreTrainPlan,
    RestoreTrainResources, TrainPair, WindowClose, CKPT_CRITIC, CKPT_CRITIC_OPTIMIZER,
    CRITIC_HEAD_WIDTH, TRAINER_ID,
};
use candle_gen::gen_core::iris::train::{
    adapter_param_route, checkpoint_root, checkpoint_staging_dir, file_stem, full_param_route,
    latest_checkpoint, linear_paths_from_keys, lr_factor, mix_seed, prune_checkpoints,
    publish_checkpoint, DataWalk, HostRng, MixedPrecision, MuonAdjustLr, OptimizerKind,
    OptimizerPlan, ParamRoute, ResumeDataPolicy, WeightsSelect, CKPT_EMA, CKPT_OPTIMIZER,
    CKPT_STATE, CKPT_TRAINABLE, TECHNIQUES,
};
use candle_gen::gen_core::iris::FAMILY;
use candle_gen::gen_core::train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques, Trainer,
    TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality};
use candle_gen::train::lora::reconstruct_lokr_delta;
use candle_gen::{CandleError as Error, Result};

use super::critic::{
    head_from_host, head_from_tensors, multilevel_loss, power_iteration, ConvNextTrunk, Critic,
    SpectralState,
};
use super::lpips::Lpips;
use super::model::{
    fnv, init_adapter, provider_dtype, save_safetensors_atomic, AdapterKind, AdapterTarget,
    TrainModel, Trainable,
};
use super::optim::{
    clip_grads, ema_update, mul_s, save_tensors, snapshot, IrisOptimizer, Params, Tensors,
};
use super::{check_backbone_keys, load_source, load_tensors, vars};
use crate::dit::{IrisDiT, TextBatch};

const BACKEND: &str = "candle";

/// Identity + capabilities of the Candle Iris restoration trainer.
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

/// The frozen training-only networks of a run.
pub struct FrozenNets {
    pub lpips: Lpips,
    pub trunk: ConvNextTrunk,
    pub trunk_cfg: ConvNextConfig,
    pub head_width: usize,
}

/// Builds the frozen networks for a critic compute dtype on a device.
pub type NetsProvider = Box<dyn Fn(DType, &Device) -> Result<FrozenNets>>;

/// The resolved resources of a restoration training load.
pub struct IrisRestoreTrainer {
    descriptor: TrainerDescriptor,
    res: RestoreTrainResources,
    nets: NetsProvider,
    device: Device,
}

impl IrisRestoreTrainer {
    /// A trainer whose frozen networks come from `nets` (the miniature lifecycle tests).
    pub fn with_networks(res: RestoreTrainResources, nets: NetsProvider, device: Device) -> Self {
        Self {
            descriptor: trainer_descriptor(),
            res,
            nets,
            device,
        }
    }
}

fn pinned_networks(res: &RestoreTrainResources) -> NetsProvider {
    let (trunk, vgg, lin) = (
        res.critic_trunk.clone(),
        res.vgg16.clone(),
        res.lpips_lin.clone(),
    );
    Box::new(move |compute, device| {
        let trunk_cfg = ConvNextConfig::xxlarge();
        Ok(FrozenNets {
            trunk: ConvNextTrunk::load(&trunk, &trunk_cfg, compute, device)?,
            lpips: Lpips::load(&vgg, &lin, device)?,
            trunk_cfg,
            head_width: CRITIC_HEAD_WIDTH,
        })
    })
}

/// Construct the trainer: `spec.weights` = the restoration export directory + the training-only
/// components ([`RestoreTrainResources::from_spec`]). Nothing heavy loads here.
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
    Ok(Box::new(IrisRestoreTrainer::with_networks(
        res,
        nets,
        candle_gen::default_device()?,
    )))
}

candle_gen::register_trainer! {
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

/// The frozen networks and fixed tensors of a run.
pub struct Frozen {
    pub lpips: Lpips,
    pub critic: Critic,
    /// `[1, T, L, D]` empty-prompt states.
    pub states: Tensor,
    pub mask: Vec<i32>,
    pub model_time: f32,
    pub sigma: f32,
    pub device: Device,
}

/// One micro-batch.
pub struct PairBatch {
    pub lq: Tensor,
    pub gt: Tensor,
}

impl PairBatch {
    pub fn from_pairs(pairs: &[TrainPair], device: &Device) -> Result<Self> {
        let s = pairs[0].size;
        let b = pairs.len();
        let lq: Vec<f32> = pairs.iter().flat_map(|p| p.lq.iter().copied()).collect();
        let gt: Vec<f32> = pairs.iter().flat_map(|p| p.gt.iter().copied()).collect();
        Ok(Self {
            lq: Tensor::from_vec(lq, (b, 3, s, s), device)?,
            gt: Tensor::from_vec(gt, (b, 3, s, s), device)?,
        })
    }
}

impl Frozen {
    /// `x̂ = x_lq − σ·v(x_lq, σ·N, ∅)` (f32).
    pub fn restore(&self, dit: &IrisDiT, lq: &Tensor) -> Result<Tensor> {
        let b = lq.dim(0)?;
        let states = if b == 1 {
            self.states.clone()
        } else {
            Tensor::cat(&vec![self.states.clone(); b], 0)?
        };
        let masks = vec![self.mask.clone(); b];
        let t = Tensor::from_vec(vec![self.model_time; b], b, &self.device)?;
        let v = dit
            .forward(
                lq,
                &t,
                &TextBatch {
                    states: &states,
                    mask: &masks,
                },
            )?
            .to_dtype(DType::F32)?;
        Ok(lq.sub(&mul_s(&v, self.sigma as f64)?)?)
    }
}

/// The weighted Eq. 6 terms on `x̂` (critic in eval mode), each a scalar tensor.
pub fn generator_terms(
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Tensors,
    sn: &SpectralState,
    xhat: &Tensor,
    gt: &Tensor,
) -> Result<[Tensor; 3]> {
    let w = plan.weights;
    let zero = Tensor::new(0f32, &frozen.device)?;
    let l2 = mul_s(&xhat.sub(gt)?.sqr()?.mean_all()?, w.l2)?;
    let lpips = if w.lpips != 0.0 {
        mul_s(&frozen.lpips.distance(xhat, gt)?.mean_all()?, w.lpips)?
    } else {
        zero.clone()
    };
    let gan = if w.gan != 0.0 {
        mul_s(
            &frozen
                .critic
                .generator_term(head, sn, xhat, plan.gan_real_label)?
                .mean_all()?,
            w.gan,
        )?
    } else {
        zero
    };
    Ok([l2, lpips, gan])
}

/// One generator micro-batch: `[total, l2, lpips, gan]` and the gradient of `total · grad_scale`
/// over the trainable tensors (zero for a tensor the graph never reads).
#[allow(clippy::too_many_arguments)]
pub fn generator_grads(
    model: &TrainModel,
    params: &Params,
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Tensors,
    sn: &SpectralState,
    batch: &PairBatch,
    grad_scale: f32,
) -> Result<([f64; 4], Tensors)> {
    let dit = model.dit(&snapshot(params))?;
    let xhat = frozen.restore(&dit, &batch.lq)?;
    let [l2, lp, gan] = generator_terms(frozen, plan, head, sn, &xhat, &batch.gt)?;
    let total = l2.add(&lp)?.add(&gan)?;
    let store = mul_s(&total, grad_scale as f64)?.backward()?;
    drop(dit);
    let mut grads = Tensors::new();
    for (k, v) in params {
        let g = match store.get(v.as_tensor()) {
            Some(g) => g.detach(),
            None => v.as_tensor().zeros_like()?.detach(),
        };
        grads.insert(k.clone(), g);
    }
    let h = |t: &Tensor| -> Result<f64> { Ok(t.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64) };
    Ok(([h(&total)?, h(&l2)?, h(&lp)?, h(&gan)?], grads))
}

/// One critic micro-batch: two power iterations (`D(y)` then `D(x̂)`), the loss, the gradient of
/// `loss · grad_scale` over the head, and the advanced spectral state.
pub fn critic_grads(
    frozen: &Frozen,
    plan: &RestoreTrainPlan,
    head: &Params,
    sn: &SpectralState,
    xhat: &Tensor,
    gt: &Tensor,
    grad_scale: f32,
) -> Result<(f64, Tensors, SpectralState)> {
    let cfg = &frozen.critic.head;
    let cur = snapshot(head);
    let sn1 = power_iteration(cfg, &cur, sn)?;
    let sn2 = power_iteration(cfg, &cur, &sn1)?;
    let real = multilevel_loss(&frozen.critic.logits(&cur, &sn1, gt)?, plan.gan_real_label)?;
    let fake = multilevel_loss(&frozen.critic.logits(&cur, &sn2, &xhat.detach())?, 0.0)?;
    let loss = real.mean_all()?.add(&fake.mean_all()?)?;
    let store = mul_s(&loss, grad_scale as f64)?.backward()?;
    let mut grads = Tensors::new();
    for (k, v) in head {
        let g = match store.get(v.as_tensor()) {
            Some(g) => g.detach(),
            None => v.as_tensor().zeros_like()?.detach(),
        };
        grads.insert(k.clone(), g);
    }
    Ok((loss.to_vec0::<f32>()? as f64, grads, sn2))
}

#[derive(Default)]
struct Accum {
    grads: Option<Tensors>,
}

impl Accum {
    fn add(&mut self, g: Tensors) -> Result<()> {
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

/// The frozen networks, conditioning and fresh critic head of a run.
fn build_frozen(
    res: &RestoreTrainResources,
    plan: &RestoreTrainPlan,
    nets: &dyn Fn(DType, &Device) -> Result<FrozenNets>,
    device: &Device,
) -> Result<(Frozen, Params, SpectralState)> {
    let critic_compute = if plan.critic_bf16 {
        if device.is_cpu() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: critic_dtype bf16 needs a GPU device (Candle's CPU backend \
                 has no bf16 GEMM) — set model_options.{}.critic_dtype = \"fp32\"",
                contract::OPTIONS_KEY
            )));
        }
        DType::BF16
    } else {
        DType::F32
    };
    let FrozenNets {
        lpips,
        trunk,
        trunk_cfg,
        head_width,
    } = nets(critic_compute, device)?;
    let head_cfg = trunk_cfg.head(head_width);
    let prompt = res.export.empty_prompt()?;
    let states = Tensor::from_vec(prompt.embeddings.clone(), prompt.shape.to_vec(), device)?;
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
        device: device.clone(),
    };
    let (head, sn) = head_from_host(&head_cfg, &critic_head_init(&head_cfg, plan.seed), device)?;
    Ok((frozen, head, sn))
}

/// The pieces of a full-fine-tune run at its initial state (tests driving the step functions).
pub fn miniature_parts(
    res: &RestoreTrainResources,
    plan: &RestoreTrainPlan,
    nets: NetsProvider,
    device: &Device,
) -> Result<(TrainModel, Params, Frozen, Params, SpectralState)> {
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
    let config = res.export.config();
    let source = load_source(&res.init_weights(plan.init), None, device)?;
    check_backbone_keys(&config.model, &source, "the generator weights")?;
    let (frozen, head, sn) = build_frozen(res, plan, nets.as_ref(), device)?;
    Ok((
        TrainModel {
            cfg: config.model.clone(),
            compute,
            device: device.clone(),
            trainable: Trainable::Full,
        },
        vars(source)?,
        frozen,
        head,
        sn,
    ))
}

/// PEFT `init_lora_weights="gaussian"`: `A ~ N(0, (1/r)²)` from the host stream (backend-portable).
fn gaussian_lora(
    params: &mut Params,
    targets: &[AdapterTarget],
    rank: usize,
    seed: u64,
    device: &Device,
) -> Result<()> {
    for t in targets {
        let k = format!("{}.lora_A.weight", t.path);
        let mut rng = HostRng::new(mix_seed(&[seed, fnv(&k)]));
        let data: Vec<f32> = rng
            .normals_f32(rank * t.in_f)
            .into_iter()
            .map(|v| v / rank as f32)
            .collect();
        params.insert(
            k,
            Var::from_tensor(&Tensor::from_vec(data, (rank, t.in_f), device)?)?,
        );
    }
    Ok(())
}

struct Run {
    plan: RestoreTrainPlan,
    model: TrainModel,
    params: Params,
    ema: Option<Tensors>,
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
    export: RestorationResources,
    init_weights: PathBuf,
    device: Device,
    step: u64,
    sched_step: u64,
    lr: f64,
    lr_critic: f64,
    nan_count: u32,
    epoch: u32,
    pos: usize,
    losses: [f64; 5],
}

impl Run {
    fn prepare(
        trainer: &IrisRestoreTrainer,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let res = &trainer.res;
        let device = trainer.device.clone();
        let config = res.export.config().clone();
        let pre = trainer.plan(req, &[])?;
        let compute = match pre.mixed_precision {
            MixedPrecision::Bf16 => DType::BF16,
            MixedPrecision::Fp32 => DType::F32,
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
        let source = load_source(&init_weights, adapter_compute, &device)?;
        check_backbone_keys(&config.model, &source, "the generator weights")?;
        let shapes = candle_gen::gen_core::iris::train::backbone_tensor_shapes(&config.model);
        let linear_paths =
            linear_paths_from_keys(shapes.iter().map(|(k, s)| (k.as_str(), s.len())));
        let plan = trainer.plan(req, &linear_paths)?;
        let base_identity = base_identity(&res.export, &init_weights)?;

        let (model, mut params) = match &plan.artifact {
            RestoreArtifact::Full => (
                TrainModel {
                    cfg: config.model.clone(),
                    compute,
                    device: device.clone(),
                    trainable: Trainable::Full,
                },
                vars(source)?,
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
                            out_f: w.dims()[0],
                            in_f: w.dims()[1],
                        }
                    })
                    .collect();
                let mut base = HashMap::with_capacity(source.len());
                for (k, a) in source {
                    let a = provider_dtype(&k, &a, compute)?;
                    base.insert(k, a);
                }
                let mut params = init_adapter(kind, &targets, plan.seed, &device)?;
                if let (AdapterKind::Lora { rank, .. }, true) = (kind, plan.lora_gaussian_init) {
                    gaussian_lora(&mut params, &targets, rank, plan.seed, &device)?;
                }
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
        let mut ema = plan.ema.map(|_| snapshot_detached(&params));

        let (frozen, mut head, mut sn) = build_frozen(res, &plan, trainer.nets.as_ref(), &device)?;
        let head_cfg = frozen.critic.head;
        let c = &plan.critic;
        let mut critic_opt =
            IrisOptimizer::new(&adamw_plan(c.lr, c.betas, c.weight_decay), &head, |_, _| {
                ParamRoute::AdamW
            })?;

        let walk = DataWalk::new(req.items.len(), plan.batch_size)?;
        if let Some(w) = walk.unused_tail_warning() {
            eprintln!("{w}");
        }
        let total_steps = candle_gen::gen_core::iris::train::total_optimizer_steps(
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
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let restored = load_tensors(&dir.join(CKPT_TRAINABLE), &device)?;
            if restored.len() != params.len() || params.keys().any(|k| !restored.contains_key(k)) {
                return Err(Error::Msg(
                    "iris restore resume: the checkpoint's trainable tensors do not match this run"
                        .into(),
                ));
            }
            for (k, v) in &params {
                v.set(&restored[k])?;
            }
            if let Some(e) = ema.as_mut() {
                *e = load_tensors(&dir.join(CKPT_EMA), &device)?
                    .into_iter()
                    .collect();
            }
            opt.load_state(&load_tensors(&dir.join(CKPT_OPTIMIZER), &device)?)?;
            let critic = load_tensors(&dir.join(CKPT_CRITIC), &device)?;
            (head, sn) = head_from_tensors(&head_cfg, |k| critic.get(k).cloned())?;
            critic_opt.load_state(&load_tensors(&dir.join(CKPT_CRITIC_OPTIMIZER), &device)?)?;
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
        let _ = &mut params;
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
            device,
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
        PairBatch::from_pairs(&pairs, &self.device)
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
                if req.cancel.is_cancelled() && batch_count.is_multiple_of(per_step) {
                    cancelled = true;
                    break;
                }
                let batch = self.batch(req, epoch, pos)?;
                let (phase, sync) = phase_of(batch_count, accum);
                batch_count += 1;
                match phase {
                    Phase::Generator => {
                        let head = snapshot_detached(&self.head);
                        let (terms, grads) = generator_grads(
                            &self.model,
                            &self.params,
                            &self.frozen,
                            &plan,
                            &head,
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
                                let g: Tensors = g
                                    .into_iter()
                                    .filter(|(k, _)| self.opt.is_trained(k))
                                    .collect();
                                let (_, g) = clip_grads(g, plan.generator.clip)?;
                                self.opt.step(&self.params, &g, lr)?;
                                if let (Some(e), Some(d)) = (self.ema.as_mut(), plan.ema) {
                                    ema_update(e, &self.params, d)?;
                                }
                                self.losses[..4].copy_from_slice(&g_sums);
                            }
                            g_sums = [0.0; 4];
                        }
                    }
                    Phase::Critic => {
                        let dit = self.model.dit(&snapshot_detached(&self.params))?;
                        let xhat = self.frozen.restore(&dit, &batch.lq)?.detach();
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
                                        self.critic_opt.step(&self.head, &g, lr)?;
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
            backend: BACKEND.into(),
            state_identity: self.plan.state_identity(),
            data_identity: self.plan.data_identity(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            base_identity: self.base_identity.clone(),
        }
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
        let list = |m: &Tensors| -> Vec<(String, Tensor)> {
            m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        save_tensors(
            &staging.join(CKPT_TRAINABLE),
            &list(&snapshot_detached(&self.params)),
        )?;
        if let Some(e) = &self.ema {
            save_tensors(&staging.join(CKPT_EMA), &list(e))?;
        }
        save_tensors(&staging.join(CKPT_OPTIMIZER), &self.opt.state_tensors()?)?;
        let mut critic = list(&snapshot_detached(&self.head));
        for (prefix, (u, v)) in &self.sn {
            critic.push((format!("{prefix}.weight_u"), u.clone()));
            critic.push((format!("{prefix}.weight_v"), v.clone()));
        }
        save_tensors(&staging.join(CKPT_CRITIC), &critic)?;
        save_tensors(
            &staging.join(CKPT_CRITIC_OPTIMIZER),
            &self.critic_opt.state_tensors()?,
        )?;
        candle_gen::gen_core::iris::train::atomic_write(
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

    fn selected(&self, which: WeightsSelect) -> Tensors {
        match (which, &self.ema) {
            (WeightsSelect::Ema, Some(e)) => e.clone(),
            _ => snapshot_detached(&self.params),
        }
    }

    fn merged(&self, which: WeightsSelect) -> Result<Vec<(String, Tensor)>> {
        let tensors = self.selected(which);
        let mut out: Vec<(String, Tensor)> = match &self.model.trainable {
            Trainable::Full => tensors
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
                .collect::<Result<_>>()?,
            Trainable::Adapter { kind, targets, .. } => {
                let mut base = load_source(&self.init_weights, None, &self.device)?;
                for t in targets {
                    let key = format!("{}.weight", t.path);
                    let w = base.get(&key).ok_or_else(|| {
                        Error::Msg(format!("iris restore export: base lacks {key}"))
                    })?;
                    let merged = w.add(&adapter_delta(*kind, t, &tensors)?)?;
                    base.insert(key, merged);
                }
                base.into_iter().collect()
            }
        };
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn export(&self, req: &TrainingRequest) -> Result<PathBuf> {
        let dir = req.output_dir.join(file_stem(&req.file_name));
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::Msg(format!("iris restore export: {}: {e}", dir.display())))?;
        write_export_sidecars(&self.export, &dir)?;
        let which = self.plan.export_weights;
        let dtype = if self.plan.export_bf16 {
            DType::BF16
        } else {
            DType::F32
        };
        let list = self
            .merged(which)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_dtype(dtype)?.to_device(&Device::Cpu)?)))
            .collect::<Result<Vec<_>>>()?;
        let path = dir.join(candle_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
        save_safetensors_atomic(
            &path,
            &list,
            &export_metadata(which, self.step, &self.base_identity),
        )?;
        Ok(path)
    }

    fn render_preview(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pair = training_pair(&req.items[0], &self.plan, 1, 0)?;
        let batch = PairBatch::from_pairs(std::slice::from_ref(&pair), &self.device)?;
        let dit = self.model.dit(&self.selected(self.plan.preview_weights))?;
        let x = self.frozen.restore(&dit, &batch.lq)?.clamp(-1f32, 1f32)?;
        let image = crate::pipeline::to_image(&x)?;
        on_progress(TrainingProgress::Sample {
            step: self.micro_step(),
            index: 1,
            total: 1,
            prompt: format!("restoration preview: {}", req.items[0].image_path.display()),
            image,
        });
        Ok(())
    }
}

/// Detached copies of every `Var` (plain data, off the autograd tape).
fn snapshot_detached(params: &Params) -> Tensors {
    params
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect()
}

/// The `[out, in]` f32 delta of one adapter target — what the exported merge adds.
pub fn adapter_delta(kind: AdapterKind, t: &AdapterTarget, p: &Tensors) -> Result<Tensor> {
    let get = |n: &str| p.get(&format!("{}.{n}", t.path));
    match kind {
        AdapterKind::Lora { rank, alpha } => {
            let a = get("lora_A.weight").ok_or_else(|| Error::Msg("lora_A missing".into()))?;
            let b = get("lora_B.weight").ok_or_else(|| Error::Msg("lora_B missing".into()))?;
            mul_s(
                &b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?,
                (alpha / rank as f32) as f64,
            )
        }
        AdapterKind::Lokr { rank, alpha, .. } => reconstruct_lokr_delta(
            get("lokr_w1"),
            None,
            None,
            get("lokr_w2"),
            get("lokr_w2_a"),
            get("lokr_w2_b"),
            alpha,
            rank as f32,
            1.0,
            (t.out_f, t.in_f),
        ),
    }
}
