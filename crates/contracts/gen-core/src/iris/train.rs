//! Backend-neutral **training contract** of the Iris-3B generation task (epic sc-25678, story
//! sc-25685): everything about an Iris training run that is not a tensor — the request surface
//! (the shared [`TrainingRequest`] plus the Iris options bag), the resolved run plan, the
//! positional randomness, the rectified-flow training schedule, the data walk and image
//! preprocessing, the Muon/AdamW parameter routing, the random-init law, the adapter and
//! full-model artifact schemas, and the durable checkpoint layout.
//!
//! The MLX trainer (`mlx-gen-iris`) and the Candle trainer read this one definition, so a run is
//! described — and resumes, exports and refuses — identically on both backends. The SceneWorks
//! worker maps its training plan onto [`TrainingRequest`] and the [`OPTIONS_KEY`] options bag.
//!
//! Frozen source: `speridlabs/iris-3b` @ [`super::UPSTREAM_CODE_REVISION`] — `train/trainer.py`
//! (loop, accumulation, clipping, EMA-before-step, positional seeding, data position),
//! `flow/transport.py` + `flow/schedule.py` + `flow/timesteps.py` (objective and sampling),
//! `train/optim.py` (Dion Muon + AdamW routing, `scale_lr`), `train/lr.py`, `train/ema.py`,
//! `train/ckpt.py` (atomic write, retention), `data/datasets.py` + `data/samplers.py`
//! (preprocessing, the ranged walk), `seeding.py` (`mix_seed`), `models/dit.py`
//! (`initialize_weights`), `scripts/export_checkpoint.py` (the full-model export layout).
//!
//! ## Unit conventions (read before wiring a request)
//!
//! * The shared [`TrainingConfig`] units hold: [`TrainingConfig::steps`],
//!   [`TrainingConfig::save_every`], [`TrainingConfig::sample_every`] and
//!   [`TrainingConfig::lr_warmup_steps`] count **micro-steps**, and every `TrainingProgress` /
//!   `TrainingOutput` step is a micro-step. Upstream counts optimizer steps (`train.max_steps`,
//!   `save_every_steps`, `sample_every_steps`, `warmup_steps`), one per `gradient_accumulation`
//!   micro-batches, so [`IrisTrainPlan::resolve`] divides by the accumulation length: `steps`,
//!   `save_every` and `sample_every` must be multiples of it (refused otherwise — never rounded),
//!   and the warmup rounds up (as the shared `schedule_updates` does). With
//!   `gradient_accumulation = 1` the two units coincide. The Iris-only `milestone_steps` /
//!   `keep_last_checkpoints` options and the checkpoint `state.json` `step` count optimizer steps
//!   (upstream's checkpoint names).
//! * Previews follow the shared contract: empty [`TrainingConfig::sample_prompts`] (or
//!   `sample_every = 0`) renders nothing; at most [`PREVIEW_PROMPT_CAP`] prompts render per cadence;
//!   no extra render at step 1. Upstream's seven default validation prompts are an explicit opt-in
//!   (`upstream_validation_prompts`).
//! * [`TrainingConfig::resolution`] is upstream's fixed-square `data.image_size` (shortest-side
//!   bicubic resize + center crop); it must be a multiple of the backbone's `patch_size` and is
//!   never silently changed.
//! * `TrainingConfig::network_type` + `full_finetune` choose the artifact: a full model
//!   (random-init or weights-init), a LoRA or a LoKr adapter.
//!
//! Everything else upstream exposes for a single-device run travels in
//! `TrainingConfig::model_options["iris"]` ([`IrisTrainOptions`]); an unknown key there is a
//! refusal, never ignored.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{json, Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};

use super::{IrisConfig, ModelConfig, GENERATION_MODEL_ID};
use crate::train::{
    LrSchedule, NetworkType, TrainingConfig, TrainingItem, TrainingRequest, TrainingTechniques,
};
use crate::{Error, Result};

/// Registry id of the Iris generation trainer (the same id as the generator it trains for).
pub const TRAINER_ID: &str = GENERATION_MODEL_ID;
/// `TrainingConfig::model_options` key carrying the [`IrisTrainOptions`] object.
pub const OPTIONS_KEY: &str = "iris";
/// `TrainingItem::model_options` key carrying an item's named captions (`{field: text}`), the
/// per-sample `info` dict upstream's `select_caption` reads.
pub const ITEM_CAPTIONS_KEY: &str = "captions";
/// The optional techniques the Iris trainer implements: none of the epic-2123 knobs.
pub const TECHNIQUES: TrainingTechniques = TrainingTechniques::NONE;
/// `data/samplers.py` `CANONICAL_CHUNKS`: the ranged walk covers `chunks · (n / chunks)` samples.
pub const CANONICAL_CHUNKS: usize = 640;
/// `TrainConfig.validation_prompts` default (`DEFAULT_VALIDATION_PROMPTS`): the preview prompts a
/// run renders only when the request names none **and** opts in with the
/// `upstream_validation_prompts` option (they are not capped by [`PREVIEW_PROMPT_CAP`]).
pub const DEFAULT_VALIDATION_PROMPTS: [&str; 7] = [
    "a golden retriever puppy sitting in a field of tall grass at sunset",
    "close-up portrait of an elderly fisherman with a weathered face, soft window light, shallow depth of field",
    "a cozy reading nook with a window seat and rain on the glass, warm lamp light, watercolor illustration",
    "an astronaut tending a vegetable garden inside a glass dome on the moon, cinematic lighting",
    "a snow-covered mountain village at dawn with smoke rising from the chimneys, highly detailed",
    "a neon sign above a small night-market stall that reads \u{201c}OPEN LATE\u{201d}",
    "a hand-painted wooden sign in a flower shop window that says \u{201c}Fresh Tulips Today\u{201d}",
];

/// The most request prompts a preview cadence renders (the shared family cap; extra prompts are
/// not rendered, as in every other family trainer).
pub const PREVIEW_PROMPT_CAP: usize = 4;

// =============================================================================================
// Positional randomness (`seeding.py`)
// =============================================================================================

/// `iris3b.seeding.mix_seed`: fold integers into one 63-bit seed (splitmix64 finalizer per part).
/// Every stochastic draw of a run is keyed on `(seed, rank, epoch, position)` through it, so a
/// resumed run continues the unbroken draw sequence exactly.
pub fn mix_seed(parts: &[u64]) -> u64 {
    let mut h: u64 = 0;
    for &p in parts {
        h = h.wrapping_add(p).wrapping_add(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
    }
    h & 0x7FFF_FFFF_FFFF_FFFF
}

/// The positional draw stream of one key: a SplitMix64 generator with Box–Muller normals. Host-side
/// and backend-independent, so the MLX and Candle trainers draw the **same** dropout masks,
/// timesteps, noise and caption choices for the same `(seed, epoch, position)` (upstream draws
/// these from `torch.manual_seed(mix_seed(...))`; the torch stream itself is not reproduced).
#[derive(Clone, Debug)]
pub struct HostRng {
    state: u64,
    spare: Option<f64>,
}

impl HostRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box–Muller; the second value of each pair is kept for the next call).
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // u1 in (0, 1] so the log is finite.
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }

    /// `n` standard normals as f32.
    pub fn normals_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal() as f32).collect()
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

/// The key of micro-batch `position` (0-based within its epoch) of `epoch` (1-based) —
/// `torch.manual_seed(mix_seed(train.seed, rank, epoch, position))` upstream; single device, so
/// rank is 0.
pub fn batch_seed(seed: u64, epoch: u32, position: usize) -> u64 {
    mix_seed(&[seed, 0, epoch as u64, position as u64])
}

/// The caption-choice key of dataset `index` in `epoch` (`_caption_rng`: `mix_seed(seed, epoch,
/// idx)`).
pub fn caption_seed(seed: u64, epoch: u32, index: usize) -> u64 {
    mix_seed(&[seed, epoch as u64, index as u64])
}

/// The draws of one micro-batch, in upstream's order: the CFG-dropout mask (only when
/// `text_dropout > 0`), the timestep indices, then the noise (`randn_like(x0)`, `[B, C, H, W]`
/// row-major).
#[derive(Clone, Debug, PartialEq)]
pub struct BatchDraws {
    pub drop: Vec<bool>,
    pub timestep_idx: Vec<usize>,
    pub noise: Vec<f32>,
}

/// Draw one micro-batch's randomness from its positional key.
pub fn draw_batch(
    key: u64,
    batch: usize,
    sample_numel: usize,
    text_dropout: f64,
    sampler: TimestepSampler,
    num_timesteps: usize,
) -> BatchDraws {
    let mut rng = HostRng::new(key);
    let drop = if text_dropout > 0.0 {
        (0..batch).map(|_| rng.uniform() < text_dropout).collect()
    } else {
        vec![false; batch]
    };
    let timestep_idx = (0..batch)
        .map(|_| sampler.sample(&mut rng, num_timesteps))
        .collect();
    let noise = rng.normals_f32(batch * sample_numel);
    BatchDraws {
        drop,
        timestep_idx,
        noise,
    }
}

// =============================================================================================
// Rectified-flow objective (`flow/transport.py`, `flow/schedule.py`, `flow/timesteps.py`)
// =============================================================================================

/// `flow.prediction` is the shared [`super::Prediction`] (`Velocity` = `v`, the release; `Clean` =
/// `x`: the loss stays in velocity space via `v̂ = (x_t − x̂0) / max(σ, x_pred_sigma_min)`).
pub use super::Prediction;

/// Parse upstream's `flow.prediction` spelling, refusing anything else by name.
pub fn parse_prediction(s: &str) -> Result<Prediction> {
    Prediction::from_name(s).ok_or_else(|| {
        Error::Msg(format!(
            "iris training: flow.prediction must be v or x, got {s:?}"
        ))
    })
}

/// Upstream's spelling of a prediction type (`v` / `x`).
pub fn prediction_name(p: Prediction) -> &'static str {
    match p {
        Prediction::Velocity => "v",
        Prediction::Clean => "x",
    }
}

/// `flow.timestep_sampler`: the training density over the 1000-point index grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimestepSampler {
    /// `idx = floor(sigmoid(N(mean, std)) · N)`, clamped to `N − 1` (SD3 logit-normal).
    LogitNormal { mean: f64, std: f64 },
    /// `idx ~ U{0, …, N − 1}`.
    Uniform,
}

impl TimestepSampler {
    pub fn sample(self, rng: &mut HostRng, n: usize) -> usize {
        match self {
            TimestepSampler::LogitNormal { mean, std } => {
                let z = mean + std * rng.normal();
                let u = 1.0 / (1.0 + (-z).exp());
                ((u * n as f64) as usize).min(n - 1)
            }
            TimestepSampler::Uniform => rng.below(n),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            TimestepSampler::LogitNormal { .. } => "logit_normal",
            TimestepSampler::Uniform => "uniform",
        }
    }
}

/// `flow.shift_law` (`resolution_shift`): resolve the stage's shift from its image token count.
pub fn resolution_shift(
    tokens: usize,
    law: &str,
    base_shift: f64,
    base_tokens: usize,
) -> Result<f64> {
    match law {
        "none" => Ok(base_shift),
        _ if tokens == 0 => Err(Error::Msg(
            "iris training: token count must be positive".into(),
        )),
        "sd3" => Ok(base_shift * (tokens as f64 / base_tokens as f64).sqrt()),
        "flux" => {
            let slope = (1.15 - 0.5) / (4096.0 - 256.0);
            let intercept = 0.5 - slope * 256.0;
            Ok((slope * tokens as f64 + intercept).exp())
        }
        other => Err(Error::Msg(format!(
            "iris training: unknown flow.shift_law {other:?} (none | sd3 | flux)"
        ))),
    }
}

/// `FlowSchedule(num_timesteps, shift)`: the ascending training grid `σ' = shift(1 − linspace(1,
/// 0.001, N))` (f64, cast to f32) and the integer-truncated model times `int64(1000·σ')`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSchedule {
    pub sigmas: Vec<f32>,
    pub model_times: Vec<f32>,
}

impl TrainSchedule {
    pub fn new(num_timesteps: usize, shift: f64) -> Self {
        let n = num_timesteps;
        let (start, end) = (1.0f64, 0.001f64);
        let step = if n > 1 {
            (end - start) / (n - 1) as f64
        } else {
            0.0
        };
        let mut sigmas = Vec::with_capacity(n);
        let mut model_times = Vec::with_capacity(n);
        for i in 0..n {
            // torch.linspace's symmetric evaluation (see `super::time_grid`).
            let lin = if n == 1 {
                start
            } else if i < n / 2 {
                start + step * i as f64
            } else {
                end - step * (n - 1 - i) as f64
            };
            let s = super::shift_sigma(1.0 - lin, shift);
            sigmas.push(s as f32);
            model_times.push((s * n as f64).trunc() as i64 as f32);
        }
        Self {
            sigmas,
            model_times,
        }
    }
}

/// The resolved flow objective of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowObjective {
    pub num_train_timesteps: usize,
    /// The resolved stage shift (after `shift_law`).
    pub shift: f64,
    /// The configured base shift and law (recorded in the exported config).
    pub base_shift: f64,
    pub shift_law: String,
    pub shift_base_tokens: usize,
    pub sampler: TimestepSampler,
    pub prediction: Prediction,
    pub x_pred_sigma_min: f64,
}

// =============================================================================================
// Optimizer, schedule, EMA
// =============================================================================================

/// `train.optimizer.name`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// `torch.optim.AdamW` over every trainable tensor.
    AdamW,
    /// Dion `Muon` (pinned `microsoft/dion@58d38adb`) on hidden matrices, AdamW (same betas, eps
    /// `1e-8`) on embeddings, heads, vectors and the boundary matrices (`build_muon_param_groups`).
    Muon,
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OptimizerKind::AdamW => "adamw",
            OptimizerKind::Muon => "muon",
        }
    }
}

/// `train.optimizer.muon_adjust_lr` (Dion `adjust_lr`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuonAdjustLr {
    /// `lr · 0.2 · √max(fan_out, fan_in)` (the release).
    RmsNorm,
    /// `lr · √(fan_out / fan_in)`.
    SpectralNorm,
    /// `lr`.
    None,
}

impl MuonAdjustLr {
    /// The adjusted learning-rate ratio for an `[rows, cols]` matrix (`adjust_lr_*` with `lr = 1`).
    pub fn ratio(self, rows: usize, cols: usize) -> f64 {
        match self {
            MuonAdjustLr::RmsNorm => 0.2 * (rows.max(cols) as f64).sqrt(),
            MuonAdjustLr::SpectralNorm => (rows as f64 / cols as f64).sqrt(),
            MuonAdjustLr::None => 1.0,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            MuonAdjustLr::RmsNorm => "rms_norm",
            MuonAdjustLr::SpectralNorm => "spectral_norm",
            MuonAdjustLr::None => "none",
        }
    }
}

/// Per-block learning-rate corrections of a row-split Muon matrix (`compute_split_lr_scales`):
/// each block sees the adjustment its own shape would get as a separate parameter.
pub fn muon_split_scales(adjust: MuonAdjustLr, splits: &[usize], cols: usize) -> Vec<f64> {
    let rows: usize = splits.iter().sum();
    let full = adjust.ratio(rows, cols);
    splits
        .iter()
        .map(|&r| adjust.ratio(r, cols) / full)
        .collect()
}

/// The Newton–Schulz quintic coefficients Dion's Muon iterates (`zeropower_via_newtonschulz5` /
/// `newton_schulz_triton`, identical): five `(a, b, c)` steps of `X ← aX + (bA + cA²)X`,
/// `A = XXᵀ`, in bf16, after `X ← X / (‖X‖_F + ε)` with the wide orientation.
pub const NEWTON_SCHULZ_COEFFS: [(f32, f32, f32); 5] = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
];
/// The `epsilon` upstream passes to Dion (`epsilon=1.0e-8`) — the NS normalisation and the AdamW
/// groups' `eps`.
pub const MUON_EPSILON: f64 = 1.0e-8;
/// `torch.optim.AdamW`'s `eps` for the plain AdamW optimizer.
pub const ADAMW_EPSILON: f64 = 1.0e-8;

/// The resolved optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerPlan {
    pub kind: OptimizerKind,
    /// The configured `optimizer.lr`.
    pub base_lr: f64,
    /// `scale_lr(lr, auto_lr, effective_batch, base_batch_size)` — the rate the schedule scales.
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
    pub muon_momentum: f64,
    pub muon_nesterov: bool,
    pub muon_adjust_lr: MuonAdjustLr,
    pub auto_lr: String,
    pub base_batch_size: usize,
}

/// `scale_lr`: rescale by the effective/base batch ratio.
pub fn scale_lr(lr: f64, rule: &str, effective_batch: usize, base_batch: usize) -> Result<f64> {
    let ratio = effective_batch as f64 / base_batch as f64;
    match rule {
        "" | "none" => Ok(lr),
        "sqrt" => Ok(lr * ratio.sqrt()),
        "linear" => Ok(lr * ratio),
        other => Err(Error::Msg(format!(
            "iris training: unknown auto_lr rule {other:?} (none | sqrt | linear)"
        ))),
    }
}

/// `train.optimizer.schedule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrShape {
    Constant,
    Cosine,
}

/// The `LambdaLR` factor of `train/lr.py` at scheduler step `step` (the number of optimizer steps
/// already taken): linear `step / warmup` ramp from **0**, then constant 1 or a half-cosine to 0 at
/// `total`. The LR used by optimizer step `n` (1-based) is `lr · factor(n − 1)`.
pub fn lr_factor(shape: LrShape, step: u64, warmup: u64, total: u64) -> f64 {
    if step < warmup {
        return step as f64 / (warmup.max(1)) as f64;
    }
    match shape {
        LrShape::Constant => 1.0,
        LrShape::Cosine => {
            let progress = (step - warmup) as f64 / (total.saturating_sub(warmup).max(1)) as f64;
            (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())).max(0.0)
        }
    }
}

/// `torch.nn.utils.clip_grad_norm_`'s coefficient: `min(1, max_norm / (total + 1e-6))`.
pub fn clip_coefficient(max_norm: f64, total_norm: f64) -> f64 {
    (max_norm / (total_norm + 1e-6)).min(1.0)
}

/// Where a tensor goes in the optimizer (`build_muon_param_groups`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamRoute {
    /// Orthogonalized update; `split` = number of equal row blocks orthogonalized independently
    /// (3 for a fused QKV, 6 for a shared adaLN core).
    Muon { split: Option<usize> },
    /// Elementwise AdamW (embeddings, heads, vectors, boundary matrices).
    AdamW,
    /// Not updated: the tensor receives no gradient upstream (the discarded text path of a final
    /// dual-stream block under `final_block_text: keep`), so torch skips it entirely.
    Frozen,
}

/// The Linear weights `_muon_boundary_ids` sends to AdamW although they are matrices: every input
/// projection, the timestep MLP, the layer pool, the final head and the PiT `adaln` (whose
/// output rows interleave modulation chunks per pixel).
pub fn is_muon_boundary(key: &str) -> bool {
    key == "s_embedder.proj.weight"
        || key == "pixel_embedder.proj.weight"
        || key == "y_embedder.refiner.proj.weight"
        || key == "y_embedder.layer_pool.weight"
        || key == "final_layer.linear.weight"
        || (key.starts_with("t_embedder.") && key.ends_with(".weight"))
        || (key.starts_with("pixel_blocks.") && key.ends_with(".adaln.weight"))
}

/// The equal row split `_muon_split_map` records for a fused matrix, if any.
pub fn muon_split(key: &str) -> Option<usize> {
    let module = key.strip_suffix(".weight")?;
    let last = module.rsplit('.').next()?;
    if module.starts_with("modulation_cores.") && last.starts_with("adaln") {
        return Some(6);
    }
    // SelfAttention.qkv (text-adapter + PiT blocks), JointAttention.qkv_x/qkv_y and the
    // single-stream block's fused qkv (both only when the trunk is not GQA).
    if matches!(last, "qkv" | "qkv_x" | "qkv_y") {
        return Some(3);
    }
    None
}

/// Whether `key` is one of the parameters that receive no gradient upstream: the text-output path
/// of the last block when it is a dual-stream block whose text output is kept but discarded.
pub fn is_dead_parameter(key: &str, cfg: &ModelConfig) -> bool {
    if cfg.final_block_text != "keep" || cfg.depth == 0 || cfg.dual_depth < cfg.depth {
        return false;
    }
    let prefix = format!("blocks.{}.", cfg.depth - 1);
    let Some(rest) = key.strip_prefix(&prefix) else {
        return false;
    };
    [
        "norm_y2.",
        "attn.proj_y.",
        "attn_gate_y.",
        "attn_post_norm_y.",
        "mlp_y.",
        "mlp_post_norm_y.",
    ]
    .iter()
    .any(|p| rest.starts_with(p))
}

/// The optimizer route of a **full-model** tensor `key` of rank `ndim`.
pub fn full_param_route(
    key: &str,
    ndim: usize,
    kind: OptimizerKind,
    cfg: &ModelConfig,
) -> ParamRoute {
    if is_dead_parameter(key, cfg) {
        return ParamRoute::Frozen;
    }
    match kind {
        OptimizerKind::AdamW => ParamRoute::AdamW,
        OptimizerKind::Muon => {
            let linear_weight = ndim == 2 && key.ends_with(".weight");
            if linear_weight && !is_muon_boundary(key) {
                ParamRoute::Muon {
                    split: muon_split(key),
                }
            } else {
                ParamRoute::AdamW
            }
        }
    }
}

/// The optimizer route of an **adapter factor** of the Linear at `target` (`key` is the factor
/// key, `ndim` its rank). Upstream trains no adapters; the native rule keeps Muon's split of the
/// parameter space: a 2-D factor of a hidden matrix Muon would route is orthogonalized as its own
/// matrix (no row split — a factor's rows are not the fused blocks), everything else is AdamW.
pub fn adapter_param_route(target: &str, ndim: usize, kind: OptimizerKind) -> ParamRoute {
    match kind {
        OptimizerKind::AdamW => ParamRoute::AdamW,
        OptimizerKind::Muon => {
            let weight = format!("{target}.weight");
            if ndim == 2 && !is_muon_boundary(&weight) {
                ParamRoute::Muon { split: None }
            } else {
                ParamRoute::AdamW
            }
        }
    }
}

// =============================================================================================
// Random init (`IrisDiT.initialize_weights`)
// =============================================================================================

/// The distribution one backbone tensor is drawn from under `init: random`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitKind {
    Zeros,
    Ones,
    /// `N(0, std²)`.
    Normal {
        std: f64,
    },
    /// `U(−bound, bound)`.
    Uniform {
        bound: f64,
    },
}

/// The upstream init law of tensor `key` with `shape`. `fan_in` is the input width of the Linear a
/// `.bias` belongs to (PyTorch's default bias bound). `adaln_zero_init` is the config switch.
///
/// * Linear weights: `kaiming_uniform_(a=√5)` ⇒ `U(±1/√fan_in)`; biases `U(±1/√fan_in)`.
/// * `s_embedder.proj`: xavier-uniform weight, zero bias. `t_embedder.mlp.{0,2}.weight`: `N(0,
///   0.02²)`. `final_layer.linear`: zeros.
/// * every `adaln*` Linear (the shared cores, the PiT `adaln`) and every `SharedCoreBias` bias:
///   zeros under `adaln_zero_init`.
/// * RMSNorm gains: ones. `y_pos_embedding`: `N(0, 1)`.
pub fn init_kind(
    key: &str,
    shape: &[usize],
    fan_in: Option<usize>,
    adaln_zero_init: bool,
) -> InitKind {
    let (module, leaf) = key.rsplit_once('.').unwrap_or(("", key));
    let module_last = module.rsplit('.').next().unwrap_or(module);
    if key == "y_pos_embedding" {
        return InitKind::Normal { std: 1.0 };
    }
    if module_last.starts_with("adaln") {
        // Per-block SharedCoreBias parameters are zero-initialised unconditionally; the Linear
        // cores and the PiT adaln only under adaln_zero_init.
        let shared_bias = module.starts_with("blocks.") && leaf == "bias";
        if shared_bias || adaln_zero_init {
            return InitKind::Zeros;
        }
    }
    if module == "final_layer.linear" {
        return InitKind::Zeros;
    }
    match (leaf, shape.len()) {
        ("weight", 1) => InitKind::Ones,
        ("weight", 2) => {
            if module == "s_embedder.proj" {
                let (fan_out, fan_in) = (shape[0] as f64, shape[1] as f64);
                InitKind::Uniform {
                    bound: (6.0 / (fan_in + fan_out)).sqrt(),
                }
            } else if module.starts_with("t_embedder.mlp") {
                InitKind::Normal { std: 0.02 }
            } else {
                InitKind::Uniform {
                    bound: 1.0 / (shape[1] as f64).sqrt(),
                }
            }
        }
        ("bias", _) => {
            if module == "s_embedder.proj" {
                InitKind::Zeros
            } else {
                InitKind::Uniform {
                    bound: 1.0 / (fan_in.unwrap_or(1).max(1) as f64).sqrt(),
                }
            }
        }
        _ => InitKind::Zeros,
    }
}

/// Every backbone tensor `(key, shape)` the configured architecture owns — the state dict of
/// upstream's `IrisDiT(cfg)` under the native loader's key names, in a stable order. Random init
/// draws exactly these; a weights checkpoint must carry exactly these.
pub fn backbone_tensor_shapes(cfg: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    let d = cfg.hidden_size;
    let p = cfg.patch_size;
    let c = cfg.in_channels;
    let hd = d / cfg.num_heads;
    let gqa = cfg.num_kv_heads.is_some_and(|kv| kv != cfg.num_heads);
    let kv = cfg.num_kv_heads.unwrap_or(cfg.num_heads);
    let swiglu = |dim: usize, ratio: f64| -> usize {
        ((2 * ((dim as f64 * ratio) as usize)) as f64 / 3.0) as usize
    };
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let lin =
        |out: &mut Vec<(String, Vec<usize>)>, name: String, o: usize, i: usize, bias: bool| {
            out.push((format!("{name}.weight"), vec![o, i]));
            if bias {
                out.push((format!("{name}.bias"), vec![o]));
            }
        };
    let norm = |out: &mut Vec<(String, Vec<usize>)>, name: String, n: usize| {
        out.push((format!("{name}.weight"), vec![n]));
    };
    lin(&mut out, "s_embedder.proj".into(), d, p * p * c, true);
    lin(&mut out, "t_embedder.mlp.0".into(), d, 256, true);
    lin(&mut out, "t_embedder.mlp.2".into(), d, d, true);
    // LayerwiseTextEmbedder: two per-token layer-attention blocks, the layer pool, the refiner.
    let td = cfg.text_dim;
    let lap_hidden = (td as f64 * cfg.text_lap_mlp_ratio) as usize;
    for i in 0..2 {
        let b = format!("y_embedder.layer_blocks.{i}");
        norm(&mut out, format!("{b}.norm1"), td);
        lin(&mut out, format!("{b}.attn.qkv"), 3 * td, td, false);
        lin(&mut out, format!("{b}.attn.proj"), td, td, true);
        norm(&mut out, format!("{b}.norm2"), td);
        lin(&mut out, format!("{b}.mlp.0"), lap_hidden, td, true);
        lin(&mut out, format!("{b}.mlp.2"), td, lap_hidden, true);
    }
    lin(
        &mut out,
        "y_embedder.layer_pool".into(),
        1,
        cfg.text_lap_num_layers,
        true,
    );
    lin(&mut out, "y_embedder.refiner.proj".into(), d, td, true);
    for i in 0..2 {
        let b = format!("y_embedder.refiner.blocks.{i}");
        norm(&mut out, format!("{b}.norm1"), d);
        lin(&mut out, format!("{b}.attn.qkv"), 3 * d, d, false);
        if cfg.qk_norm {
            norm(&mut out, format!("{b}.attn.q_norm"), hd);
            norm(&mut out, format!("{b}.attn.k_norm"), hd);
        }
        lin(&mut out, format!("{b}.attn.proj"), d, d, true);
        norm(&mut out, format!("{b}.norm2"), d);
        let h = swiglu(d, cfg.mlp_ratio);
        lin(&mut out, format!("{b}.mlp.w1"), h, d, false);
        lin(&mut out, format!("{b}.mlp.w2"), d, h, false);
        lin(&mut out, format!("{b}.mlp.w3"), h, d, false);
    }
    norm(&mut out, "y_embedder.refiner.norm".into(), d);
    out.push(("y_pos_embedding".into(), vec![1, cfg.text_len, d]));
    lin(
        &mut out,
        "modulation_cores.adaln_img".into(),
        6 * d,
        d,
        true,
    );
    if cfg.dual_depth > 0 {
        lin(
            &mut out,
            "modulation_cores.adaln_txt".into(),
            6 * d,
            d,
            true,
        );
    }
    let mlp_h = swiglu(d, cfg.mlp_ratio);
    let qkv =
        |out: &mut Vec<(String, Vec<usize>)>, fused: String, q: String, k: String, v: String| {
            if gqa {
                out.push((format!("{q}.weight"), vec![cfg.num_heads * hd, d]));
                out.push((format!("{k}.weight"), vec![kv * hd, d]));
                out.push((format!("{v}.weight"), vec![kv * hd, d]));
            } else {
                out.push((format!("{fused}.weight"), vec![3 * d, d]));
            }
        };
    for i in 0..cfg.depth {
        let b = format!("blocks.{i}");
        let text_out = cfg.final_block_text == "keep" || i + 1 < cfg.depth;
        let swi = |out: &mut Vec<(String, Vec<usize>)>, name: String| {
            out.push((format!("{name}.w1.weight"), vec![mlp_h, d]));
            out.push((format!("{name}.w2.weight"), vec![d, mlp_h]));
            out.push((format!("{name}.w3.weight"), vec![mlp_h, d]));
        };
        if i < cfg.dual_depth {
            for n in ["norm_x1", "norm_x2", "norm_y1"] {
                norm(&mut out, format!("{b}.{n}"), d);
            }
            qkv(
                &mut out,
                format!("{b}.attn.qkv_x"),
                format!("{b}.attn.q_proj_x"),
                format!("{b}.attn.k_proj_x"),
                format!("{b}.attn.v_proj_x"),
            );
            qkv(
                &mut out,
                format!("{b}.attn.qkv_y"),
                format!("{b}.attn.q_proj_y"),
                format!("{b}.attn.k_proj_y"),
                format!("{b}.attn.v_proj_y"),
            );
            for n in [
                "attn.q_norm_x",
                "attn.k_norm_x",
                "attn.q_norm_y",
                "attn.k_norm_y",
            ] {
                norm(&mut out, format!("{b}.{n}"), hd);
            }
            lin(&mut out, format!("{b}.attn.proj_x"), d, d, true);
            lin(&mut out, format!("{b}.attn_gate_x"), d, d, false);
            norm(&mut out, format!("{b}.attn_post_norm_x"), d);
            swi(&mut out, format!("{b}.mlp_x"));
            norm(&mut out, format!("{b}.mlp_post_norm_x"), d);
            if text_out {
                norm(&mut out, format!("{b}.norm_y2"), d);
                lin(&mut out, format!("{b}.attn.proj_y"), d, d, true);
                lin(&mut out, format!("{b}.attn_gate_y"), d, d, false);
                norm(&mut out, format!("{b}.attn_post_norm_y"), d);
                swi(&mut out, format!("{b}.mlp_y"));
                norm(&mut out, format!("{b}.mlp_post_norm_y"), d);
            }
            out.push((format!("{b}.adaln_img.bias"), vec![6 * d]));
            out.push((format!("{b}.adaln_txt.bias"), vec![6 * d]));
        } else {
            norm(&mut out, format!("{b}.norm1"), d);
            norm(&mut out, format!("{b}.norm2"), d);
            qkv(
                &mut out,
                format!("{b}.qkv"),
                format!("{b}.q_proj"),
                format!("{b}.k_proj"),
                format!("{b}.v_proj"),
            );
            norm(&mut out, format!("{b}.q_norm"), hd);
            norm(&mut out, format!("{b}.k_norm"), hd);
            lin(&mut out, format!("{b}.attn_gate"), d, d, false);
            lin(&mut out, format!("{b}.attn_proj"), d, d, true);
            norm(&mut out, format!("{b}.attn_post_norm"), d);
            swi(&mut out, format!("{b}.mlp"));
            norm(&mut out, format!("{b}.mlp_post_norm"), d);
            out.push((format!("{b}.adaln.bias"), vec![6 * d]));
        }
    }
    let px = &cfg.pixel;
    let pix = px.hidden_size;
    let ppp = p * p;
    lin(&mut out, "pixel_embedder.proj".into(), pix, c, true);
    for i in 0..px.depth {
        let b = format!("pixel_blocks.{i}");
        norm(&mut out, format!("{b}.norm1"), pix);
        norm(&mut out, format!("{b}.norm2"), pix);
        lin(&mut out, format!("{b}.adaln"), 4 * pix * ppp, d, true);
        lin(
            &mut out,
            format!("{b}.compress"),
            px.attn_hidden_size,
            ppp * pix,
            true,
        );
        lin(
            &mut out,
            format!("{b}.expand"),
            ppp * pix,
            px.attn_hidden_size,
            true,
        );
        let a = px.attn_hidden_size;
        lin(&mut out, format!("{b}.attn.qkv"), 3 * a, a, false);
        if cfg.qk_norm {
            norm(&mut out, format!("{b}.attn.q_norm"), a / px.num_heads);
            norm(&mut out, format!("{b}.attn.k_norm"), a / px.num_heads);
        }
        lin(&mut out, format!("{b}.attn.proj"), a, a, true);
        let fc = (pix as f64 * px.mlp_ratio) as usize;
        lin(&mut out, format!("{b}.mlp.fc1"), fc, pix, true);
        lin(&mut out, format!("{b}.mlp.fc2"), pix, fc, true);
    }
    norm(&mut out, "final_layer.norm".into(), pix);
    lin(&mut out, "final_layer.linear".into(), c, pix, true);
    out
}

// =============================================================================================
// Data: the ranged walk, captions, preprocessing (`data/samplers.py`, `data/datasets.py`)
// =============================================================================================

/// The single-device `RangedSampler` walk: the first `chunks · ⌊n / chunks⌋` dataset indices in
/// order (`chunks = min(640, n)`; the shorter tail is left unassigned exactly as upstream), batched
/// by `batch_size` with the last partial batch kept (`drop_last=False`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataWalk {
    pub items: usize,
    pub covered: usize,
    pub batch_size: usize,
}

impl DataWalk {
    pub fn new(items: usize, batch_size: usize) -> Result<Self> {
        if items == 0 {
            return Err(Error::Msg("iris training: the dataset is empty".into()));
        }
        if batch_size == 0 {
            return Err(Error::Msg("iris training: batch_size must be >= 1".into()));
        }
        let chunks = CANONICAL_CHUNKS.min(items);
        Ok(Self {
            items,
            covered: chunks * (items / chunks),
            batch_size,
        })
    }

    /// Dataset items the ranged walk never visits (`items − covered`): upstream's
    /// `chunks · (n / chunks)` coverage drops the tail of a dataset larger than
    /// [`CANONICAL_CHUNKS`] that is not a multiple of it.
    pub fn unused_items(&self) -> usize {
        self.items - self.covered
    }

    /// The run-start warning a trainer emits when [`Self::unused_items`] is non-zero (`None`
    /// otherwise), naming how many items — and which index range — no epoch will train on.
    pub fn unused_tail_warning(&self) -> Option<String> {
        let unused = self.unused_items();
        (unused > 0).then(|| {
            format!(
                "iris training: warning — {unused} of {} dataset items (indices {}..{}) are never \
                 trained on: the upstream ranged sampler covers {} · ({} / {}) = {} items; add or \
                 remove items to a multiple of {} to use them all",
                self.items,
                self.covered,
                self.items,
                CANONICAL_CHUNKS,
                self.items,
                CANONICAL_CHUNKS,
                self.covered,
                CANONICAL_CHUNKS
            )
        })
    }

    /// Batches per epoch (`len(dataloader)`).
    pub fn batches_per_epoch(&self) -> usize {
        self.covered.div_ceil(self.batch_size)
    }

    /// The dataset indices of batch `position` (0-based) of an epoch.
    pub fn batch(&self, position: usize) -> std::ops::Range<usize> {
        let start = position * self.batch_size;
        start.min(self.covered)..(start + self.batch_size).min(self.covered)
    }
}

/// The optimizer-step horizon of a run (`total_steps` in `Trainer.run`): `ceil(batches /
/// accum) · num_epochs`, capped by `max_steps`; with no epoch cap the horizon is `max_steps`.
pub fn total_optimizer_steps(
    batches_per_epoch: usize,
    grad_accum: usize,
    num_epochs: Option<u32>,
    max_steps: u32,
) -> u64 {
    match num_epochs {
        None => max_steps as u64,
        Some(epochs) => {
            let per_epoch = batches_per_epoch.div_ceil(grad_accum.max(1)) as u64;
            let total = per_epoch * epochs as u64;
            if max_steps > 0 {
                total.min(max_steps as u64)
            } else {
                total
            }
        }
    }
}

/// `select_caption`: with `caption_fields` set, a uniform choice among the fields the item carries
/// (falling back to `caption_field`); otherwise `caption_field`. The item's fields are its
/// [`ITEM_CAPTIONS_KEY`] map; `caption_field == "caption"` with no map entry reads
/// [`TrainingItem::caption`].
pub fn select_caption(
    item: &TrainingItem,
    caption_field: &str,
    caption_fields: &[String],
    rng: &mut HostRng,
) -> Result<String> {
    let field = |name: &str| -> Option<String> {
        if let Some(text) = item
            .model_options
            .get(ITEM_CAPTIONS_KEY)
            .and_then(|m| m.get(name))
            .and_then(JsonValue::as_str)
        {
            return Some(text.to_string());
        }
        (name == "caption").then(|| item.caption.clone())
    };
    if !caption_fields.is_empty() {
        let present: Vec<String> = caption_fields
            .iter()
            .filter_map(|f| field(f).filter(|t| !t.is_empty()))
            .collect();
        if !present.is_empty() {
            let i = rng.below(present.len());
            return Ok(present[i].clone());
        }
        return Ok(field(caption_field).unwrap_or_default());
    }
    field(caption_field).ok_or_else(|| {
        Error::Msg(format!(
            "iris training: item {} has no caption field {caption_field:?}",
            item.image_path.display()
        ))
    })
}

/// Every caption an item can contribute (the text-conditioning memo encodes each once).
pub fn item_caption_variants(
    item: &TrainingItem,
    caption_field: &str,
    caption_fields: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    let names: Vec<&str> = if caption_fields.is_empty() {
        vec![caption_field]
    } else {
        caption_fields
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(caption_field))
            .collect()
    };
    for name in names {
        let text = item
            .model_options
            .get(ITEM_CAPTIONS_KEY)
            .and_then(|m| m.get(name))
            .and_then(JsonValue::as_str)
            .map(str::to_string)
            .or_else(|| (name == "caption").then(|| item.caption.clone()));
        if let Some(text) = text {
            if !out.contains(&text) {
                out.push(text);
            }
        }
    }
    out
}

/// One preprocessed training image: `[3, size, size]` f32 CHW in `[−1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainImage {
    pub size: usize,
    pub chw: Vec<f32>,
}

/// `PixelDataset._get` under the fixed-square policy: decode → RGB → PIL-bicubic resize of the
/// shortest side to `size` (the other side `int(size · long / short)`) → center crop
/// (`round((h − size)/2)`) → `x / 127.5 − 1`. The resize is the PIL-exact fixed-point resampler
/// ([`crate::imageops::resize_bicubic_u8`]).
pub fn preprocess_image(path: &Path, size: usize) -> Result<TrainImage> {
    let img = image::open(path)
        .map_err(|e| Error::Msg(format!("iris training: decode {}: {e}", path.display())))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    preprocess_rgb(img.as_raw(), w, h, size)
}

/// [`preprocess_image`] on an already-decoded RGB8 buffer (`h · w · 3`).
pub fn preprocess_rgb(rgb: &[u8], w: usize, h: usize, size: usize) -> Result<TrainImage> {
    if w == 0 || h == 0 || size == 0 {
        return Err(Error::Msg("iris training: empty image".into()));
    }
    let (rw, rh) = if w <= h {
        (size, (size as f64 * h as f64 / w as f64) as usize)
    } else {
        ((size as f64 * w as f64 / h as f64) as usize, size)
    };
    let resized = crate::imageops::resize_bicubic_u8(rgb, h, w, rh, rw)?;
    // PIL center crop: top/left = round((dim − size) / 2) (Python round-half-even).
    let round_half_even = |v: f64| -> usize {
        let r = v.round();
        if (v - v.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
            (r - 1.0) as usize
        } else {
            r as usize
        }
    };
    let top = round_half_even((rh - size) as f64 / 2.0);
    let left = round_half_even((rw - size) as f64 / 2.0);
    let mut chw = vec![0f32; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let src = ((top + y) * rw + (left + x)) * 3;
            for c in 0..3 {
                chw[c * size * size + y * size + x] = resized[src + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(TrainImage { size, chw })
}

// =============================================================================================
// Request surface: the Iris options bag and the resolved plan
// =============================================================================================

/// How the run's weights start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitMode {
    /// The backbone resource's `model.safetensors` (weights-only init — upstream `load_from`
    /// semantics: weights only, a fresh optimizer, EMA seeded from the loaded weights).
    Weights,
    /// Upstream `initialize_weights` from the backbone's `config.yaml` alone (full training only).
    Random,
    /// The raw weights of a previous run's checkpoint directory or an exported full-model
    /// directory (`train.load_from`, full training only).
    LoadFrom(PathBuf),
}

/// What a run trains and emits.
#[derive(Clone, Debug, PartialEq)]
pub enum ArtifactPlan {
    /// Every backbone tensor; exported as a full model directory.
    Full,
    /// LoRA on `targets` (backbone Linear module paths).
    Lora {
        rank: usize,
        alpha: f32,
        targets: Vec<String>,
    },
    /// LoKr on `targets` with LyCORIS factorization `decompose_factor` (−1 = balanced).
    Lokr {
        rank: usize,
        alpha: f32,
        decompose_factor: i32,
        targets: Vec<String>,
    },
}

impl ArtifactPlan {
    pub fn kind(&self) -> &'static str {
        match self {
            ArtifactPlan::Full => "full",
            ArtifactPlan::Lora { .. } => "lora",
            ArtifactPlan::Lokr { .. } => "lokr",
        }
    }
    pub fn targets(&self) -> &[String] {
        match self {
            ArtifactPlan::Full => &[],
            ArtifactPlan::Lora { targets, .. } | ArtifactPlan::Lokr { targets, .. } => targets,
        }
    }
}

/// `train.mixed_precision`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedPrecision {
    /// `bf16` autocast over f32 master parameters (the release).
    Bf16,
    /// `no`: f32 throughout.
    Fp32,
}

/// Which weights a preview render or an export reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsSelect {
    Raw,
    Ema,
}

impl WeightsSelect {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightsSelect::Raw => "raw",
            WeightsSelect::Ema => "ema",
        }
    }
    fn parse(key: &str, s: &str) -> Result<Self> {
        match s {
            "raw" => Ok(WeightsSelect::Raw),
            "ema" => Ok(WeightsSelect::Ema),
            other => Err(Error::Msg(format!(
                "iris training: {key} must be raw or ema, got {other:?}"
            ))),
        }
    }
}

/// How the frozen Qwen3-VL conditioning is produced. Both are bit-identical (the encoder is
/// frozen and deterministic); `OnTheFly` is upstream's (encode each batch), `Cached` memoizes every
/// distinct caption once and drops the encoder before training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextConditioningMode {
    OnTheFly,
    Cached,
}

/// `train.resume_data_policy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeDataPolicy {
    /// Restore the checkpoint's exact data cursor.
    Exact,
    /// Keep model / EMA / optimizer / scheduler / step, start the current dataset at epoch 1.
    NewPhase,
}

/// Preview rendering (`_render_validation`).
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewPlan {
    /// Every `every` optimizer steps (`sample_every / gradient_accumulation`); 0 = off.
    pub every: u32,
    pub prompts: Vec<String>,
    pub steps: usize,
    pub cfg_scale: f32,
    pub negative_prompt: String,
    pub weights: WeightsSelect,
    /// Rendered side (upstream: the stage's `data.image_size`).
    pub size: usize,
    pub seed: u64,
}

/// The `TrainingConfig::model_options["iris"]` object, every key optional (upstream defaults).
///
/// | key | upstream | default |
/// | --- | --- | --- |
/// | `init` | `load_from` / fresh model | `"weights"` (`"random"`, or a `load_from` path) |
/// | `load_from` | `train.load_from` | unset |
/// | `prediction` | `flow.prediction` | the backbone config's (`"v"`) |
/// | `flow_shift`, `shift_law`, `shift_base_tokens` | `flow.*` | the backbone config's |
/// | `logit_mean`, `logit_std` | `flow.*` | backbone config's (0, 1) |
/// | `x_pred_sigma_min` | `flow.x_pred_sigma_min` | 0.05 |
/// | `text_dropout` | `train.text_dropout` | 0.1 |
/// | `gradient_clip` | `train.gradient_clip` | 0.5 |
/// | `betas` | `optimizer.betas` | `[0.9, 0.95]` |
/// | `muon_momentum`, `muon_nesterov`, `muon_adjust_lr` | `optimizer.*` | 0.95, true, `"rms_norm"` |
/// | `auto_lr`, `base_batch_size` | `optimizer.*` | `"none"`, 256 |
/// | `ema_enabled`, `ema_decay` | `train.ema.*` | true, 0.9999 |
/// | `num_epochs` | `train.num_epochs` | unset (the `steps` horizon) |
/// | `nan_loss_tolerance` | `train.nan_loss_tolerance` | 20 |
/// | `caption_field`, `caption_fields` | `data.*` | `"caption"`, `[]` |
/// | `on_caption_overflow` | `text_encoder.on_caption_overflow` | backbone config's |
/// | `text_conditioning` | (on the fly) | `"on_the_fly"` (`"cached"`) |
/// | `keep_last_checkpoints`, `milestone_steps` | `train.*` | 0, `[]` |
/// | `resume_from` | `train.resume_from` (path) | unset |
/// | `resume_data_policy`, `override_lr_on_resume` | `train.*` | `"exact"`, false |
/// | `preview_weights` | (raw `core`) | the `export_weights` default (`"raw"` / `"ema"`) |
/// | `preview_negative_prompt` | `sample.negative_prompt` | `""` |
/// | `upstream_validation_prompts` | `validation_prompts` default | false (true: render [`DEFAULT_VALIDATION_PROMPTS`] when `sample_prompts` is empty) |
/// | `export_weights` | `export_checkpoint.py` (EMA when on) | full model: `"ema"` when EMA is on, else `"raw"`; LoRA / LoKr: `"raw"` |
///
/// The adapter default is `raw` because upstream ships no adapter EMA/export and an EMA at the
/// default decay seeded from the adapter's zero delta (LoRA `B = 0`, LoKr `w1 = 0`) stays close to
/// that zero over a typical 1–3k-step adapter run; `export_weights: "ema"` still selects it.
/// | `export_dtype` | `--dtype` | `"fp32"` (`"bf16"`) |
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IrisTrainOptions(pub JsonMap<String, JsonValue>);

const OPTION_KEYS: [&str; 33] = [
    "init",
    "load_from",
    "prediction",
    "flow_shift",
    "shift_law",
    "shift_base_tokens",
    "logit_mean",
    "logit_std",
    "x_pred_sigma_min",
    "text_dropout",
    "gradient_clip",
    "betas",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "caption_field",
    "caption_fields",
    "on_caption_overflow",
    "text_conditioning",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_from",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_weights",
    "preview_negative_prompt",
    "export_weights",
];
const OPTION_KEYS_EXTRA: [&str; 2] = ["export_dtype", "upstream_validation_prompts"];

impl IrisTrainOptions {
    /// Read the options object from a request (absent = all defaults). Unknown keys are refused.
    pub fn from_config(cfg: &TrainingConfig) -> Result<Self> {
        let map = match cfg.model_options.get(OPTIONS_KEY) {
            None | Some(JsonValue::Null) => JsonMap::new(),
            Some(JsonValue::Object(m)) => m.clone(),
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris training: model_options.{OPTIONS_KEY} must be an object, got {other}"
                )))
            }
        };
        for key in map.keys() {
            if !OPTION_KEYS.contains(&key.as_str()) && !OPTION_KEYS_EXTRA.contains(&key.as_str()) {
                return Err(Error::Unsupported(format!(
                    "iris training: model_options.{OPTIONS_KEY}.{key} is not an Iris training \
                     option (known: {}, {})",
                    OPTION_KEYS.join(", "),
                    OPTION_KEYS_EXTRA.join(", ")
                )));
            }
        }
        Ok(Self(map))
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key).filter(|v| !v.is_null())
    }
    fn f64(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                Error::Msg(format!(
                    "iris training: {key} must be a finite number, got {v}"
                ))
            }),
        }
    }
    fn u64(&self, key: &str) -> Result<Option<u64>> {
        self.get(key)
            .map(|v| {
                v.as_u64().ok_or_else(|| {
                    Error::Msg(format!(
                        "iris training: {key} must be a non-negative integer, got {v}"
                    ))
                })
            })
            .transpose()
    }
    fn bool(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_bool()
                .ok_or_else(|| Error::Msg(format!("iris training: {key} must be a bool, got {v}"))),
        }
    }
    fn string(&self, key: &str) -> Result<Option<String>> {
        self.get(key)
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    Error::Msg(format!("iris training: {key} must be a string, got {v}"))
                })
            })
            .transpose()
    }
    fn strings(&self, key: &str) -> Result<Vec<String>> {
        match self.get(key) {
            None => Ok(Vec::new()),
            Some(JsonValue::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        Error::Msg(format!("iris training: {key} must be a list of strings"))
                    })
                })
                .collect(),
            Some(v) => Err(Error::Msg(format!(
                "iris training: {key} must be a list of strings, got {v}"
            ))),
        }
    }
}

/// The training-relevant `flow` keys of a backbone `config.yaml` that [`super::FlowConfig`] does
/// not carry (it keeps only the inference-relevant ones).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainFlowDefaults {
    pub timestep_sampler: String,
    pub logit_mean: f64,
    pub logit_std: f64,
    pub x_pred_sigma_min: f64,
    pub shift_base_tokens: usize,
    pub adaln_zero_init: bool,
}

impl TrainFlowDefaults {
    /// Read from the YAML text (upstream defaults for absent keys).
    pub fn parse(text: &str) -> Result<Self> {
        let root = super::yaml::parse(text)?;
        let flow = super::Section(root.get("flow"));
        let model = super::Section(root.get("model"));
        Ok(Self {
            timestep_sampler: flow.string("timestep_sampler", "logit_normal")?,
            logit_mean: flow.f64("logit_mean", 0.0)?,
            logit_std: flow.f64("logit_std", 1.0)?,
            x_pred_sigma_min: flow.f64("x_pred_sigma_min", 0.05)?,
            shift_base_tokens: flow.usize("shift_base_tokens", 256)?,
            adaln_zero_init: model.bool("adaln_zero_init", true)?,
        })
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(super::BACKBONE_CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("iris: read {}: {e}", path.display())))?;
        Self::parse(&text)
    }
}

/// The fully resolved, validated plan of one training run.
#[derive(Clone, Debug, PartialEq)]
pub struct IrisTrainPlan {
    pub artifact: ArtifactPlan,
    pub init: InitMode,
    pub image_size: usize,
    pub batch_size: usize,
    pub grad_accum: usize,
    /// Optimizer-step horizon (`train.max_steps` = `TrainingConfig::steps / grad_accum`).
    pub max_steps: u32,
    pub num_epochs: Option<u32>,
    pub seed: u64,
    pub mixed_precision: MixedPrecision,
    pub flow: FlowObjective,
    pub optimizer: OptimizerPlan,
    pub schedule: LrShape,
    pub warmup_steps: u64,
    pub gradient_clip: f64,
    pub text_dropout: f64,
    /// EMA decay when enabled.
    pub ema: Option<f64>,
    pub nan_loss_tolerance: u32,
    pub caption_field: String,
    pub caption_fields: Vec<String>,
    pub on_caption_overflow: String,
    pub text_conditioning: TextConditioningMode,
    /// Checkpoint cadence in optimizer steps (`TrainingConfig::save_every / grad_accum`); 0 = off.
    pub save_every: u32,
    pub keep_last_checkpoints: usize,
    pub milestone_steps: Vec<u64>,
    pub preview: PreviewPlan,
    pub resume: bool,
    pub resume_from: Option<PathBuf>,
    pub resume_data_policy: ResumeDataPolicy,
    pub override_lr_on_resume: bool,
    pub export_weights: WeightsSelect,
    pub export_bf16: bool,
}

impl IrisTrainPlan {
    /// Resolve and validate a request against the backbone config. `linear_keys` are the backbone
    /// Linear module paths (from the backbone weights or, under random init, the architecture) —
    /// used to resolve the adapter targets; pass an empty slice to skip target resolution (the
    /// request-only `validate`).
    pub fn resolve(
        req: &TrainingRequest,
        config: &IrisConfig,
        flow_defaults: &TrainFlowDefaults,
        linear_paths: &[String],
    ) -> Result<Self> {
        let cfg = &req.config;
        let opts = IrisTrainOptions::from_config(cfg)?;
        let unsupported = |what: String| Err(Error::Unsupported(format!("iris training: {what}")));

        if cfg.gradient_checkpointing {
            return unsupported(
                "gradient (activation) checkpointing is not implemented by the native Iris trainer \
                 (upstream train.activation_checkpointing policies)"
                    .into(),
            );
        }
        let loss = cfg.loss_type.trim().to_ascii_lowercase();
        if !(loss.is_empty() || loss == "mse") {
            return unsupported(format!(
                "loss_type {:?} — the Iris objective is the velocity MSE",
                cfg.loss_type
            ));
        }
        let bias = cfg.timestep_bias.trim().to_ascii_lowercase();
        if !(bias.is_empty() || bias == "balanced") {
            return unsupported(format!(
                "timestep_bias {:?} — upstream samples logit-normal or uniform indices only",
                cfg.timestep_bias
            ));
        }
        let mixed_precision = match cfg.train_dtype.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => MixedPrecision::Bf16,
            "f32" | "fp32" | "float32" | "no" => MixedPrecision::Fp32,
            other => {
                return unsupported(format!(
                    "train_dtype {other:?} (upstream mixed_precision bf16 | no; fp16 is not \
                     implemented)"
                ))
            }
        };
        let schedule = match cfg.lr_scheduler {
            LrSchedule::Constant => LrShape::Constant,
            LrSchedule::Cosine => LrShape::Cosine,
            LrSchedule::Linear => {
                return unsupported("lr_scheduler linear (upstream: constant | cosine)".into())
            }
        };
        let kind = match cfg.optimizer.trim().to_ascii_lowercase().as_str() {
            "adamw" => OptimizerKind::AdamW,
            "muon" => OptimizerKind::Muon,
            other => return unsupported(format!("optimizer {other:?} (upstream: adamw | muon)")),
        };
        if !(cfg.learning_rate.is_finite() && cfg.learning_rate >= 0.0) {
            return Err(Error::Msg(
                "iris training: learning_rate must be >= 0".into(),
            ));
        }
        if !(cfg.weight_decay.is_finite() && cfg.weight_decay >= 0.0) {
            return Err(Error::Msg(
                "iris training: weight_decay must be >= 0".into(),
            ));
        }
        if cfg.steps == 0 {
            return Err(Error::Msg("iris training: steps must be >= 1".into()));
        }
        let patch = config.model.patch_size;
        let image_size = cfg.resolution as usize;
        if image_size == 0 || !image_size.is_multiple_of(patch) {
            return Err(Error::Msg(format!(
                "iris training: resolution {image_size} must be a positive multiple of the \
                 backbone patch_size {patch} (upstream data.image_size)"
            )));
        }
        let batch_size = cfg.batch_size.max(1) as usize;
        let grad_accum = cfg.gradient_accumulation.max(1) as usize;
        // The shared contract counts micro-steps; the loop counts optimizer steps. A cadence that
        // is not a whole number of accumulation windows has no optimizer-step equivalent: refuse.
        for (field, value) in [
            ("steps", cfg.steps),
            ("save_every", cfg.save_every),
            ("sample_every", cfg.sample_every),
        ] {
            if !(value as usize).is_multiple_of(grad_accum) {
                return Err(Error::Msg(format!(
                    "iris training: {field} = {value} micro-steps is not a multiple of \
                     gradient_accumulation = {grad_accum} (Iris steps, checkpoints and previews \
                     land on optimizer-step boundaries)"
                )));
            }
        }
        let accum_u32 = grad_accum as u32;

        // ---- artifact --------------------------------------------------------------------------
        let artifact = if cfg.full_finetune {
            ArtifactPlan::Full
        } else {
            if cfg.rank == 0 {
                return Err(Error::Msg(
                    "iris training: adapter rank must be >= 1".into(),
                ));
            }
            if !(cfg.alpha.is_finite() && cfg.alpha > 0.0) {
                return Err(Error::Msg(
                    "iris training: adapter alpha must be > 0".into(),
                ));
            }
            let targets = if linear_paths.is_empty() {
                Vec::new()
            } else {
                select_adapter_targets(linear_paths, &cfg.lora_target_modules)?
            };
            match cfg.network_type {
                NetworkType::Lora => ArtifactPlan::Lora {
                    rank: cfg.rank as usize,
                    alpha: cfg.alpha,
                    targets,
                },
                NetworkType::Lokr => ArtifactPlan::Lokr {
                    rank: cfg.rank as usize,
                    alpha: cfg.alpha,
                    decompose_factor: cfg.decompose_factor,
                    targets,
                },
            }
        };

        // ---- init ------------------------------------------------------------------------------
        let load_from = opts.string("load_from")?.map(PathBuf::from);
        let init = match (opts.string("init")?.as_deref(), load_from) {
            (None | Some("weights"), None) => InitMode::Weights,
            (Some("random"), None) => InitMode::Random,
            (None | Some("load_from"), Some(p)) => InitMode::LoadFrom(p),
            (Some(other), lf) => {
                return Err(Error::Msg(format!(
                    "iris training: init {other:?} with load_from {lf:?} (init: weights | random; \
                     load_from implies a weights-only start from that path)"
                )))
            }
        };
        if !matches!(artifact, ArtifactPlan::Full) && init != InitMode::Weights {
            return unsupported(
                "init random / load_from applies to full training only; an adapter trains on \
                 the backbone resource's weights"
                    .into(),
            );
        }

        // ---- flow ------------------------------------------------------------------------------
        let prediction = match opts.string("prediction")? {
            Some(p) => parse_prediction(&p)?,
            None => parse_prediction(&config.flow.prediction)?,
        };
        let base_shift = opts.f64("flow_shift", config.flow.shift)?;
        if base_shift <= 0.0 {
            return Err(Error::Msg("iris training: flow_shift must be > 0".into()));
        }
        let shift_law = opts
            .string("shift_law")?
            .unwrap_or_else(|| config.flow.shift_law.clone());
        let shift_base_tokens = opts
            .u64("shift_base_tokens")?
            .map(|v| v as usize)
            .unwrap_or(flow_defaults.shift_base_tokens);
        let tokens = (image_size / patch) * (image_size / patch);
        let shift = resolution_shift(tokens, &shift_law, base_shift, shift_base_tokens)?;
        let timestep_type = cfg.timestep_type.trim().to_ascii_lowercase();
        let sampler_name = if timestep_type.is_empty() {
            flow_defaults.timestep_sampler.clone()
        } else {
            timestep_type
        };
        let sampler = match sampler_name.as_str() {
            "logit_normal" | "sigmoid" | "logit-normal" => TimestepSampler::LogitNormal {
                mean: opts.f64("logit_mean", flow_defaults.logit_mean)?,
                std: opts.f64("logit_std", flow_defaults.logit_std)?,
            },
            "uniform" => TimestepSampler::Uniform,
            other => {
                return unsupported(format!(
                    "timestep_type {other:?} (upstream flow.timestep_sampler: logit_normal | \
                     uniform)"
                ))
            }
        };
        let flow = FlowObjective {
            num_train_timesteps: config.flow.num_train_timesteps,
            shift,
            base_shift,
            shift_law,
            shift_base_tokens,
            sampler,
            prediction,
            x_pred_sigma_min: opts.f64("x_pred_sigma_min", flow_defaults.x_pred_sigma_min)?,
        };

        // ---- optimizer -------------------------------------------------------------------------
        let betas = match opts.get("betas") {
            None => (0.9, 0.95),
            Some(JsonValue::Array(b)) if b.len() == 2 => {
                let f = |v: &JsonValue| v.as_f64().filter(|x| (0.0..1.0).contains(x));
                match (f(&b[0]), f(&b[1])) {
                    (Some(a), Some(c)) => (a, c),
                    _ => return Err(Error::Msg("iris training: betas must be in [0, 1)".into())),
                }
            }
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris training: betas must be a pair, got {v}"
                )))
            }
        };
        let muon_momentum = opts.f64("muon_momentum", 0.95)?;
        if !(0.0..1.0).contains(&muon_momentum) {
            return Err(Error::Msg(
                "iris training: muon_momentum must be in [0, 1)".into(),
            ));
        }
        let muon_adjust_lr = match opts.string("muon_adjust_lr")?.as_deref() {
            None | Some("rms_norm") => MuonAdjustLr::RmsNorm,
            Some("spectral_norm") => MuonAdjustLr::SpectralNorm,
            Some("none") => MuonAdjustLr::None,
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris training: muon_adjust_lr {other:?} (rms_norm | spectral_norm | none)"
                )))
            }
        };
        let auto_lr = opts.string("auto_lr")?.unwrap_or_else(|| "none".into());
        let base_batch_size = opts.u64("base_batch_size")?.unwrap_or(256) as usize;
        if base_batch_size == 0 {
            return Err(Error::Msg(
                "iris training: base_batch_size must be >= 1".into(),
            ));
        }
        let base_lr = cfg.learning_rate as f64;
        let lr = scale_lr(base_lr, &auto_lr, batch_size * grad_accum, base_batch_size)?;
        let optimizer = OptimizerPlan {
            kind,
            base_lr,
            lr,
            betas,
            weight_decay: cfg.weight_decay as f64,
            muon_momentum,
            muon_nesterov: opts.bool("muon_nesterov", true)?,
            muon_adjust_lr,
            auto_lr,
            base_batch_size,
        };

        // ---- loop knobs ------------------------------------------------------------------------
        let text_dropout = opts.f64("text_dropout", 0.1)?;
        if !(0.0..=1.0).contains(&text_dropout) {
            return Err(Error::Msg(
                "iris training: text_dropout must be in [0, 1]".into(),
            ));
        }
        let gradient_clip = opts.f64("gradient_clip", 0.5)?;
        if gradient_clip <= 0.0 {
            return Err(Error::Msg(
                "iris training: gradient_clip must be > 0".into(),
            ));
        }
        let ema = if opts.bool("ema_enabled", true)? {
            let d = opts.f64("ema_decay", 0.9999)?;
            if !(0.0..=1.0).contains(&d) {
                return Err(Error::Msg(
                    "iris training: ema_decay must be in [0, 1]".into(),
                ));
            }
            Some(d)
        } else {
            None
        };
        let num_epochs = opts.u64("num_epochs")?.map(|v| v as u32);
        if num_epochs == Some(0) {
            return Err(Error::Msg("iris training: num_epochs must be >= 1".into()));
        }
        let on_caption_overflow = opts
            .string("on_caption_overflow")?
            .unwrap_or_else(|| config.text_encoder.on_caption_overflow.clone());
        if !matches!(on_caption_overflow.as_str(), "warn" | "error" | "silent") {
            return Err(Error::Msg(format!(
                "iris training: on_caption_overflow {on_caption_overflow:?} (warn | error | silent)"
            )));
        }
        let text_conditioning = match opts.string("text_conditioning")?.as_deref() {
            None | Some("on_the_fly") => TextConditioningMode::OnTheFly,
            Some("cached") => TextConditioningMode::Cached,
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris training: text_conditioning {other:?} (on_the_fly | cached)"
                )))
            }
        };
        let resume_data_policy = match opts.string("resume_data_policy")?.as_deref() {
            None | Some("exact") => ResumeDataPolicy::Exact,
            Some("new_phase") => ResumeDataPolicy::NewPhase,
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris training: resume_data_policy {other:?} (exact | new_phase)"
                )))
            }
        };
        // Upstream exports the EMA of a full model (`export_checkpoint.py`); it has no adapter
        // EMA, and an adapter EMA seeded from the zero delta lags far behind a typical run, so an
        // adapter exports its raw factors unless the request asks otherwise. Previews show what the
        // run will export unless `preview_weights` says otherwise.
        let default_weights = match (&artifact, ema) {
            (ArtifactPlan::Full, Some(_)) => WeightsSelect::Ema,
            _ => WeightsSelect::Raw,
        };
        let preview_weights = match opts.string("preview_weights")? {
            Some(s) => WeightsSelect::parse("preview_weights", &s)?,
            None => default_weights,
        };
        if preview_weights == WeightsSelect::Ema && ema.is_none() {
            return Err(Error::Msg(
                "iris training: preview_weights ema needs ema_enabled".into(),
            ));
        }
        let export_weights = match opts.string("export_weights")? {
            Some(s) => WeightsSelect::parse("export_weights", &s)?,
            None => default_weights,
        };
        if export_weights == WeightsSelect::Ema && ema.is_none() {
            return Err(Error::Msg(
                "iris training: export_weights ema needs ema_enabled".into(),
            ));
        }
        let export_bf16 = match opts.string("export_dtype")?.as_deref() {
            None | Some("fp32") => false,
            Some("bf16") => true,
            Some(other) => {
                return Err(Error::Msg(format!(
                    "iris training: export_dtype {other:?} (fp32 | bf16)"
                )))
            }
        };
        let milestone_steps = match opts.get("milestone_steps") {
            None => Vec::new(),
            Some(JsonValue::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        Error::Msg("iris training: milestone_steps must be integers".into())
                    })
                })
                .collect::<Result<_>>()?,
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris training: milestone_steps must be a list, got {v}"
                )))
            }
        };
        let upstream_prompts = opts.bool("upstream_validation_prompts", false)?;
        let prompts: Vec<String> = match (cfg.sample_prompts.is_empty(), upstream_prompts) {
            (true, true) => DEFAULT_VALIDATION_PROMPTS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            (true, false) => Vec::new(),
            (false, false) => cfg
                .sample_prompts
                .iter()
                .take(PREVIEW_PROMPT_CAP)
                .cloned()
                .collect(),
            (false, true) => {
                return Err(Error::Msg(
                    "iris training: upstream_validation_prompts renders upstream's default \
                     prompts in place of sample_prompts — set one or the other"
                        .into(),
                ))
            }
        };
        if cfg.sample_every > 0 && !prompts.is_empty() && cfg.sample_steps == 0 {
            return Err(Error::Msg(
                "iris training: sample_steps must be >= 1".into(),
            ));
        }
        let preview = PreviewPlan {
            // Empty prompts disable previews whatever the cadence (the shared contract).
            every: if prompts.is_empty() {
                0
            } else {
                cfg.sample_every / accum_u32
            },
            prompts,
            steps: cfg.sample_steps as usize,
            cfg_scale: cfg.sample_guidance_scale,
            negative_prompt: opts.string("preview_negative_prompt")?.unwrap_or_default(),
            weights: preview_weights,
            size: image_size,
            seed: cfg.seed,
        };
        let caption_field = opts
            .string("caption_field")?
            .unwrap_or_else(|| "caption".into());
        Ok(Self {
            artifact,
            init,
            image_size,
            batch_size,
            grad_accum,
            max_steps: cfg.steps / accum_u32,
            num_epochs,
            seed: cfg.seed,
            mixed_precision,
            flow,
            optimizer,
            schedule,
            warmup_steps: cfg.lr_warmup_steps.div_ceil(accum_u32) as u64,
            gradient_clip,
            text_dropout,
            ema,
            nan_loss_tolerance: opts.u64("nan_loss_tolerance")?.unwrap_or(20) as u32,
            caption_field,
            caption_fields: opts.strings("caption_fields")?,
            on_caption_overflow,
            text_conditioning,
            save_every: cfg.save_every / accum_u32,
            keep_last_checkpoints: opts.u64("keep_last_checkpoints")?.unwrap_or(0) as usize,
            milestone_steps,
            preview,
            resume: cfg.resume,
            resume_from: opts.string("resume_from")?.map(PathBuf::from),
            resume_data_policy,
            override_lr_on_resume: opts.bool("override_lr_on_resume", false)?,
            export_weights,
            export_bf16,
        })
    }

    /// The stage token count (`(image_size / patch)²`).
    pub fn tokens(&self, patch: usize) -> usize {
        (self.image_size / patch) * (self.image_size / patch)
    }

    /// The identity a resume must match. `exact` also pins the data-shaping knobs; a `new_phase`
    /// continuation only needs the trained state to be compatible (artifact, routing, EMA).
    pub fn state_identity(&self) -> String {
        let artifact = match &self.artifact {
            ArtifactPlan::Full => "full".to_string(),
            ArtifactPlan::Lora {
                rank,
                alpha,
                targets,
            } => format!("lora:{rank}:{alpha}:{}", digest_strings(targets)),
            ArtifactPlan::Lokr {
                rank,
                alpha,
                decompose_factor,
                targets,
            } => format!(
                "lokr:{rank}:{alpha}:{decompose_factor}:{}",
                digest_strings(targets)
            ),
        };
        format!(
            "artifact={artifact};optimizer={};ema={}",
            self.optimizer.kind.as_str(),
            self.ema.is_some()
        )
    }

    /// The data-shaping identity an `exact` resume additionally requires.
    pub fn data_identity(&self) -> String {
        format!(
            "image_size={};batch={};accum={};seed={};caption_field={};caption_fields={:?};\
             text_dropout={};sampler={:?};prediction={};shift={}",
            self.image_size,
            self.batch_size,
            self.grad_accum,
            self.seed,
            self.caption_field,
            self.caption_fields,
            self.text_dropout,
            self.flow.sampler,
            prediction_name(self.flow.prediction),
            self.flow.shift
        )
    }
}

fn digest_strings(items: &[String]) -> String {
    let mut h = Sha256::new();
    for s in items {
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
}

/// The request's dataset fingerprint: [`crate::train::resume::request_fingerprint`] (resolution,
/// order, captions, image contents) plus every item's named-captions map.
pub fn dataset_fingerprint(req: &TrainingRequest) -> Result<String> {
    let base = crate::train::resume::request_fingerprint(req)?;
    let mut h = Sha256::new();
    h.update(base.as_bytes());
    for item in &req.items {
        if let Some(m) = item.model_options.get(ITEM_CAPTIONS_KEY) {
            h.update(m.to_string().as_bytes());
        }
        h.update([0u8]);
    }
    Ok(format!("{:x}", h.finalize()))
}

// =============================================================================================
// Adapter targets and the adapter artifact schema
// =============================================================================================

/// Every Linear module path of a backbone, from its tensor keys: each `‹path›.weight` of rank 2
/// (`y_pos_embedding` is rank 3 and not a Linear). Sorted.
pub fn linear_paths_from_keys<'a>(keys: impl IntoIterator<Item = (&'a str, usize)>) -> Vec<String> {
    let mut out: Vec<String> = keys
        .into_iter()
        .filter(|(_, ndim)| *ndim == 2)
        .filter_map(|(k, _)| k.strip_suffix(".weight").map(str::to_string))
        .collect();
    out.sort();
    out
}

/// The adapter targets of a run. With no selectors: every Linear of the patch trunk
/// (`blocks.*` — attention projections, gates and SwiGLU MLPs of the dual- and single-stream
/// blocks). Each selector matches a module path exactly or as a dotted suffix (`attn.proj_x`,
/// `mlp.w1`, `qkv_x`); a selector that matches nothing is a refusal.
pub fn select_adapter_targets(
    linear_paths: &[String],
    selectors: &[String],
) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    if selectors.is_empty() {
        out = linear_paths
            .iter()
            .filter(|p| p.starts_with("blocks."))
            .cloned()
            .collect();
    } else {
        for sel in selectors {
            let sel = sel.trim();
            let hits: Vec<&String> = linear_paths
                .iter()
                .filter(|p| p.as_str() == sel || p.ends_with(&format!(".{sel}")))
                .collect();
            if hits.is_empty() {
                return Err(Error::Msg(format!(
                    "iris training: lora_target_modules entry {sel:?} matches no Linear of the \
                     Iris backbone"
                )));
            }
            for h in hits {
                if !out.contains(h) {
                    out.push(h.clone());
                }
            }
        }
        out.sort();
    }
    if out.is_empty() {
        return Err(Error::Msg(
            "iris training: no adapter targets resolved on the backbone".into(),
        ));
    }
    Ok(out)
}

/// `__metadata__` key: what an Iris training artifact is (`"adapter"` | `"full_model"`).
pub const META_ARTIFACT: &str = "irisArtifact";
/// `__metadata__` key: the task the artifact belongs to ([`super::IrisTask::name`]).
pub const META_TASK: &str = super::ADAPTER_TASK_KEY;
/// `__metadata__` key: whether the exported tensors are the EMA or the raw weights.
pub const META_WEIGHTS: &str = "irisWeights";
/// `__metadata__` key: optimizer steps trained.
pub const META_STEPS: &str = "irisTrainedSteps";
/// `__metadata__` key: the base backbone identity (see [`backbone_identity`]).
pub const META_BASE: &str = "irisBaseIdentity";
/// `__metadata__` key: the training objective's prediction type (`v` | `x`).
pub const META_PREDICTION: &str = "irisPrediction";
/// `__metadata__` key: the trained flow shift.
pub const META_SHIFT: &str = "irisFlowShift";
/// `__metadata__` key: the Iris code revision the trainer mirrors.
pub const META_UPSTREAM: &str = "irisUpstreamRevision";
/// `__metadata__` key: the JSON list of adapted module paths.
pub const META_TARGETS: &str = "irisTargets";

/// The **Iris adapter artifact** — one `.safetensors` file, readable with any safetensors reader:
///
/// * **LoRA** (`networkType = "lora"`): per adapted Linear at backbone module path `p`
///   (the upstream state-dict key minus `.weight`, e.g. `blocks.3.attn.q_proj_x`):
///   `p.lora_A.weight` `[rank, in]` f32, `p.lora_B.weight` `[out, rank]` f32, `p.alpha` `[1]` f32.
///   Application: `W ← W + strength · (alpha / rank) · B · A`.
/// * **LoKr** (`networkType = "lokr"`): `p.lokr_w1` `[out_a, in_a]` and either `p.lokr_w2`
///   `[out_b, in_b]` or the low-rank pair `p.lokr_w2_a` `[out_b, rank]` + `p.lokr_w2_b`
///   `[rank, in_b]` (LyCORIS factorization, `out = out_a·out_b`, `in = in_a·in_b`), metadata
///   `decomposeFactor`. Application: `W ← W + strength · (alpha / rank) · kron(w1, w2)` reshaped to
///   `[out, in]` (the repo's `reconstruct_lokr_delta`).
///
/// Metadata (all strings): `networkType`, `rank`, `alpha`, (`decomposeFactor`), `family = "iris"`,
/// `baseModel = "iris_3b"`, [`META_ARTIFACT`] `= "adapter"`, [`META_TASK`] `= "generation"`,
/// [`META_WEIGHTS`] (`ema` | `raw`), [`META_STEPS`], [`META_BASE`], [`META_PREDICTION`],
/// [`META_SHIFT`], [`META_UPSTREAM`], [`META_TARGETS`]. This is the same PEFT/LyCORIS key layout
/// every other family's adapter uses, so the shared adapter parsers read it unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterMetadata {
    pub network_type: String,
    pub rank: usize,
    pub alpha: f32,
    pub decompose_factor: Option<i32>,
    pub weights: WeightsSelect,
    pub steps: u64,
    pub base_identity: String,
    pub prediction: Prediction,
    pub shift: f64,
    pub targets: Vec<String>,
}

impl AdapterMetadata {
    pub fn to_map(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("networkType".into(), self.network_type.clone());
        m.insert("rank".into(), self.rank.to_string());
        m.insert("alpha".into(), self.alpha.to_string());
        if let Some(f) = self.decompose_factor {
            m.insert("decomposeFactor".into(), f.to_string());
        }
        // The adapter identity stamp the providers check (`check_adapter_identity`).
        for (k, v) in super::adapter_provenance(super::IrisTask::Generation, GENERATION_MODEL_ID) {
            m.insert(k.into(), v);
        }
        m.insert(META_ARTIFACT.into(), "adapter".into());
        m.insert(META_WEIGHTS.into(), self.weights.as_str().into());
        m.insert(META_STEPS.into(), self.steps.to_string());
        m.insert(META_BASE.into(), self.base_identity.clone());
        m.insert(
            META_PREDICTION.into(),
            prediction_name(self.prediction).into(),
        );
        m.insert(META_SHIFT.into(), self.shift.to_string());
        m.insert(META_UPSTREAM.into(), super::UPSTREAM_CODE_REVISION.into());
        m.insert(
            META_TARGETS.into(),
            JsonValue::from(self.targets.clone()).to_string(),
        );
        m
    }

    /// Read back (and check it is an Iris generation adapter).
    pub fn from_map(m: &BTreeMap<String, String>) -> Result<Self> {
        let get = |k: &str| -> Result<&String> {
            m.get(k)
                .ok_or_else(|| Error::Msg(format!("iris adapter: metadata key {k:?} is missing")))
        };
        if get(META_ARTIFACT)? != "adapter" || get(META_TASK)? != "generation" {
            return Err(Error::Msg(
                "iris adapter: not an Iris generation adapter artifact".into(),
            ));
        }
        let network_type = get("networkType")?.clone();
        if !matches!(network_type.as_str(), "lora" | "lokr") {
            return Err(Error::Msg(format!(
                "iris adapter: networkType {network_type:?} (lora | lokr)"
            )));
        }
        let num = |k: &str| -> Result<f64> {
            get(k)?
                .parse::<f64>()
                .map_err(|_| Error::Msg(format!("iris adapter: {k} is not a number")))
        };
        let targets: Vec<String> = serde_json::from_str::<JsonValue>(get(META_TARGETS)?)
            .ok()
            .and_then(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(str::to_string))
                        .collect()
                })
            })
            .ok_or_else(|| Error::Msg("iris adapter: irisTargets is not a JSON list".into()))?;
        Ok(Self {
            decompose_factor: if network_type == "lokr" {
                Some(num("decomposeFactor")? as i32)
            } else {
                None
            },
            network_type,
            rank: num("rank")? as usize,
            alpha: num("alpha")? as f32,
            weights: WeightsSelect::parse(META_WEIGHTS, get(META_WEIGHTS)?)?,
            steps: num(META_STEPS)? as u64,
            base_identity: get(META_BASE)?.clone(),
            prediction: parse_prediction(get(META_PREDICTION)?)?,
            shift: num(META_SHIFT)?,
            targets,
        })
    }
}

/// The **Iris full-model artifact** is a directory in exactly `scripts/export_checkpoint.py`'s
/// layout — `config.yaml` (the `model` / `text_encoder` / `flow` sections) + `model.safetensors`
/// (every backbone tensor under its upstream key, f32 or bf16) — so it loads as the generation
/// backbone resource (`LoadSpec::weights`) of the inference provider unchanged. The safetensors
/// metadata carries `format = "pt"`, [`META_ARTIFACT`] `= "full_model"`, [`META_TASK`],
/// [`META_WEIGHTS`], [`META_STEPS`], [`META_BASE`], [`META_UPSTREAM`].
pub fn full_model_metadata(
    weights: WeightsSelect,
    steps: u64,
    base_identity: &str,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("format".into(), "pt".into());
    m.insert(META_ARTIFACT.into(), "full_model".into());
    m.insert(META_TASK.into(), super::IrisTask::Generation.name().into());
    m.insert(META_WEIGHTS.into(), weights.as_str().into());
    m.insert(META_STEPS.into(), steps.to_string());
    m.insert(META_BASE.into(), base_identity.into());
    m.insert(META_UPSTREAM.into(), super::UPSTREAM_CODE_REVISION.into());
    m
}

/// A cheap, content-sensitive identity of a backbone resource: sha256 of its `config.yaml` bytes
/// and of the `model.safetensors` header (keys, dtypes, shapes, offsets) plus the file size.
/// `"random-init"` when the run has no source weights.
pub fn backbone_identity(dir: &Path, with_weights: bool) -> Result<String> {
    let mut h = Sha256::new();
    let cfg_path = dir.join(super::BACKBONE_CONFIG_FILE);
    let cfg = std::fs::read(&cfg_path)
        .map_err(|e| Error::Msg(format!("iris: read {}: {e}", cfg_path.display())))?;
    h.update(&cfg);
    if !with_weights {
        return Ok(format!("random-init:{:x}", h.finalize()));
    }
    let w_path = dir.join(super::BACKBONE_WEIGHTS_FILE);
    let mut f = std::fs::File::open(&w_path)
        .map_err(|e| Error::Msg(format!("iris: open {}: {e}", w_path.display())))?;
    let size = f
        .metadata()
        .map_err(|e| Error::Msg(format!("iris: stat {}: {e}", w_path.display())))?
        .len();
    let mut len = [0u8; 8];
    std::io::Read::read_exact(&mut f, &mut len)
        .map_err(|e| Error::Msg(format!("iris: read {}: {e}", w_path.display())))?;
    let n = u64::from_le_bytes(len).min(64 << 20) as usize;
    let mut header = vec![0u8; n];
    std::io::Read::read_exact(&mut f, &mut header)
        .map_err(|e| Error::Msg(format!("iris: read {}: {e}", w_path.display())))?;
    h.update(size.to_le_bytes());
    h.update(&header);
    Ok(format!("{:x}", h.finalize()))
}

// =============================================================================================
// Exported config.yaml
// =============================================================================================

/// Render the inference sections (`model` / `text_encoder` / `flow`) of an exported backbone, in
/// the YAML subset [`IrisConfig::parse`] reads. The flow section records the trained objective.
pub fn export_config_yaml(cfg: &IrisConfig, flow: &FlowObjective, adaln_zero_init: bool) -> String {
    let m = &cfg.model;
    let p = &m.pixel;
    let t = &cfg.text_encoder;
    let fl = |v: f64| -> String {
        // Exact round-trip: Rust's shortest repr parses back bit-identically.
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
            s
        } else {
            format!("{s}.0")
        }
    };
    let mut y = String::new();
    let mut line = |s: String| {
        y.push_str(&s);
        y.push('\n');
    };
    line("model:".into());
    line(format!("  block: {}", m.block));
    line(format!("  dual_depth: {}", m.dual_depth));
    line(format!("  final_block_text: {}", m.final_block_text));
    line(format!("  hidden_size: {}", m.hidden_size));
    line(format!("  depth: {}", m.depth));
    line(format!("  num_heads: {}", m.num_heads));
    line(format!(
        "  num_kv_heads: {}",
        m.num_kv_heads
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into())
    ));
    line(format!("  gated_attention: {}", m.gated_attention));
    line(format!("  sandwich_norm: {}", m.sandwich_norm));
    line(format!("  patch_size: {}", m.patch_size));
    line(format!("  in_channels: {}", m.in_channels));
    line(format!("  mlp_ratio: {}", fl(m.mlp_ratio)));
    line(format!("  qkv_bias: {}", m.qkv_bias));
    line(format!("  qk_norm: {}", m.qk_norm));
    line(format!("  norm_eps: {}", fl(m.norm_eps)));
    line(format!("  modulation: {}", m.modulation));
    line(format!(
        "  timestep_max_period: {}",
        fl(m.timestep_max_period)
    ));
    line(format!("  adaln_zero_init: {adaln_zero_init}"));
    line(format!("  rope_theta: {}", fl(m.rope_theta)));
    line(format!("  rope_scale: {}", fl(m.rope_scale)));
    line(format!("  rope_aspect: {}", m.rope_aspect));
    line(format!("  rope_frame_pairs: {}", m.rope_frame_pairs));
    line(format!("  text_rope: {}", m.text_rope));
    line(format!("  text_rope_theta: {}", fl(m.text_rope_theta)));
    line(format!("  text_abs_pos_embed: {}", m.text_abs_pos_embed));
    line(format!("  text_dim: {}", m.text_dim));
    line(format!("  text_len: {}", m.text_len));
    line(format!("  text_adapter: {}", m.text_adapter));
    line(format!("  text_lap_num_layers: {}", m.text_lap_num_layers));
    line(format!("  text_lap_num_heads: {}", m.text_lap_num_heads));
    line(format!(
        "  text_lap_mlp_ratio: {}",
        fl(m.text_lap_mlp_ratio)
    ));
    line("  pixel:".into());
    line(format!("    enabled: {}", p.enabled));
    line(format!("    depth: {}", p.depth));
    line(format!("    hidden_size: {}", p.hidden_size));
    line(format!("    attn_hidden_size: {}", p.attn_hidden_size));
    line(format!("    num_heads: {}", p.num_heads));
    line(format!("    mlp_ratio: {}", fl(p.mlp_ratio)));
    line(format!("    modulation: {}", p.modulation));
    line(format!("    abs_pos_embed: {}", p.abs_pos_embed));
    line("text_encoder:".into());
    line(format!("  name: {}", t.name));
    line(format!("  pretrained: {}", t.pretrained));
    line(format!("  dim: {}", t.dim));
    line(format!("  max_length: {}", t.max_length));
    line(format!("  dtype: {}", t.dtype));
    line("  hidden_layers:".into());
    for l in &t.hidden_layers {
        line(format!("  - {l}"));
    }
    line(format!("  on_caption_overflow: {}", t.on_caption_overflow));
    line("flow:".into());
    line(format!(
        "  num_train_timesteps: {}",
        flow.num_train_timesteps
    ));
    line(format!("  shift: {}", fl(flow.base_shift)));
    line(format!("  timestep_sampler: {}", flow.sampler.name()));
    let (mean, std) = match flow.sampler {
        TimestepSampler::LogitNormal { mean, std } => (mean, std),
        TimestepSampler::Uniform => (0.0, 1.0),
    };
    line(format!("  logit_mean: {}", fl(mean)));
    line(format!("  logit_std: {}", fl(std)));
    line(format!(
        "  prediction: {}",
        prediction_name(flow.prediction)
    ));
    line(format!("  x_pred_sigma_min: {}", fl(flow.x_pred_sigma_min)));
    line(format!("  shift_law: {}", flow.shift_law));
    line(format!("  shift_base_tokens: {}", flow.shift_base_tokens));
    y
}

// =============================================================================================
// Durable checkpoints (`train/ckpt.py` semantics: atomic publish, retention, latest)
// =============================================================================================

/// Checkpoint format tag (`state.json` `format`).
pub const CHECKPOINT_FORMAT: &str = "iris-train-checkpoint-v1";
/// File names inside one checkpoint directory.
pub const CKPT_STATE: &str = "state.json";
pub const CKPT_TRAINABLE: &str = "trainable.safetensors";
pub const CKPT_EMA: &str = "ema.safetensors";
pub const CKPT_OPTIMIZER: &str = "optimizer.safetensors";
/// The pointer file naming the newest complete checkpoint (`latest.pth` upstream).
pub const CKPT_LATEST: &str = "latest";

/// Where a run keeps its checkpoints: `<output_dir>/<stem>.checkpoints/`.
pub fn checkpoint_root(output_dir: &Path, file_name: &str) -> PathBuf {
    output_dir.join(format!("{}.checkpoints", file_stem(file_name)))
}

/// `step_00000042` — zero padded so a lexical sort is a step sort.
pub fn checkpoint_dir_name(step: u64) -> String {
    format!("step_{step:08}")
}

/// The stem of the output file name (`style.safetensors` → `style`).
pub fn file_stem(file_name: &str) -> String {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("iris")
        .to_string()
}

/// The resumable, non-tensor state of a run (`state.json`). Tensors live beside it:
/// [`CKPT_TRAINABLE`] (the trained tensors keyed as trained — backbone keys for a full run, the
/// adapter factor keys otherwise), [`CKPT_EMA`] (same keys), [`CKPT_OPTIMIZER`] (`‹key›::m`,
/// `‹key›::v` for AdamW-routed, `‹key›::momentum` for Muon-routed tensors).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointState {
    pub step: u64,
    /// Epoch (1-based) and batches consumed in it at this optimizer-step boundary.
    pub epoch: u32,
    pub batches_consumed: usize,
    /// `LambdaLR` scheduler steps taken (== optimizer steps).
    pub scheduler_step: u64,
    /// The LR the schedule scales (`initial_lr`; `override_lr_on_resume` replaces it).
    pub lr: f64,
    pub nan_count: u32,
    pub last_loss: f64,
    pub artifact: String,
    pub optimizer: String,
    pub ema_decay: Option<f64>,
    pub seed: u64,
    pub backend: String,
    pub state_identity: String,
    pub data_identity: String,
    pub dataset_fingerprint: String,
    pub base_identity: String,
}

impl CheckpointState {
    pub fn to_json(&self) -> JsonValue {
        json!({
            "format": CHECKPOINT_FORMAT,
            "trainer": TRAINER_ID,
            "task": super::IrisTask::Generation.name(),
            "step": self.step,
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
            "scheduler_step": self.scheduler_step,
            "lr": self.lr,
            "nan_count": self.nan_count,
            "last_loss": self.last_loss,
            "artifact": self.artifact,
            "optimizer": self.optimizer,
            "ema_decay": self.ema_decay,
            // u64 as a string: JSON numbers are f64 to most readers.
            "seed": self.seed.to_string(),
            "rng": "positional: mix_seed(seed, 0, epoch, position)",
            "backend": self.backend,
            "state_identity": self.state_identity,
            "data_identity": self.data_identity,
            "dataset_fingerprint": self.dataset_fingerprint,
            "base_identity": self.base_identity,
        })
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let bad = |k: &str| {
            Error::Msg(format!(
                "iris checkpoint: state.json field {k:?} is missing or malformed"
            ))
        };
        if v.get("format").and_then(JsonValue::as_str) != Some(CHECKPOINT_FORMAT) {
            return Err(Error::Msg(format!(
                "iris checkpoint: not a {CHECKPOINT_FORMAT} state.json"
            )));
        }
        let u = |k: &str| v.get(k).and_then(JsonValue::as_u64).ok_or_else(|| bad(k));
        let f = |k: &str| v.get(k).and_then(JsonValue::as_f64).ok_or_else(|| bad(k));
        let s = |k: &str| {
            v.get(k)
                .and_then(JsonValue::as_str)
                .map(str::to_string)
                .ok_or_else(|| bad(k))
        };
        Ok(Self {
            step: u("step")?,
            epoch: u("epoch")? as u32,
            batches_consumed: u("batches_consumed")? as usize,
            scheduler_step: u("scheduler_step")?,
            lr: f("lr")?,
            nan_count: u("nan_count")? as u32,
            last_loss: f("last_loss")?,
            artifact: s("artifact")?,
            optimizer: s("optimizer")?,
            ema_decay: v.get("ema_decay").and_then(JsonValue::as_f64),
            seed: s("seed")?.parse().map_err(|_| bad("seed"))?,
            backend: s("backend")?,
            state_identity: s("state_identity")?,
            data_identity: s("data_identity")?,
            dataset_fingerprint: s("dataset_fingerprint")?,
            base_identity: s("base_identity")?,
        })
    }
}

/// Write `bytes` to `path` atomically: a sibling temp file, fsync, rename (a crash never leaves a
/// truncated file where a reader would find it).
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_file_name(format!(
        "{}.tmp{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    let io = |what: &str, e: std::io::Error| {
        Error::Msg(format!("iris checkpoint: {what} {}: {e}", tmp.display()))
    };
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| io("create", e))?;
        f.write_all(bytes).map_err(|e| io("write", e))?;
        f.sync_all().map_err(|e| io("sync", e))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| io("rename", e))?;
    Ok(())
}

/// The staging directory a checkpoint is written into before it is published by rename.
pub fn checkpoint_staging_dir(root: &Path, step: u64) -> PathBuf {
    root.join(format!(
        ".{}.tmp{}",
        checkpoint_dir_name(step),
        std::process::id()
    ))
}

/// `fsync` every regular file directly inside `dir`, then (best effort, as directory handles are
/// not syncable on every platform) `dir` itself. [`publish_checkpoint`] runs it on the staging
/// directory before the publishing rename; a trainer that publishes a directory by other means
/// calls it the same way.
pub fn sync_dir_files(dir: &Path) -> Result<()> {
    let io = |what: &str, p: &Path, e: std::io::Error| {
        Error::Msg(format!("iris checkpoint: {what} {}: {e}", p.display()))
    };
    for entry in std::fs::read_dir(dir).map_err(|e| io("list", dir, e))? {
        let path = entry.map_err(|e| io("list", dir, e))?.path();
        if path.is_file() {
            // Write access: Windows' `FlushFileBuffers` refuses a read-only handle ("Access is
            // denied"); POSIX `fsync` accepts either.
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|f| f.sync_all())
                .map_err(|e| io("sync", &path, e))?;
        }
    }
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// Publish a fully written staging directory as `step_XXXXXXXX` (replacing an older copy of the
/// same step), fsync, then repoint [`CKPT_LATEST`] atomically. A checkpoint is complete exactly
/// when its directory exists under its final name: readers never see a half-written one.
pub fn publish_checkpoint(root: &Path, staging: &Path, step: u64) -> Result<PathBuf> {
    let final_dir = root.join(checkpoint_dir_name(step));
    let io = |what: &str, p: &Path, e: std::io::Error| {
        Error::Msg(format!("iris checkpoint: {what} {}: {e}", p.display()))
    };
    if !staging.join(CKPT_STATE).is_file() {
        return Err(Error::Msg(format!(
            "iris checkpoint: staging directory {} has no {CKPT_STATE}",
            staging.display()
        )));
    }
    // Every staged file reaches the disk before the directory becomes visible under its final
    // name, so a crash right after the rename can never publish a truncated tensor file.
    sync_dir_files(staging)?;
    if final_dir.exists() {
        std::fs::remove_dir_all(&final_dir).map_err(|e| io("replace", &final_dir, e))?;
    }
    std::fs::rename(staging, &final_dir).map_err(|e| io("publish", &final_dir, e))?;
    if let Ok(d) = std::fs::File::open(root) {
        let _ = d.sync_all();
    }
    atomic_write(
        &root.join(CKPT_LATEST),
        checkpoint_dir_name(step).as_bytes(),
    )?;
    Ok(final_dir)
}

/// Every complete checkpoint under `root`, ascending by step.
pub fn list_checkpoints(root: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<(u64, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let step = name.strip_prefix("step_")?.parse::<u64>().ok()?;
            let path = e.path();
            path.join(CKPT_STATE).is_file().then_some((step, path))
        })
        .collect();
    out.sort();
    out
}

/// The newest complete checkpoint: the [`CKPT_LATEST`] pointer when it names a complete one, else
/// the highest complete step (a pointer write lost to a crash cannot hide a published checkpoint).
pub fn latest_checkpoint(root: &Path) -> Option<PathBuf> {
    let all = list_checkpoints(root);
    if let Ok(name) = std::fs::read_to_string(root.join(CKPT_LATEST)) {
        let p = root.join(name.trim());
        if p.join(CKPT_STATE).is_file() {
            let pointed = name
                .trim()
                .strip_prefix("step_")
                .and_then(|s| s.parse::<u64>().ok());
            if pointed.is_some() && pointed == all.last().map(|(s, _)| *s) {
                return Some(p);
            }
        }
    }
    all.last().map(|(_, p)| p.clone())
}

/// Retention (`prune_checkpoints`): with `keep_last > 0`, delete all but the newest `keep_last`
/// step checkpoints, never a milestone step. Returns the removed steps.
pub fn prune_checkpoints(root: &Path, keep_last: usize, milestones: &[u64]) -> Result<Vec<u64>> {
    if keep_last == 0 {
        return Ok(Vec::new());
    }
    let all = list_checkpoints(root);
    let n = all.len();
    let mut removed = Vec::new();
    for (i, (step, path)) in all.into_iter().enumerate() {
        if i + keep_last >= n || milestones.contains(&step) {
            continue;
        }
        std::fs::remove_dir_all(&path)
            .map_err(|e| Error::Msg(format!("iris checkpoint: prune {}: {e}", path.display())))?;
        removed.push(step);
    }
    Ok(removed)
}

/// Read a checkpoint's `state.json`.
pub fn read_checkpoint_state(dir: &Path) -> Result<CheckpointState> {
    let path = dir.join(CKPT_STATE);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Msg(format!("iris checkpoint: read {}: {e}", path.display())))?;
    let v: JsonValue = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("iris checkpoint: parse {}: {e}", path.display())))?;
    CheckpointState::from_json(&v)
}

/// Check a checkpoint against the resuming run: the trained state must be compatible always; an
/// `exact` resume also needs the same data shaping and dataset.
pub fn check_resume(
    saved: &CheckpointState,
    plan: &IrisTrainPlan,
    dataset_fingerprint: &str,
    base_identity: &str,
) -> Result<()> {
    if saved.state_identity != plan.state_identity() {
        return Err(Error::Msg(format!(
            "iris resume: the checkpoint trained a different state ({}) than this run ({}); \
             use load_from for a weights-only start",
            saved.state_identity,
            plan.state_identity()
        )));
    }
    if !matches!(plan.artifact, ArtifactPlan::Full) && saved.base_identity != base_identity {
        return Err(Error::Msg(
            "iris resume: the adapter checkpoint was trained on a different backbone".into(),
        ));
    }
    if plan.resume_data_policy == ResumeDataPolicy::Exact {
        if saved.data_identity != plan.data_identity() {
            return Err(Error::Msg(format!(
                "iris resume: data shaping differs (saved {}, requested {}); resume_data_policy \
                 new_phase starts the current dataset fresh",
                saved.data_identity,
                plan.data_identity()
            )));
        }
        if saved.dataset_fingerprint != dataset_fingerprint {
            return Err(Error::Msg(
                "iris resume: the dataset differs from the checkpoint's; resume_data_policy \
                 new_phase starts the current dataset fresh"
                    .into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
