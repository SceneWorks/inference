//! Krea 2 **pose-ControlNet control-branch trainer** — the callable library form (sc-8462, epic
//! 8459).
//!
//! The S0 spike ([`crate::control`]) proved the recipe; this module lifts the spike CLI's training
//! loop into a reusable [`ControlTrainer`] so a caller other than the example binary — the SceneWorks
//! ControlNet Training Studio worker driver (epic 10159 B2 / sc-10163) — can drive a run and stream
//! its progress. The numerics are unchanged: every step still routes through
//! [`control_loss_grads`] and [`probe_forward`], so a run built here is bit-for-bit the run the
//! `krea-control-train` example produced (which now wraps this).
//!
//! **Seam.** The trainer owns the *optimization* concern only — the two optimizer groups (the run-3
//! magnitude-control fix: branch bodies at the full lr, zero-init projections at a reduced lr with
//! decoupled weight decay), gradient accumulation, clipping, LR warmup, checkpointing, and live
//! residual/main-RMS telemetry. It consumes already-encoded [`ControlSample`]s (VAE latent + control
//! latent + caption embedding, CPU-resident) — the heavy VAE/TE *encode* stays a data-prep concern
//! (the example's cache builder; the studio's ingest stage) so this stays gen-core-neutral and the
//! MLX training lane (epic 10159 B5) is a bounded add, not a rewrite.
//!
//! The frozen base ([`KreaTrainDit`]) and the trainable [`ControlBranch`] are built by the caller and
//! moved in — branch construction (block count, inject offset, dtype, residual clamp) lives on
//! [`ControlBranch`], not here.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::train::flow_match::{self, sample_noise, sample_unit_timestep};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::{CandleError, Result};

use crate::control::{control_loss_grads, probe_forward, ControlBranch};
use crate::train_dit::KreaTrainDit;

/// One pre-encoded training example, CPU-resident: the VAE-encoded target latent `x0`, the
/// VAE-encoded pose-skeleton control latent `ctrl`, and the caption embedding stack `cap` (the
/// unbatched `(L, layers, hidden)` the loss consumes). A GPU-resident cache scales with dataset size
/// — at 5k items it filled the card and WDDM-paged the run (~400 s/step); the trainer copies each
/// sample to the device for its micro-step only.
///
/// On-disk form (`ControlSample::{load, save}`): a flat `.safetensors` keyed `x0` / `ctrl` / `cap`.
/// This is the exact format the spike's encode cache wrote and the studio ingest stage should emit.
#[derive(Clone)]
pub struct ControlSample {
    /// Target-image VAE latent, f32 (the flow-match mix runs in f32).
    pub x0: Tensor,
    /// Pose-skeleton VAE control latent, f32.
    pub ctrl: Tensor,
    /// Caption embedding stack `(L, layers, hidden)`; typically bf16 (the DiT casts to bf16 at
    /// forward anyway — identical values, half the RAM).
    pub cap: Tensor,
}

impl ControlSample {
    /// Read a cached sample (`x0` / `ctrl` / `cap` keys) onto the CPU.
    pub fn load(path: &Path) -> Result<Self> {
        let cpu = Device::Cpu;
        let m = candle_gen::candle_core::safetensors::load(path, &cpu)?;
        let get = |k: &str| -> Result<Tensor> {
            m.get(k).cloned().ok_or_else(|| {
                CandleError::Msg(format!("control sample {} missing key {k}", path.display()))
            })
        };
        Ok(Self {
            x0: get("x0")?,
            ctrl: get("ctrl")?,
            cap: get("cap")?,
        })
    }

    /// Persist this sample to `path` in the `x0` / `ctrl` / `cap` `.safetensors` form.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut m = std::collections::HashMap::new();
        m.insert("x0".to_string(), self.x0.clone());
        m.insert("ctrl".to_string(), self.ctrl.clone());
        m.insert("cap".to_string(), self.cap.clone());
        candle_gen::candle_core::safetensors::save(&m, path)?;
        Ok(())
    }
}

/// Optimization hyperparameters for a [`ControlTrainer`] run. Branch *architecture* (block count,
/// inject offset, dtype, residual clamp) is set on the [`ControlBranch`] the caller builds, not here.
#[derive(Clone, Debug)]
pub struct ControlTrainConfig {
    /// Base learning rate for the branch-block bodies (AdamW, no decay).
    pub lr: f32,
    /// lr multiplier for the zero-init injection-projection group (run-3 fix; default 0.1).
    pub proj_lr_mult: f32,
    /// Decoupled weight decay for the injection-projection group (run-3 fix; default 0.05).
    pub proj_weight_decay: f32,
    /// Micro-steps per optimizer update — gradient accumulation, memory-flat in batch size.
    pub batch: u32,
    /// Optimizer updates this run performs (added to the trainer's current step on `run`).
    pub max_steps: u32,
    /// Linear LR warmup from 0 over the first N updates (0 = off).
    pub warmup_steps: u32,
    /// Timestep sampler for the rectified-flow σ (`sigmoid|uniform|linear|weighted`).
    pub timestep_type: String,
    /// Seed for the deterministic per-micro-step σ / noise draws.
    pub seed: u64,
    /// Gradient-checkpointed backward (default on — the dense backward OOMs ≥ 512² on a 96 GB card).
    pub grad_checkpoint: bool,
    /// MAE (L1) velocity loss instead of MSE (default false = MSE).
    pub mae: bool,
    /// Activation/compute dtype for the loss forward (bf16 on CUDA; f32 for CPU tests).
    pub compute_dtype: DType,
    /// Write a checkpoint every N updates (0 = only at the end of `run`).
    pub save_every: u32,
    /// Bucketed square edge the samples were encoded at — recorded in the checkpoint meta sidecar.
    pub resolution: u32,
    /// Control type this branch is trained for (`"pose"`/`"canny"`/`"depth"`/…). Recorded in the
    /// checkpoint/overlay meta `kind` (`"{control_type}_control_branch"`) so registration describes it
    /// correctly rather than a hardcoded label. `None` ⇒ `"pose"` (the first control type).
    pub control_type: Option<String>,
}

impl Default for ControlTrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            proj_lr_mult: 0.1,
            proj_weight_decay: 0.05,
            batch: 1,
            max_steps: 200,
            warmup_steps: 0,
            timestep_type: "uniform".into(),
            seed: 42,
            grad_checkpoint: true,
            mae: false,
            compute_dtype: DType::BF16,
            save_every: 100,
            resolution: 512,
            control_type: None,
        }
    }
}

/// Per-update training telemetry — the seam a driver streams into a job's progress feed.
#[derive(Clone, Debug)]
pub struct StepReport {
    /// 1-indexed optimizer update just completed.
    pub step: u32,
    /// Mean velocity loss over the batch's micro-steps.
    pub loss: f32,
    /// Pre-clip global gradient L2 norm.
    pub grad_norm: f64,
    /// Effective learning rate this update (after warmup).
    pub lr: f32,
    /// Wall-clock seconds for the update.
    pub secs: f32,
}

/// Events emitted over a [`ControlTrainer::run`], in occurrence order. A driver forwards these to the
/// job queue for live progress, loss curves, and resumability; the example CLI renders them to stdout
/// + a JSONL log.
#[derive(Clone, Debug)]
pub enum TrainEvent {
    /// An optimizer update completed.
    Step(StepReport),
    /// Fixed-probe residual/main-RMS telemetry (the run-3 early-warning): per injection point, the
    /// ratio `‖residual‖ / ‖main image tokens‖` pre- and post-clamp. A healthy run plateaus well
    /// under the clamp; a value pinned at the clamp ceiling is the stream-overwrite degeneracy.
    Telemetry {
        step: u32,
        pre: Vec<f64>,
        post: Vec<f64>,
    },
    /// A checkpoint (`control_step{step}.safetensors` + `.json` meta sidecar) was written.
    Checkpoint { step: u32, path: PathBuf },
}

/// Interval (in updates) between fixed-probe telemetry emissions during `run`.
const TELEMETRY_EVERY: u32 = 50;

/// Accepted `timestep_type` values (mirrors `sample_unit_timestep`'s recognized set — anything else
/// it silently treats as sigmoid, so the trainer rejects unknown values instead).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "uniform", "linear", "weighted"];

/// Drives control-branch training over a fixed set of pre-encoded [`ControlSample`]s against a frozen
/// [`KreaTrainDit`] base. Construct with [`ControlTrainer::new`], then either [`run`](Self::run) to
/// completion (streaming [`TrainEvent`]s) or single-step with [`step`](Self::step).
pub struct ControlTrainer {
    dit: KreaTrainDit,
    branch: ControlBranch,
    samples: Vec<ControlSample>,
    cfg: ControlTrainConfig,
    device: Device,
    /// All trainable branch vars (for accumulate/scale/clip over the combined gradient store).
    vars: Vec<candle_gen::candle_core::Var>,
    /// Branch-block bodies at the full lr, no decay.
    opt_body: TrainOptimizer,
    /// Zero-init injection projections at the reduced lr + decoupled weight decay.
    opt_proj: TrainOptimizer,
    /// Updates completed so far (nonzero when resumed from a checkpoint).
    step: u32,
    out_dir: PathBuf,
}

impl ControlTrainer {
    /// Wire a trainer over the caller-built frozen `dit` and trainable `branch` (with its clamp /
    /// inject-offset / dtype already set), the pre-encoded `samples`, and `cfg`. `start_step` is the
    /// resume point (0 for a fresh branch, the checkpoint's step when resuming). Checkpoints land in
    /// `out_dir`.
    pub fn new(
        dit: KreaTrainDit,
        branch: ControlBranch,
        samples: Vec<ControlSample>,
        cfg: ControlTrainConfig,
        out_dir: PathBuf,
        start_step: u32,
        device: Device,
    ) -> Result<Self> {
        if samples.is_empty() {
            return Err(CandleError::Msg("control trainer: no samples".into()));
        }
        if cfg.batch == 0 {
            return Err(CandleError::Msg(
                "control trainer: batch must be >= 1".into(),
            ));
        }
        // Reject an unsupported timestep sampler up front rather than let `sample_unit_timestep`
        // silently route it to its sigmoid fallback (repo policy: typed reject, never silent
        // fallback — the operator would otherwise train on a schedule they did not ask for).
        if !TIMESTEP_TYPES.contains(&cfg.timestep_type.as_str()) {
            return Err(CandleError::Msg(format!(
                "control trainer: unsupported timestep_type {:?} (expected one of {:?})",
                cfg.timestep_type, TIMESTEP_TYPES
            )));
        }
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| CandleError::Msg(format!("create out dir {}: {e}", out_dir.display())))?;
        let vars = branch.vars();
        // Two optimizer groups (run-3 fix): the branch-block bodies at the full lr with no decay, and
        // the injection projections at a reduced lr WITH decoupled weight decay — magnitude control on
        // the injection gain must be structural (AdamW's normalized steps otherwise regrow a zero-init
        // projection to unit gain regardless of lr). Both step from the same clipped GradStore; each
        // optimizer only updates its own vars.
        let opt_body = TrainOptimizer::from_config("adamw", branch.body_vars(), cfg.lr, 0.0)?;
        let opt_proj = TrainOptimizer::from_config(
            "adamw",
            branch.proj_vars(),
            cfg.lr * cfg.proj_lr_mult,
            cfg.proj_weight_decay,
        )?;
        Ok(Self {
            dit,
            branch,
            samples,
            cfg,
            device,
            vars,
            opt_body,
            opt_proj,
            step: start_step,
            out_dir,
        })
    }

    /// Updates completed so far.
    pub fn current_step(&self) -> u32 {
        self.step
    }

    /// Run one optimizer update: `batch` accumulated micro-steps, averaged + clipped gradients, warmup
    /// LR, both optimizer groups stepped. Advances the internal step counter and returns its
    /// [`StepReport`]. A caller stepping manually should inspect [`StepReport::loss`] for a non-finite
    /// value (divergence) — [`run`](Self::run) does this and aborts; a bare `step` loop must too.
    pub fn step(&mut self) -> Result<StepReport> {
        let t0 = Instant::now();
        let mut acc = None;
        let mut loss_sum = 0f32;
        for j in 0..self.cfg.batch {
            let micro = self.step * self.cfg.batch + j;
            // The cache is CPU-resident (dataset-size-independent VRAM); copy this sample's tensors to
            // the device for the micro-step only — they drop at the end of the iteration.
            let s = &self.samples[(micro as usize) % self.samples.len()];
            let x0 = s.x0.to_device(&self.device)?;
            let ctrl = s.ctrl.to_device(&self.device)?;
            let cap = s.cap.to_device(&self.device)?;
            let sigma = sample_unit_timestep(
                &self.cfg.timestep_type,
                "none",
                flow_match::timestep_seed(self.cfg.seed, micro),
            );
            let noise = sample_noise(
                x0.dims(),
                flow_match::noise_seed(self.cfg.seed, micro),
                &self.device,
            )?;
            let (loss, grads) = control_loss_grads(
                &self.dit,
                &self.branch,
                &x0,
                &ctrl,
                &cap,
                sigma,
                &noise,
                self.cfg.mae,
                self.cfg.compute_dtype,
                self.cfg.grad_checkpoint,
            )?;
            loss_sum += loss;
            accumulate_grads(&mut acc, grads, &self.vars)?;
        }
        let mut grads = acc.expect("batch >= 1");
        // Average the accumulated micro-step gradients; skip at batch == 1 (the common config) — the
        // ×1.0 pass would still reallocate every one of the ~3B-param branch gradient tensors.
        if self.cfg.batch > 1 {
            scale_grads(&mut grads, &self.vars, 1.0 / self.cfg.batch as f64)?;
        }
        let grad_norm = clip_grad_norm(&mut grads, &self.vars, 1.0)?;
        // Linear LR warmup from 0 over the first `warmup_steps` updates.
        let warm = if self.cfg.warmup_steps > 0 {
            ((self.step + 1) as f32 / self.cfg.warmup_steps as f32).min(1.0)
        } else {
            1.0
        };
        self.opt_body.set_lr_scaled(warm);
        self.opt_proj.set_lr_scaled(warm);
        self.opt_body.step(&grads)?;
        self.opt_proj.step(&grads)?;

        self.step += 1;
        Ok(StepReport {
            step: self.step,
            loss: loss_sum / self.cfg.batch as f32,
            grad_norm,
            lr: self.cfg.lr * warm,
            secs: t0.elapsed().as_secs_f32(),
        })
    }

    /// Fixed-probe residual/main-RMS telemetry (the run-3 early-warning): sample 0 at σ = 0.5 with a
    /// fixed noise draw, so the trajectory is comparable across steps. Graph-free (detached weight
    /// reads). Returns per-injection `(pre-clamp, post-clamp)` `‖res‖/‖main‖` ratios.
    pub fn probe(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        let s = &self.samples[0];
        let x0 = s.x0.to_device(&self.device)?;
        let ctrl = s.ctrl.to_device(&self.device)?;
        let cap = s.cap.to_device(&self.device)?.unsqueeze(0)?;
        let sigma = 0.5f32;
        let noise = sample_noise(
            x0.dims(),
            flow_match::noise_seed(self.cfg.seed, u32::MAX),
            &self.device,
        )?;
        let (x_t, _) = flow_match::build_batch(&x0, &noise, sigma as f64)?;
        let t = Tensor::from_vec(vec![sigma], (1,), &self.device)?;
        let (rep, _, _) = probe_forward(&self.dit, &self.branch, &x_t, &t, &cap, &ctrl, 1.0)?;
        let ratio = |num: f64, den: f64| (num / (den + 1e-9) * 1e4).round() / 1e4;
        let pre = rep.iter().map(|(p, _, m)| ratio(*p, *m)).collect();
        let post = rep.iter().map(|(_, q, m)| ratio(*q, *m)).collect();
        Ok((pre, post))
    }

    /// Overlay meta sidecar contents: the branch's block count + encode resolution, the base model,
    /// and the `kind` derived from `cfg.control_type` (`"{control_type}_control_branch"`, default
    /// `"pose"`). `step` is included for intermediate checkpoints, omitted for a final overlay.
    fn overlay_meta(&self, step: Option<u32>, base_model: &str) -> serde_json::Value {
        let kind = format!(
            "{}_control_branch",
            self.cfg.control_type.as_deref().unwrap_or("pose")
        );
        let mut meta = serde_json::json!({
            "n_blocks": self.branch.num_blocks(),
            "baseModel": base_model,
            "family": "krea_2",
            "kind": kind,
            "resolution": self.cfg.resolution,
        });
        if let Some(step) = step {
            meta["step"] = serde_json::json!(step);
        }
        meta
    }

    /// Write an intermediate checkpoint (`control_step{step}.safetensors` + a `.json` meta sidecar)
    /// into `out_dir`; returns its path.
    pub fn save_checkpoint(&self, step: u32) -> Result<PathBuf> {
        let path = self.out_dir.join(format!("control_step{step}.safetensors"));
        self.branch.save(&path)?;
        std::fs::write(
            path.with_extension("json"),
            self.overlay_meta(Some(step), "krea_2_turbo").to_string(),
        )
        .map_err(|e| CandleError::Msg(format!("write checkpoint meta: {e}")))?;
        Ok(path)
    }

    /// Save the trained branch as a **final overlay** to an explicit `path` (e.g. the studio's
    /// `output_dir/file_name`), with a `.json` meta sidecar stamped with `base_model` + the
    /// control-type `kind`. Unlike [`save_checkpoint`](Self::save_checkpoint) the meta omits `step`.
    pub fn save_overlay(&self, path: &Path, base_model: &str) -> Result<()> {
        self.branch.save(path)?;
        std::fs::write(
            path.with_extension("json"),
            self.overlay_meta(None, base_model).to_string(),
        )
        .map_err(|e| CandleError::Msg(format!("write overlay meta: {e}")))?;
        Ok(())
    }

    /// Train for `cfg.max_steps` further updates, streaming [`TrainEvent`]s to `on_event` (a step
    /// report every update, fixed-probe telemetry every 50, and each checkpoint — the last carrying its
    /// path). Aborts with an error on a non-finite loss, *after* emitting that step's report so the
    /// divergence is visible in the caller's log. A zero `max_steps` is a clean no-op.
    pub fn run(&mut self, mut on_event: impl FnMut(&TrainEvent)) -> Result<()> {
        let start = self.step;
        let target = self.step + self.cfg.max_steps;
        while self.step < target {
            let at_start = self.step == start;
            let rep = self.step()?;
            let diverged = !rep.loss.is_finite();
            let step_no = rep.step;
            on_event(&TrainEvent::Step(rep));
            if diverged {
                return Err(CandleError::Msg(format!(
                    "non-finite loss at step {step_no}"
                )));
            }

            if self.step.is_multiple_of(TELEMETRY_EVERY) || at_start {
                let (pre, post) = self.probe()?;
                on_event(&TrainEvent::Telemetry {
                    step: self.step,
                    pre,
                    post,
                });
            }

            let done = self.step == target;
            if (self.cfg.save_every > 0 && self.step.is_multiple_of(self.cfg.save_every)) || done {
                let path = self.save_checkpoint(self.step)?;
                on_event(&TrainEvent::Checkpoint {
                    step: self.step,
                    path,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::Weights;
    use crate::testfix::{randn_seeded, tiny_batch_seeded, tiny_dit_seeded};
    use rand::{rngs::StdRng, SeedableRng};

    /// The trainer's optimizer loop lowers the loss on a fixed held-out probe point — the library twin
    /// of `control::tests::backward_reaches_branch_and_descends`, exercising `ControlTrainer::run`
    /// end-to-end (both optimizer groups, accumulation, clip, checkpoint) on the tiny DiT.
    #[test]
    fn control_trainer_descends() {
        let dev = Device::Cpu;
        // Draw the ENTIRE fixture — base DiT weights, branch nudge, probe batch, control latent —
        // from one seeded `StdRng`. candle's CPU `randn` is unseedable (it pulls the process-global
        // `rand::rng()`), so before sc-10794 every draw here was nondeterministic; a marginal descent
        // over a few steps could flip sign on an unlucky init (or on ubuntu's float reassociation vs
        // macos), red-failing CI. A fixed seed makes the loss trajectory reproducible run-to-run and
        // platform-to-platform; the 60-step budget then buys a large, unambiguous drop (see the
        // relative-floor assert below) so this still fails hard if the trainer stops learning.
        let mut rng = StdRng::seed_from_u64(10794);
        let (dit, c, path) = tiny_dit_seeded(&mut rng);
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32, 0).unwrap();
        // Nudge off the zero-init identity so there's a signal to descend (as the control tests do).
        for v in branch.vars() {
            v.set(&randn_seeded(&mut rng, 0.0, 0.02, v.as_tensor().dims()))
                .unwrap();
        }

        let (x0, cap, noise) = tiny_batch_seeded(&c, &mut rng);
        let ctrl = randn_seeded(&mut rng, 0.0, 1.0, x0.dims());
        // Fixed eval point (kept before the samples are moved into the trainer).
        let (ex0, ectrl, ecap, enoise) = (x0.clone(), ctrl.clone(), cap.clone(), noise.clone());
        let samples = vec![ControlSample { x0, ctrl, cap }];

        let cfg = ControlTrainConfig {
            lr: 1e-2,
            batch: 1,
            max_steps: 60,
            warmup_steps: 0,
            grad_checkpoint: false,
            compute_dtype: DType::F32,
            save_every: 0,
            resolution: 64,
            ..Default::default()
        };
        let out = std::env::temp_dir().join("krea-control-trainer-test");
        let mut tr = ControlTrainer::new(dit, branch, samples, cfg, out, 0, dev).unwrap();

        let eval = |tr: &ControlTrainer| -> f32 {
            control_loss_grads(
                &tr.dit,
                &tr.branch,
                &ex0,
                &ectrl,
                &ecap,
                0.5,
                &enoise,
                false,
                DType::F32,
                false,
            )
            .unwrap()
            .0
        };
        let before = eval(&tr);
        let mut ckpt = None;
        tr.run(|ev| {
            if let TrainEvent::Checkpoint { path, .. } = ev {
                ckpt = Some(path.clone());
            }
        })
        .unwrap();
        let after = eval(&tr);
        // A *correctly* working trainer drives the fixed-probe loss down by ~30% over these 60 steps
        // (seed 10794: 0.1359 -> 0.0950, ratio ~0.70). Assert a ≥10% relative drop rather than a bare
        // `after < before`: 0.90 sits ~20 points clear of the real ~0.70 ratio, so cross-platform float
        // reassociation (the ubuntu-vs-macos delta that flaked sc-10794) cannot lift it over the bar —
        // while a trainer that stopped learning (ratio ~1.0) still fails hard. This is a genuine
        // descent gate, NOT a `<= before + epsilon` no-op.
        assert!(
            after < before * 0.9,
            "trainer should lower the fixed-probe loss by >=10%: {before} -> {after} (ratio {})",
            after / before
        );
        assert!(
            ckpt.is_some_and(|p| p.exists()),
            "run must emit a final checkpoint event whose file exists"
        );
        let _ = std::fs::remove_file(path);
    }
}
