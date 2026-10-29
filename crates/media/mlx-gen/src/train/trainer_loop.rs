//! The shared MLX family-trainer loop (sc-27332): the setup helpers, the per-step flow-match draw
//! and the step → accumulate → update → progress → checkpoint → preview loop every MLX
//! flow-match trainer used to carry its own copy of.
//!
//! A family drives [`run_train_loop`] through the [`TrainLoopFamily`] trait: it supplies the one
//! micro-step's loss and gradients (its own forward + loss), the checkpoint writer and the preview
//! renders; the loop owns the cadence — gradient accumulation and the partial final window, the LR
//! schedule position, the `save_every` checkpoint cadence, the `sample_every` preview cadence, the
//! "cancelled before a single step" refusal and the final save. Everything a family does
//! differently is a trait method or an explicit parameter here, never a silently unified default.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::adapters::AdaptableHost;
use crate::media::Image;
use crate::train::checkpoint::{self, checkpoint_filename, ResumeMeta};
use crate::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, LoraParams, TrainAdapter,
};
use crate::train::perceptual::{step_sample, AuxDriver, AuxStep};
use crate::train::schedule::{lr_multiplier, schedule_updates};
use crate::train::vocab::{normalize_cfg, sample_timestep};
use crate::train::{TrainOptimizer, TrainingItem, TrainingProgress, TrainingRequest};
use crate::{CancelFlag, Error, NetworkType, Result, TrainingConfig, TrainingOutput};

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637) —
/// the one cap every MLX family trainer applies to `sample_prompts`.
pub const SAMPLE_PROMPT_CAP: usize = 4;

// ---------------------------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------------------------

/// The largest bucket edge (epic 2123 E7) — the one the pre-flight guards and the preview renders
/// size for. `0` for no edges.
pub fn largest_edge(edges: &[u32]) -> u32 {
    edges.iter().copied().max().unwrap_or(0)
}

/// Whether a free-form `train_dtype` names bf16 (`bf16` / `bfloat16`, case- and
/// surrounding-whitespace-insensitive). Anything else trains f32.
pub fn train_dtype_is_bf16(train_dtype: &str) -> bool {
    let t = train_dtype.trim();
    t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
}

/// The training compute dtype a free-form `train_dtype` selects: bf16 when
/// [`train_dtype_is_bf16`], else f32.
pub fn resolve_compute_dtype(train_dtype: &str) -> Dtype {
    if train_dtype_is_bf16(train_dtype) {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    }
}

/// How a trainer folds `loss_type` before matching `mae`/`l1`. The two forms disagree on inputs
/// the shared vocabulary check accepts (it matches through [`normalize_cfg`]), so each family keeps
/// the one it always used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossTypeFold {
    /// `to_ascii_lowercase()` only — surrounding whitespace, spaces and dashes are kept.
    Lowercase,
    /// [`normalize_cfg`] — case, whitespace and dashes folded, as the vocabulary check does.
    Normalized,
}

/// Whether `loss_type` selects the L1 (`mae` / `l1`) loss under `fold`.
pub fn mae_loss(loss_type: &str, fold: LossTypeFold) -> bool {
    let lt = match fold {
        LossTypeFold::Lowercase => loss_type.to_ascii_lowercase(),
        LossTypeFold::Normalized => normalize_cfg(loss_type),
    };
    lt == "mae" || lt == "l1"
}

/// The run's adapter-file stem: `file_name` without its extension, `fallback` when it has none
/// usable (`"lora"` for the adapter trainers).
pub fn adapter_stem(file_name: &str, fallback: &str) -> String {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback)
        .to_string()
}

/// The configured optimizer. AdamW with `weight_decay = 0` is Adam, so one optimizer covers both
/// choices: an `adam` request trains with no weight decay.
pub fn adapter_optimizer(cfg: &TrainingConfig) -> Result<TrainOptimizer> {
    let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
        0.0
    } else {
        cfg.weight_decay
    };
    TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)
}

/// Build the trainable adapter over `target_paths` on `host` — LoRA or LoKr per
/// `cfg.network_type`, at `cfg.rank` (and `cfg.decompose_factor` for LoKr), seeded by `cfg.seed`.
pub fn build_adapter<H: AdaptableHost>(
    host: &mut H,
    target_paths: &[String],
    cfg: &TrainingConfig,
) -> Result<(TrainAdapter, LoraParams)> {
    Ok(match cfg.network_type {
        NetworkType::Lora => {
            let (targets, params) =
                build_lora_targets(host, target_paths, cfg.rank as i32, cfg.seed)?;
            (TrainAdapter::Lora { targets }, params)
        }
        NetworkType::Lokr => {
            let (targets, params) = build_lokr_targets(
                host,
                target_paths,
                cfg.rank as i32,
                cfg.decompose_factor,
                cfg.seed,
            )?;
            (TrainAdapter::Lokr { targets }, params)
        }
    })
}

/// Resolve the config's target-module *suffixes* to full dotted paths by matching them against
/// `adaptable` (the host's adapter-routable paths) — the suffix match PEFT's
/// `LoraConfig(target_modules=…)` does. With no explicit `lora_target_modules` the suffixes are
/// `defaults`, restricted to paths starting with `default_scope` when one is given; an explicit
/// list matches anywhere.
pub fn resolve_suffix_targets(
    adaptable: Vec<String>,
    cfg: &TrainingConfig,
    defaults: &[&str],
    default_scope: Option<&str>,
) -> Vec<String> {
    let default = cfg.lora_target_modules.is_empty();
    let suffixes: Vec<String> = if default {
        defaults.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    let scope = if default { default_scope } else { None };
    adaptable
        .into_iter()
        .filter(|path| {
            scope.is_none_or(|prefix| path.starts_with(prefix))
                && suffixes
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Per block, the adapter-routable LOCAL paths trained on it (e.g. `"attn.to_q"`), in
/// `target_paths` order — the factors a gradient-checkpoint segment threads as explicit inputs.
/// Only `"{block_prefix}{i}.{local}"` paths with `i < n_blocks` are collected; every other target
/// trains dense.
pub fn block_local_targets(
    target_paths: &[String],
    block_prefix: &str,
    n_blocks: usize,
) -> Vec<Vec<String>> {
    let mut blocks: Vec<Vec<String>> = vec![Vec::new(); n_blocks];
    for path in target_paths {
        if let Some((idx, local)) = path
            .strip_prefix(block_prefix)
            .and_then(|rest| rest.split_once('.'))
        {
            if let Ok(i) = idx.parse::<usize>() {
                if i < n_blocks {
                    blocks[i].push(local.to_string());
                }
            }
        }
    }
    blocks
}

/// Continue from the latest resume snapshot of THIS adapter (`stem`) in `output_dir` when
/// `cfg.resume` is set (F-125): the restored factors and the snapshot's step/update position, with
/// the optimizer state restored into `opt`. `None` when resume is off or no snapshot exists.
pub fn resume_latest(
    cfg: &TrainingConfig,
    output_dir: &Path,
    stem: &str,
    opt: &mut TrainOptimizer,
) -> Result<Option<(LoraParams, ResumeMeta)>> {
    if !cfg.resume {
        return Ok(None);
    }
    match checkpoint::find_latest_resume(output_dir, stem) {
        Some((snapshot, _)) => checkpoint::load_resume(&snapshot, opt).map(Some),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------------------------
// Caching
// ---------------------------------------------------------------------------------------------

/// Walk the dataset for the caching pass: stop at a cancel (leaving whatever is cached), report
/// [`TrainingProgress::Caching`] per item, then hand the item to `cache_item`.
pub fn cache_items(
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
    mut cache_item: impl FnMut(&TrainingItem) -> Result<()>,
) -> Result<()> {
    let total = req.items.len() as u32;
    for (i, item) in req.items.iter().enumerate() {
        if req.cancel.is_cancelled() {
            break;
        }
        on_progress(TrainingProgress::Caching {
            current: i as u32 + 1,
            total,
        });
        cache_item(item)?;
    }
    Ok(())
}

/// Refuse an empty cache, disambiguating its two causes (sc-4895): a cancel tripped during caching
/// is a genuine cancellation (typed [`Error::Canceled`]); an empty cache with no cancel is a real
/// "no usable dataset items" error, prefixed with `label`.
pub fn ensure_cached(cached: usize, cancel: &CancelFlag, label: &str) -> Result<()> {
    if cached > 0 {
        return Ok(());
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    Err(Error::Msg(format!("{label}: no usable dataset items")))
}

/// Pre-encode the preview-sample prompts (at most [`SAMPLE_PROMPT_CAP`], in order) through
/// `encode`, while the family's text encoder is still resident. Empty — and `encode` never called
/// — when sampling is off (`sample_every == 0` or no prompts) or the run is already cancelled.
pub fn encode_preview_prompts<T>(
    cfg: &TrainingConfig,
    cancel: &CancelFlag,
    mut encode: impl FnMut(&str) -> Result<T>,
) -> Result<Vec<(String, T)>> {
    if cfg.sample_every == 0 || cfg.sample_prompts.is_empty() || cancel.is_cancelled() {
        return Ok(Vec::new());
    }
    let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
    for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
        caps.push((prompt.clone(), encode(prompt)?));
    }
    Ok(caps)
}

// ---------------------------------------------------------------------------------------------
// The per-step flow-match draw
// ---------------------------------------------------------------------------------------------

/// `(x_t, target)` for one sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` — the rectified-flow interpolant and velocity every MLX flow-match
/// trainer regresses (each family owns its own sign/timestep convention around it).
pub fn flow_velocity_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The step's raw flow-match noise level and its f32 noise of `shape`: the timestep draws first
/// (`sample_timestep` keyed `seed·0x9E37_79B9 + step`), then the standard normal (keyed
/// `(seed + step)·2 + 1`). One draw of each per micro-step.
pub fn flow_step_noise(cfg: &TrainingConfig, step: u32, shape: &[i32]) -> Result<(f32, Array)> {
    let t = sample_timestep(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
    )?;
    let noise = random::normal::<f32>(
        shape,
        None,
        None,
        Some(&random::key(
            cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
        )?),
    )?;
    Ok((t, noise))
}

/// One micro-step's draw ([`draw_flow_step`]).
pub struct FlowStepDraw<'a> {
    /// The item-major cache entry the step trains on.
    pub entry: usize,
    /// The step's noise level — the sampled one, or the aux-only window's remapped level.
    pub t: f32,
    /// The step's f32 noise, shaped like the entry's clean latent.
    pub noise: Array,
    /// The step's planned perceptual terms (epic 2123 E8); `None` without a perceptual path.
    pub aux: Option<AuxStep<'a>>,
}

/// Draw micro-step `step`: the (item, bucket) cache entry from `schedule` (through the perceptual
/// alternation when `perceptual` is set), its noise level and noise ([`flow_step_noise`], noise
/// shaped by `shape_of(entry)`), and the planned aux terms — an aux-only step trains at the noise
/// level remapped into the loss window.
pub fn draw_flow_step<'a, S: AsRef<[i32]>>(
    perceptual: Option<&'a mut AuxDriver>,
    step: u32,
    schedule: &gen_core::BucketSchedule,
    cfg: &TrainingConfig,
    shape_of: impl FnOnce(usize) -> S,
) -> Result<FlowStepDraw<'a>> {
    let sample = step_sample(perceptual, step, schedule);
    let entry = sample.entry;
    let (mut t, noise) = flow_step_noise(cfg, step, shape_of(entry).as_ref())?;
    let aux = sample.plan(t)?;
    if let Some(p) = &aux {
        t = p.plan.noise_level;
    }
    Ok(FlowStepDraw {
        entry,
        t,
        noise,
        aux,
    })
}

// ---------------------------------------------------------------------------------------------
// Previews
// ---------------------------------------------------------------------------------------------

/// Where a cadence's preview renders report: the run's progress callback (one
/// [`TrainingProgress::Sample`] per rendered prompt) and its warnings (sc-27451).
pub struct PreviewSink<'a> {
    /// The run's progress callback.
    pub on_progress: &'a mut dyn FnMut(TrainingProgress),
    /// The run's warnings.
    pub warnings: &'a mut Vec<gen_core::GenerationWarning>,
}

/// How a family logs and reports a failed preview render.
pub struct PreviewLog<'a> {
    /// The stderr log tag (e.g. `"[sc-5637]"`).
    pub tag: &'a str,
    /// The trainer id named in the log line.
    pub trainer: &'a str,
    /// What failed, as the log line names it (e.g. `"preview sample"`).
    pub what: &'a str,
    /// Whether a [`Error::Canceled`] render ends the cadence's previews (`false`: it is reported
    /// like any other failure and the next prompt's cancel check ends the walk).
    pub canceled_stops: bool,
    /// Whether a failure is also reported as a
    /// [`PREVIEW_SAMPLE_FAILED`](gen_core::train::PREVIEW_SAMPLE_FAILED) warning (`false`: stderr
    /// only).
    pub report_failures: bool,
}

/// The preview seed of prompt `index` at `step`.
pub fn preview_seed(seed: u64, step: u32, index: usize) -> u64 {
    seed.wrapping_add(step as u64)
        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
        .wrapping_add(index as u64)
}

/// Render one cadence's previews, best-effort: `render(prompt, seed)` per pre-encoded prompt in
/// order (stopping at a cancel), each success reported as a [`TrainingProgress::Sample`]. A failed
/// render never aborts training — it is logged (and reported) per `log`, and the walk continues.
#[allow(clippy::too_many_arguments)]
pub fn render_preview_prompts<S>(
    sink: &mut PreviewSink<'_>,
    step: u32,
    seed: u64,
    cancel: &CancelFlag,
    prompts: &[S],
    prompt_of: impl Fn(&S) -> &str,
    log: &PreviewLog<'_>,
    mut render: impl FnMut(&S, u64) -> Result<Image>,
) {
    let total = prompts.len() as u32;
    for (i, prompt) in prompts.iter().enumerate() {
        if cancel.is_cancelled() {
            break;
        }
        match render(prompt, preview_seed(seed, step, i)) {
            Ok(image) => (sink.on_progress)(TrainingProgress::Sample {
                step,
                index: i as u32 + 1,
                total,
                prompt: prompt_of(prompt).to_string(),
                image,
            }),
            Err(Error::Canceled) if log.canceled_stops => break,
            Err(e) => {
                eprintln!(
                    "{} {} {} failed at step {step} (prompt {}): {e} — skipping this preview, \
                     training continues",
                    log.tag,
                    log.trainer,
                    log.what,
                    i + 1
                );
                if log.report_failures {
                    sink.warnings.push(gen_core::train::preview_sample_failed(
                        step,
                        i as u32 + 1,
                        &e,
                    ));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------------------------

/// Writes a trained adapter in the diffusers/PEFT layout: `cfg.alpha`, `cfg.rank`,
/// `cfg.decompose_factor`, no key prefix, plus the family's `provenance` `__metadata__` entries
/// (empty for none).
pub struct AdapterSaver<'a> {
    /// The trained adapter's targets.
    pub adapter: &'a TrainAdapter,
    /// The run's config.
    pub cfg: &'a TrainingConfig,
    /// Extra `__metadata__` entries.
    pub provenance: &'a [(&'a str, &'a str)],
}

impl AdapterSaver<'_> {
    /// Write `params` to `path`.
    pub fn save(&self, params: &LoraParams, path: &Path) -> Result<()> {
        self.adapter.save_with_meta(
            params,
            self.cfg.alpha,
            self.cfg.rank as f32,
            self.cfg.decompose_factor,
            "",
            self.provenance,
            path,
        )
    }

    /// The intermediate checkpoint: the adapter at `{stem}-step{step}` in the output dir, then
    /// the resume bundle beside it.
    pub fn save_checkpoint(&self, at: &CheckpointAt<'_>) -> Result<()> {
        self.save_checkpoint_adapter(at)?;
        checkpoint::save_resume(
            at.output_dir,
            at.stem,
            at.step,
            at.update_idx,
            at.opt,
            at.params,
        )
    }

    /// Only the checkpoint's adapter file (creating the output dir).
    pub fn save_checkpoint_adapter(&self, at: &CheckpointAt<'_>) -> Result<()> {
        std::fs::create_dir_all(at.output_dir)?;
        let ckpt = at.output_dir.join(checkpoint_filename(at.stem, at.step));
        self.save(at.params, &ckpt)
    }

    /// The final adapter at the request's `output_dir/file_name` (creating the dir).
    pub fn save_final(&self, req: &TrainingRequest, params: &LoraParams) -> Result<PathBuf> {
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        self.save(params, &adapter_path)?;
        Ok(adapter_path)
    }
}

// ---------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------

/// The trainable state the loop starts from: the (possibly resumed) parameters and optimizer, and
/// the resume position (`0`/`0` for a fresh run).
pub struct LoopStart {
    /// The optimizer.
    pub opt: TrainOptimizer,
    /// The trainable parameters.
    pub params: LoraParams,
    /// The last completed micro-step; the loop runs `start_step + 1..=steps`.
    pub start_step: u32,
    /// The optimizer update index (the LR-schedule position).
    pub update_idx: u32,
}

impl LoopStart {
    /// A fresh run at step 0 / update 0.
    pub fn fresh(opt: TrainOptimizer, params: LoraParams) -> Self {
        Self {
            opt,
            params,
            start_step: 0,
            update_idx: 0,
        }
    }

    /// Continue from [`resume_latest`] when it finds a snapshot (its factors, step and update
    /// index replace the fresh ones; `on_resume` sees the snapshot's position), else unchanged.
    pub fn resume_latest(
        mut self,
        cfg: &TrainingConfig,
        output_dir: &Path,
        stem: &str,
        on_resume: impl FnOnce(&ResumeMeta),
    ) -> Result<Self> {
        if let Some((loaded, meta)) = resume_latest(cfg, output_dir, stem, &mut self.opt)? {
            self.params = loaded;
            self.start_step = meta.step;
            self.update_idx = meta.update_idx;
            on_resume(&meta);
        }
        Ok(self)
    }
}

/// One `save_every` checkpoint's position and state.
pub struct CheckpointAt<'a> {
    /// The run's output dir.
    pub output_dir: &'a Path,
    /// The run's adapter stem.
    pub stem: &'a str,
    /// The micro-step just completed.
    pub step: u32,
    /// The optimizer update index after this step.
    pub update_idx: u32,
    /// The gradient-accumulation window.
    pub accum: u32,
    /// The optimizer.
    pub opt: &'a TrainOptimizer,
    /// The trainable parameters.
    pub params: &'a LoraParams,
}

/// What a family supplies to [`run_train_loop`].
pub trait TrainLoopFamily {
    /// One micro-step's `(loss, grads)` at `params`.
    fn micro_step(&mut self, step: u32, params: &LoraParams) -> Result<(f32, LoraParams)>;

    /// Before micro-step `step` (diagnostic tracing only).
    fn begin_step(&mut self, _step: u32) {}

    /// After micro-step `step` returned its gradients (diagnostic tracing only).
    fn after_micro_step(&mut self, _step: u32) {}

    /// Before the step's progress report (diagnostic tracing only).
    fn before_progress(&mut self, _step: u32) {}

    /// Apply one optimizer update with the window-averaged gradients. Default: the adapter update
    /// (clip → gradient noise → step → weight noise, sc-24827), noise keyed by `cfg.seed`.
    fn optimizer_update(
        &mut self,
        opt: &mut TrainOptimizer,
        params: &mut LoraParams,
        avg: &LoraParams,
        cfg: &TrainingConfig,
        update_idx: u32,
    ) -> Result<()> {
        adapter_optimizer_update(opt, params, avg, cfg, update_idx, cfg.seed)
    }

    /// Write the `save_every` checkpoint `at`.
    fn save_checkpoint(&mut self, at: CheckpointAt<'_>) -> Result<()>;

    /// Whether this run renders previews at all (pre-encoded prompts exist).
    fn previews_enabled(&self) -> bool;

    /// Render step `step`'s previews from `params`.
    fn render_previews(
        &mut self,
        step: u32,
        params: &LoraParams,
        sink: &mut PreviewSink<'_>,
    ) -> Result<()>;

    /// At the very end of each step, after any checkpoint and previews.
    fn end_step(&mut self) {}

    /// Write the trained result; its path is the run's output.
    fn save_final(&mut self, params: &LoraParams) -> Result<PathBuf>;
}

/// Run micro-steps `start.start_step + 1..=steps` of `req` through `family`: each step's
/// gradients accumulate over `gradient_accumulation` micro-steps (the final window may be partial
/// and is averaged over its actual count, F-069), an update fires at each window end at the
/// scheduled LR, progress is reported per step, a checkpoint is written every `save_every` steps
/// (never on the final step) and previews render every `sample_every` steps. A run cancelled
/// before completing a single step is [`Error::Canceled`] — the factors are still the no-op init,
/// never written as a trained artifact (F-040).
pub fn run_train_loop<F: TrainLoopFamily + ?Sized>(
    family: &mut F,
    req: &TrainingRequest,
    stem: &str,
    start: LoopStart,
    on_progress: &mut dyn FnMut(TrainingProgress),
    warnings: &mut Vec<gen_core::GenerationWarning>,
) -> Result<TrainingOutput> {
    let cfg = &req.config;
    let LoopStart {
        mut opt,
        mut params,
        start_step,
        mut update_idx,
    } = start;
    let accum = cfg.gradient_accumulation.max(1);
    let (total_updates, warmup_updates) = schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
    let previews = family.previews_enabled();
    let mut accumulated: Option<LoraParams> = None;
    let mut last_loss = 0.0f32;
    let mut steps_run = start_step;
    for step in start_step + 1..=cfg.steps {
        if req.cancel.is_cancelled() {
            break;
        }
        family.begin_step(step);
        let (loss, grads) = family.micro_step(step, &params)?;
        last_loss = loss;
        family.after_micro_step(step);
        steps_run = step;
        accumulate_grads(&mut accumulated, grads)?;

        if step % accum == 0 || step == cfg.steps {
            opt.set_lr_scaled(lr_multiplier(
                cfg.lr_scheduler,
                update_idx,
                total_updates,
                warmup_updates,
            ));
            let window = if step % accum == 0 {
                accum
            } else {
                step % accum
            };
            let avg = average_grads(
                accumulated
                    .take()
                    .expect("an update fires only after accumulation"),
                window,
            )?;
            family.optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx)?;
            update_idx += 1;
        }

        family.before_progress(step);
        on_progress(TrainingProgress::Training {
            step,
            total: cfg.steps,
            loss: last_loss,
        });

        if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
            family.save_checkpoint(CheckpointAt {
                output_dir: &req.output_dir,
                stem,
                step,
                update_idx,
                accum,
                opt: &opt,
                params: &params,
            })?;
            on_progress(TrainingProgress::Checkpoint { step });
        }

        if previews && step % cfg.sample_every == 0 {
            family.render_previews(
                step,
                &params,
                &mut PreviewSink {
                    on_progress: &mut *on_progress,
                    warnings: &mut *warnings,
                },
            )?;
        }
        family.end_step();
    }

    if steps_run == 0 {
        return Err(Error::Canceled);
    }
    on_progress(TrainingProgress::Saving);
    let adapter_path = family.save_final(&params)?;
    Ok(TrainingOutput {
        adapter_path,
        steps: steps_run,
        final_loss: last_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_dtype_parse_matches_bf16_spellings_only() {
        for s in ["bf16", "BF16", " bfloat16 ", "BFloat16"] {
            assert!(train_dtype_is_bf16(s), "{s}");
            assert_eq!(resolve_compute_dtype(s), Dtype::Bfloat16);
        }
        for s in ["f32", "fp16", "", "bf 16"] {
            assert!(!train_dtype_is_bf16(s), "{s}");
            assert_eq!(resolve_compute_dtype(s), Dtype::Float32);
        }
    }

    #[test]
    fn mae_fold_keeps_each_familys_matching() {
        assert!(mae_loss("MAE", LossTypeFold::Lowercase));
        assert!(mae_loss("L1", LossTypeFold::Normalized));
        assert!(!mae_loss("mse", LossTypeFold::Normalized));
        // The two folds disagree on whitespace the vocabulary check accepts.
        assert!(!mae_loss(" mae", LossTypeFold::Lowercase));
        assert!(mae_loss(" mae", LossTypeFold::Normalized));
    }

    #[test]
    fn largest_edge_and_stem() {
        assert_eq!(largest_edge(&[512, 1024, 768]), 1024);
        assert_eq!(largest_edge(&[]), 0);
        assert_eq!(adapter_stem("my_lora.safetensors", "lora"), "my_lora");
        assert_eq!(adapter_stem("", "finetune"), "finetune");
    }

    #[test]
    fn suffix_targets_scope_only_the_defaults() {
        let paths = vec![
            "transformer_blocks.0.attn.to_q".to_string(),
            "text_fusion.to_q".to_string(),
            "transformer_blocks.0.attn.to_v".to_string(),
        ];
        let mut cfg = TrainingConfig::default();
        assert_eq!(
            resolve_suffix_targets(paths.clone(), &cfg, &["to_q"], Some("transformer_blocks.")),
            vec!["transformer_blocks.0.attn.to_q"]
        );
        assert_eq!(
            resolve_suffix_targets(paths.clone(), &cfg, &["to_q"], None),
            vec!["transformer_blocks.0.attn.to_q", "text_fusion.to_q"]
        );
        cfg.lora_target_modules = vec!["to_q".into()];
        assert_eq!(
            resolve_suffix_targets(paths, &cfg, &["to_v"], Some("transformer_blocks.")),
            vec!["transformer_blocks.0.attn.to_q", "text_fusion.to_q"]
        );
    }

    #[test]
    fn block_locals_collect_in_bounds_block_targets_in_order() {
        let paths: Vec<String> = [
            "layers.1.attention.to_q",
            "layers.0.attention.to_k",
            "layers.1.attention.to_v",
            "layers.9.attention.to_q",
            "refiner.0.to_q",
            "layers.x.to_q",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let blocks = block_local_targets(&paths, "layers.", 2);
        assert_eq!(
            blocks,
            vec![
                vec!["attention.to_k".to_string()],
                vec!["attention.to_q".to_string(), "attention.to_v".to_string()],
            ]
        );
    }

    #[test]
    fn preview_seed_is_the_trainers_mix() {
        let s = 7u64;
        assert_eq!(
            preview_seed(s, 3, 2),
            s.wrapping_add(3)
                .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                .wrapping_add(2)
        );
    }

    #[test]
    fn empty_cache_is_canceled_only_under_a_cancel() {
        let cancel = CancelFlag::new();
        assert!(ensure_cached(1, &cancel, "x trainer").is_ok());
        let err = ensure_cached(0, &cancel, "x trainer").unwrap_err();
        assert_eq!(err.to_string(), "x trainer: no usable dataset items");
        cancel.cancel();
        assert!(matches!(
            ensure_cached(0, &cancel, "x trainer"),
            Err(Error::Canceled)
        ));
    }

    #[test]
    fn preview_prompts_are_capped_and_gated() {
        let cancel = CancelFlag::new();
        let mut cfg = TrainingConfig {
            sample_every: 1,
            sample_prompts: (0..6).map(|i| format!("p{i}")).collect(),
            ..Default::default()
        };
        let caps = encode_preview_prompts(&cfg, &cancel, |p| Ok(p.len())).unwrap();
        assert_eq!(caps.len(), SAMPLE_PROMPT_CAP);
        assert_eq!(caps[3].0, "p3");
        cfg.sample_every = 0;
        let none: Vec<(String, usize)> =
            encode_preview_prompts(&cfg, &cancel, |_| panic!("never encoded")).unwrap();
        assert!(none.is_empty());
        cfg.sample_every = 1;
        cancel.cancel();
        let none: Vec<(String, usize)> =
            encode_preview_prompts(&cfg, &cancel, |_| panic!("never encoded")).unwrap();
        assert!(none.is_empty());
    }

    /// A loop family with no compute: a fixed loss and an empty gradient map, recording the
    /// cadence the loop drives.
    #[derive(Default)]
    struct Recorder {
        steps: Vec<u32>,
        updates: Vec<u32>,
        checkpoints: Vec<(u32, u32)>,
        previews: Vec<u32>,
        previews_on: bool,
        finals: u32,
    }

    impl TrainLoopFamily for Recorder {
        fn micro_step(&mut self, step: u32, _params: &LoraParams) -> Result<(f32, LoraParams)> {
            self.steps.push(step);
            Ok((step as f32, LoraParams::new()))
        }
        fn optimizer_update(
            &mut self,
            _opt: &mut TrainOptimizer,
            _params: &mut LoraParams,
            _avg: &LoraParams,
            _cfg: &TrainingConfig,
            update_idx: u32,
        ) -> Result<()> {
            self.updates.push(update_idx);
            Ok(())
        }
        fn save_checkpoint(&mut self, at: CheckpointAt<'_>) -> Result<()> {
            self.checkpoints.push((at.step, at.update_idx));
            Ok(())
        }
        fn previews_enabled(&self) -> bool {
            self.previews_on
        }
        fn render_previews(
            &mut self,
            step: u32,
            _params: &LoraParams,
            _sink: &mut PreviewSink<'_>,
        ) -> Result<()> {
            self.previews.push(step);
            Ok(())
        }
        fn save_final(&mut self, _params: &LoraParams) -> Result<PathBuf> {
            self.finals += 1;
            Ok(PathBuf::from("out.safetensors"))
        }
    }

    fn request(cfg: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: Vec::new(),
            config: cfg,
            output_dir: PathBuf::new(),
            file_name: String::new(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        }
    }

    fn start() -> LoopStart {
        LoopStart {
            opt: TrainOptimizer::from_config("adamw", 1e-4, 0.0).unwrap(),
            params: LoraParams::new(),
            start_step: 0,
            update_idx: 0,
        }
    }

    #[test]
    fn loop_drives_accumulation_checkpoint_and_preview_cadence() {
        let req = request(TrainingConfig {
            steps: 7,
            gradient_accumulation: 3,
            save_every: 2,
            sample_every: 3,
            ..Default::default()
        });
        let mut family = Recorder {
            previews_on: true,
            ..Default::default()
        };
        let mut progress = Vec::new();
        let mut warnings = Vec::new();
        let out = run_train_loop(
            &mut family,
            &req,
            "lora",
            start(),
            &mut |p| progress.push(p),
            &mut warnings,
        )
        .unwrap();
        assert_eq!(family.steps, (1..=7).collect::<Vec<_>>());
        // Updates at the window ends 3, 6 and the partial final window at 7.
        assert_eq!(family.updates, vec![0, 1, 2]);
        // save_every = 2, never on the final step; the update index after each step.
        assert_eq!(family.checkpoints, vec![(2, 0), (4, 1), (6, 2)]);
        assert_eq!(family.previews, vec![3, 6]);
        assert_eq!(family.finals, 1);
        assert_eq!((out.steps, out.final_loss), (7, 7.0));
        assert!(matches!(progress.last(), Some(TrainingProgress::Saving)));
    }

    #[test]
    fn loop_resumes_after_the_snapshot_and_skips_disabled_previews() {
        let req = request(TrainingConfig {
            steps: 4,
            gradient_accumulation: 1,
            sample_every: 1,
            ..Default::default()
        });
        let mut family = Recorder::default();
        let mut s = start();
        s.start_step = 2;
        s.update_idx = 2;
        run_train_loop(&mut family, &req, "lora", s, &mut |_| {}, &mut Vec::new()).unwrap();
        assert_eq!(family.steps, vec![3, 4]);
        assert_eq!(family.updates, vec![2, 3]);
        assert!(family.previews.is_empty());
    }

    #[test]
    fn loop_cancelled_before_a_step_writes_nothing() {
        let req = request(TrainingConfig {
            steps: 4,
            ..Default::default()
        });
        req.cancel.cancel();
        let mut family = Recorder::default();
        let err = run_train_loop(
            &mut family,
            &req,
            "lora",
            start(),
            &mut |_| {},
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Canceled));
        assert!(family.steps.is_empty());
        assert_eq!(family.finals, 0);
    }
}
