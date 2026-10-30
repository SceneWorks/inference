//! One source-specific lifecycle shared by real Candle/MLX training sessions.
//! Tensor work stays in backend session implementations; no Python/fallback route.
use super::{
    checkpoint::{ResumePolicy, TrainingCursor},
    policy::{optimizer_update_budget, SourceLrSchedule},
    rng,
    sampler::BatchSampler,
    ResumeIdentity,
};
use gen_core::{train::TrainingOutput, train::TrainingProgress, train::TrainingRequest};
use gen_core::{CancelFlag, Error, Result};
use std::path::{Path, PathBuf};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 trainer: {detail}"))
}
fn cancel(flag: &CancelFlag) -> Result<()> {
    if flag.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}

pub struct ExportReceipt {
    pub path: PathBuf,
    pub created: bool,
}
/// Implemented by actual native graph/optimizer/checkpoint owners. A micro-batch
/// adds LOSS/configured_N gradients to pending state; it does not update weights.
pub trait SourceTrainingSession {
    fn identity(&self) -> &ResumeIdentity;
    fn resume_format(&self) -> &'static str;
    fn restore(
        &mut self,
        path: &Path,
        policy: ResumePolicy,
        flag: &CancelFlag,
    ) -> Result<TrainingCursor>;
    fn micro_batch(&mut self, indices: &[usize], micro_step: u64, flag: &CancelFlag)
        -> Result<f32>;
    fn optimizer_update(
        &mut self,
        lr_multiplier: f32,
        max_grad_norm: f32,
        flag: &CancelFlag,
    ) -> Result<()>;
    fn checkpoint(
        &mut self,
        req: &TrainingRequest,
        cursor: &TrainingCursor,
        policy: ResumePolicy,
        flag: &CancelFlag,
    ) -> Result<()>;
    fn export(
        &mut self,
        req: &TrainingRequest,
        cursor: &TrainingCursor,
        flag: &CancelFlag,
    ) -> Result<ExportReceipt>;
    /// Delete only a newly created export owned by this session, never a reused file.
    fn discard_created_export(&self, receipt: &ExportReceipt) -> Result<()>;
}

/// Hoisted to [`gen_core_train::output_stem`] so every trainer validates its output name through
/// [`gen_core_train::validate_training_techniques`] (sc-27282); re-exported to keep this path stable.
pub use gen_core_train::output_stem;
pub fn checkpoint_name(stem: &str, step: u64, resume: bool) -> String {
    if resume {
        format!("{stem}-step{step:06}.resume.safetensors")
    } else {
        format!("{stem}-step{step:06}.safetensors")
    }
}

/// Select by ACTUAL metadata counters, not a directory/file step hint. A matching
/// owned candidate with bad identity/header fails actionably; never restart/skip it.
pub fn find_resume(
    root: &Path,
    stem: &str,
    identity: &ResumeIdentity,
    policy: ResumePolicy,
    format: &str,
    seed: u64,
    flag: &CancelFlag,
) -> Result<Option<PathBuf>> {
    cancel(flag)?;
    if !root.exists() {
        return Ok(None);
    }
    let prefix = format!("{stem}-step");
    let mut best = None;
    for entry in std::fs::read_dir(root)? {
        cancel(flag)?;
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&prefix) || !name.ends_with(".resume.safetensors") {
            continue;
        }
        if !entry.file_type()?.is_file() {
            return Err(invalid(
                "owned resume checkpoint must be a regular file, not a link/directory",
            ));
        }
        let metadata = gen_core::weightsmeta::safetensors_file_metadata(entry.path())?;
        let field = |key: &str| {
            metadata
                .get(key)
                .map(String::as_str)
                .ok_or_else(|| invalid(format!("resume metadata missing {key}")))
        };
        if field("format")? != format {
            return Err(invalid("resume backend/container format mismatch"));
        }
        identity.ensure_matches(&ResumeIdentity::from_metadata(field("identity")?)?)?;
        if field("policy")? != policy.fingerprint() {
            return Err(invalid(
                "effective run/sampler/accumulation policy mismatch",
            ));
        }
        let cursor = TrainingCursor::from_metadata(field("cursor")?)?;
        cursor.validate(
            policy.total_micro_steps,
            policy.accumulation,
            policy.sampler_len,
        )?;
        rng::validate_state(&cursor.rng_state, seed, cursor.micro_step)?;
        let key = (cursor.micro_step, cursor.update_step);
        if best.as_ref().is_none_or(|(current, _)| key > *current) {
            best = Some((key, entry.path()));
        }
    }
    Ok(best.map(|(_, path)| path))
}

pub fn ensure_new_output(req: &TrainingRequest) -> Result<()> {
    let stem = output_stem(&req.file_name)?;
    if !req.output_dir.is_absolute() {
        return Err(invalid("output directory must be absolute"));
    }
    if req.config.resume {
        return Ok(());
    }
    if req.output_dir.join(&req.file_name).exists() {
        return Err(invalid(
            "output adapter already exists; choose a new name or explicitly resume the same run",
        ));
    }
    if req.output_dir.exists() {
        let prefix = format!("{stem}-step");
        for entry in std::fs::read_dir(&req.output_dir)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".safetensors") {
                return Err(invalid(
                    "output already contains this run's checkpoints; choose a new name or resume",
                ));
            }
        }
    }
    Ok(())
}

pub struct RunPlan {
    pub sampler: BatchSampler,
    pub schedule: SourceLrSchedule,
    pub max_grad_norm: f32,
}
impl RunPlan {
    pub fn policy(&self, req: &TrainingRequest) -> ResumePolicy {
        ResumePolicy {
            total_micro_steps: req.config.steps as u64,
            accumulation: req.config.gradient_accumulation,
            sampler_len: self.sampler.len() as u64,
        }
    }
    pub fn validate(&self, req: &TrainingRequest) -> Result<()> {
        ensure_new_output(req)?;
        self.schedule.validate()?;
        if !self.max_grad_norm.is_finite() || self.max_grad_norm < 0.0 {
            return Err(invalid("invalid maximum gradient norm"));
        }
        let total = optimizer_update_budget(
            req.config.steps as u64,
            req.config.gradient_accumulation,
            self.sampler.len() as u64,
        )?;
        if total != self.schedule.total_updates {
            return Err(invalid(
                "LR update budget disagrees with epoch-tail accumulation policy",
            ));
        }
        Ok(())
    }
}

/// This is the production loop invoked by each registered native Trainer after
/// frozen cache preparation. Real native sessions own all gradients and tensors.
pub fn run<S: SourceTrainingSession>(
    session: &mut S,
    req: &TrainingRequest,
    plan: &RunPlan,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    cancel(&req.cancel)?;
    plan.validate(req)?;
    let policy = plan.policy(req);
    let mut cursor = if req.config.resume {
        let path=find_resume(&req.output_dir,output_stem(&req.file_name)?,session.identity(),policy,session.resume_format(),req.config.seed,&req.cancel)?
            .ok_or_else(||invalid("resume requested but no compatible saved state exists; refusing to restart from zero"))?;
        session.restore(&path, policy, &req.cancel)?
    } else {
        TrainingCursor {
            micro_step: 0,
            update_step: 0,
            accumulation_count: 0,
            sampler_epoch: 0,
            sampler_cursor: 0,
            rng_state: rng::state_record(req.config.seed, 0),
            last_loss_bits: None,
        }
    };
    cursor.validate(
        policy.total_micro_steps,
        policy.accumulation,
        policy.sampler_len,
    )?;
    rng::validate_state(&cursor.rng_state, req.config.seed, cursor.micro_step)?;
    let start = cursor.micro_step;
    let mut batches = plan
        .sampler
        .epoch_batches(cursor.sampler_epoch, &req.cancel)?;
    loop {
        cancel(&req.cancel)?;
        // A saved ready-full/epoch-tail/final-tail state is flushed BEFORE any new
        // sample is drawn. Pending gradients were divided by N BEFORE backward.
        let ready = cursor.accumulation_count >= policy.accumulation
            || cursor.sampler_cursor == policy.sampler_len
            || cursor.micro_step == policy.total_micro_steps;
        if cursor.accumulation_count > 0 && ready {
            let multiplier = plan.schedule.multiplier(cursor.update_step)? as f32;
            session.optimizer_update(multiplier, plan.max_grad_norm, &req.cancel)?;
            cursor.update_step += 1;
            cursor.accumulation_count = 0;
        }
        if cursor.micro_step >= policy.total_micro_steps {
            break;
        }
        if cursor.sampler_cursor == policy.sampler_len {
            cursor.sampler_epoch += 1;
            cursor.sampler_cursor = 0;
            plan.sampler
                .advance_epoch(&mut batches, cursor.sampler_epoch, &req.cancel)?;
        }
        let batch = &batches[cursor.sampler_cursor as usize];
        let loss = session.micro_batch(batch, cursor.micro_step, &req.cancel)?;
        if !loss.is_finite() || loss < 0.0 {
            return Err(invalid(
                "native loss became nonfinite/invalid; no successful adapter will be published",
            ));
        }
        cursor.micro_step += 1;
        cursor.sampler_cursor += 1;
        cursor.accumulation_count += 1;
        cursor.last_loss_bits = Some(loss.to_bits());
        cursor.rng_state = rng::state_record(req.config.seed, cursor.micro_step);
        let ready = cursor.accumulation_count >= policy.accumulation
            || cursor.sampler_cursor == policy.sampler_len
            || cursor.micro_step == policy.total_micro_steps;
        if ready {
            cancel(&req.cancel)?;
            session.optimizer_update(
                plan.schedule.multiplier(cursor.update_step)? as f32,
                plan.max_grad_norm,
                &req.cancel,
            )?;
            cursor.update_step += 1;
            cursor.accumulation_count = 0;
        }
        on_progress(TrainingProgress::Training {
            step: cursor.micro_step as u32,
            total: req.config.steps,
            loss,
        });
        cancel(&req.cancel)?;
        if req.config.save_every > 0
            && cursor
                .micro_step
                .is_multiple_of(req.config.save_every as u64)
        {
            session.checkpoint(req, &cursor, policy, &req.cancel)?;
            on_progress(TrainingProgress::Checkpoint {
                step: cursor.micro_step as u32,
            });
            cancel(&req.cancel)?;
        }
    }
    cursor.validate(
        policy.total_micro_steps,
        policy.accumulation,
        policy.sampler_len,
    )?;
    let final_loss=cursor.last_loss()?.ok_or_else(||invalid("completed resume state has no recorded observed loss; cannot fabricate a successful zero-loss result"))?;
    on_progress(TrainingProgress::Saving);
    cancel(&req.cancel)?;
    let receipt = session.export(req, &cursor, &req.cancel)?;
    if req.cancel.is_cancelled() {
        if receipt.created {
            session.discard_created_export(&receipt)?;
        }
        return Err(Error::Canceled);
    }
    Ok(TrainingOutput {
        adapter_path: receipt.path,
        steps: (cursor.micro_step - start) as u32,
        final_loss,
    })
}
