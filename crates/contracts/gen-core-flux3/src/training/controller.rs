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
    Ok(
        find_resume_candidate(root, stem, identity, policy, (format, seed), flag, false)?
            .map(|candidate| candidate.path),
    )
}

/// Reserved backend-private trajectory field, never part of adapter identity.
pub const CANDLE_EXECUTION_PROFILE_KEY: &str = "candle_execution_profile";

#[derive(Debug)]
pub struct ResumeProfileCandidate {
    pub path: PathBuf,
    pub execution_profile: Option<String>,
}

/// Metadata-only first phase for the Candle native owner. All existing format,
/// cursor, RNG, policy and other identity checks remain mandatory. Only the
/// reserved private profile field is returned for finite backend resolution.
/// The caller MUST run strict `find_resume` with its resolved private identity
/// before any learned preparation or restore; this is not a relaxed loader.
pub fn find_resume_profile_candidate(
    root: &Path,
    stem: &str,
    identity: &ResumeIdentity,
    policy: ResumePolicy,
    format: &str,
    seed: u64,
    flag: &CancelFlag,
) -> Result<Option<ResumeProfileCandidate>> {
    if identity
        .configuration
        .contains_key(CANDLE_EXECUTION_PROFILE_KEY)
    {
        return Err(invalid(
            "profile selection requires the unchanged export identity",
        ));
    }
    find_resume_candidate(root, stem, identity, policy, (format, seed), flag, true)
}

fn find_resume_candidate(
    root: &Path,
    stem: &str,
    identity: &ResumeIdentity,
    policy: ResumePolicy,
    format_seed: (&str, u64),
    flag: &CancelFlag,
    select_profile: bool,
) -> Result<Option<ResumeProfileCandidate>> {
    let (format, seed) = format_seed;
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
        let mut saved_identity = ResumeIdentity::from_metadata(field("identity")?)?;
        let execution_profile = if select_profile {
            saved_identity
                .configuration
                .remove(CANDLE_EXECUTION_PROFILE_KEY)
        } else {
            None
        };
        identity.ensure_matches(&saved_identity)?;
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
            best = Some((
                key,
                ResumeProfileCandidate {
                    path: entry.path(),
                    execution_profile,
                },
            ));
        }
    }
    Ok(best.map(|(_, candidate)| candidate))
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

#[cfg(test)]
mod private_profile_metadata_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn identity() -> ResumeIdentity {
        ResumeIdentity {
            base_repo: "synthetic-metadata-only".into(),
            base_revision: "a".repeat(40),
            artifacts: BTreeMap::from([("source".into(), "b".repeat(64))]),
            dataset: "c".repeat(64),
            preprocessing: "d".repeat(64),
            configuration: [
                "optimizer",
                "learning_rate",
                "rank",
                "alpha",
                "scaling",
                "precision",
                "backend",
                "scheduler",
                "accumulation",
                "rng",
            ]
            .into_iter()
            .map(|k| (k.into(), "metadata".into()))
            .collect(),
            factors: BTreeMap::from([("factor".into(), vec![2, 3])]),
        }
    }
    fn cursor(step: u64) -> TrainingCursor {
        TrainingCursor {
            micro_step: step,
            update_step: step,
            accumulation_count: 0,
            sampler_epoch: step / 4,
            sampler_cursor: step % 4,
            rng_state: rng::state_record(7, step),
            last_loss_bits: Some(0.25f32.to_bits()),
        }
    }
    fn policy() -> ResumePolicy {
        ResumePolicy {
            total_micro_steps: 8,
            accumulation: 1,
            sampler_len: 4,
        }
    }
    fn write(path: &Path, id: &ResumeIdentity, cursor: &TrainingCursor) {
        let metadata = BTreeMap::from([
            ("format", "native-metadata-test".to_owned()),
            ("identity", id.to_metadata().unwrap()),
            ("policy", policy().fingerprint()),
            ("cursor", cursor.to_metadata().unwrap()),
        ]);
        // Header-only fixture; no tensor entries/payload or device constructors.
        let mut header = serde_json::to_vec(&serde_json::json!({"__metadata__":metadata})).unwrap();
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn profile_probe_preserves_strict_default_and_all_non_profile_identity_fields() {
        let dir = tempfile::tempdir().unwrap();
        let base = identity();
        let mut private = base.clone();
        private.configuration.insert(
            CANDLE_EXECUTION_PROFILE_KEY.into(),
            "finite-private-binding".into(),
        );
        let path = dir.path().join("adapter-step000002.resume.safetensors");
        let cursor = cursor(2);
        write(&path, &private, &cursor);
        let bytes = std::fs::read(&path).unwrap();
        let original = base.to_metadata().unwrap();
        let rng = cursor.rng_state.clone();
        assert!(find_resume(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new()
        )
        .is_err());
        let candidate = find_resume_profile_candidate(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(candidate.path, path);
        assert_eq!(
            candidate.execution_profile.as_deref(),
            Some("finite-private-binding")
        );
        assert_eq!(
            find_resume(
                dir.path(),
                "adapter",
                &private,
                policy(),
                "native-metadata-test",
                7,
                &CancelFlag::new()
            )
            .unwrap(),
            Some(path.clone())
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(base.to_metadata().unwrap(), original);
        assert_eq!(cursor.rng_state, rng);
        let mut wrong = private.clone();
        wrong.dataset = "e".repeat(64);
        write(&path, &wrong, &cursor);
        assert!(find_resume_profile_candidate(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new()
        )
        .is_err());
    }

    #[test]
    fn native_actual_cursor_order_rng_and_bad_owned_candidate_rules_are_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let base = identity();
        let misleading = dir.path().join("adapter-step999999.resume.safetensors");
        write(&misleading, &base, &cursor(2));
        let newest = dir.path().join("adapter-step000001.resume.safetensors");
        write(&newest, &base, &cursor(6));
        let picked = find_resume_profile_candidate(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(picked.path, newest);
        assert!(picked.execution_profile.is_none());
        assert_eq!(
            find_resume(
                dir.path(),
                "adapter",
                &base,
                policy(),
                "native-metadata-test",
                7,
                &CancelFlag::new()
            )
            .unwrap(),
            Some(newest.clone())
        );
        let mut bad = cursor(6);
        bad.rng_state = rng::state_record(8, 6);
        write(&newest, &base, &bad);
        assert!(find_resume_profile_candidate(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new()
        )
        .is_err());
        write(&newest, &base, &cursor(6));
        std::fs::write(&misleading, b"bad-owned-header").unwrap();
        assert!(find_resume_profile_candidate(
            dir.path(),
            "adapter",
            &base,
            policy(),
            "native-metadata-test",
            7,
            &CancelFlag::new()
        )
        .is_err());
    }
}
