//! Immutable, job-owned selections of complete measured Reference training arms.
//!
//! This is an internal CUDA policy, not a user configuration or proof environment
//! switch. Bind it on the synchronous training thread before preparing a fresh
//! model. Suspend it around callbacks and preview inference. Worker threads must
//! explicitly receive and bind their own value; scopes never propagate implicitly.

use std::cell::Cell;
use std::collections::HashMap;

use crate::gen_core::train::TrainingRequest;
use crate::{CandleError, Result};

use super::comparison::ComparisonStage;

const RESUME_KEY: &str = "training_execution_profile";
const VERSION: &str = "measured-reference-v1";

/// Only complete arms actually present in the comparison protocol are selectable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainingProfile {
    #[default]
    Current,
    Legacy,
    WithoutE1,
    WithoutE2,
    WithoutE3,
}

impl TrainingProfile {
    pub const fn measured_arm(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Legacy => "legacy",
            Self::WithoutE1 => "minus-e1",
            Self::WithoutE2 => "minus-e2",
            Self::WithoutE3 => "minus-e3",
        }
    }

    fn from_arm(value: &str) -> Option<Self> {
        [
            Self::Current,
            Self::Legacy,
            Self::WithoutE1,
            Self::WithoutE2,
            Self::WithoutE3,
        ]
        .into_iter()
        .find(|profile| profile.measured_arm() == value)
    }
}

/// Private fields prevent arbitrary mixed strategy bits or model rebinding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobTrainingPolicy {
    model: &'static str,
    profile: TrainingProfile,
    resumed: bool,
}

impl JobTrainingPolicy {
    /// Resolve before any model preparation. Untagged historical resumes retain
    /// Current; a tagged resume must match this model's supported complete arm.
    pub fn resolve(model_id: &str, is_cuda: bool, req: &TrainingRequest) -> Result<Self> {
        let (model, selected) = selected_profile(model_id, is_cuda);
        let mut result = Self {
            model,
            profile: selected,
            resumed: false,
        };
        // CUDA LTX25 and ZImage own this shared snapshot contract. Other native
        // trainers (including MOSS) own different resume formats; resolving
        // their unchanged Current policy must not inspect or intercept them.
        if req.config.resume && is_cuda && matches!(model, "ltx_2_5_distilled" | "z_image_turbo") {
            if let Some((path, _)) = super::checkpoint::find_latest_resume(
                &req.output_dir,
                super::checkpoint::file_stem(&req.file_name),
            ) {
                let metadata = crate::gen_core::safetensors_file_metadata(&path)?
                    .into_iter()
                    .collect();
                let saved = resume_profile(&metadata)?;
                if saved != TrainingProfile::Current && saved != selected {
                    return Err(CandleError::Msg(
                        "resume: execution profile is incompatible with this model/backend".into(),
                    ));
                }
                result.profile = saved;
                result.resumed = true;
            }
        }
        Ok(result)
    }

    pub const fn profile(self) -> TrainingProfile {
        self.profile
    }
    pub const fn model(self) -> &'static str {
        self.model
    }
}

fn selected_profile(model: &str, cuda: bool) -> (&'static str, TrainingProfile) {
    let (model, selected) = match model {
        "wan2_2_ti2v_5b" => ("wan2_2_ti2v_5b", TrainingProfile::WithoutE1),
        "kolors" => ("kolors", TrainingProfile::WithoutE3),
        "z_image_turbo" => ("z_image_turbo", TrainingProfile::WithoutE3),
        "ltx_2_5_distilled" => ("ltx_2_5_distilled", TrainingProfile::WithoutE2),
        "sd3_5_medium" => ("sd3_5_medium", TrainingProfile::Legacy),
        _ => ("unselected", TrainingProfile::Current),
    };
    (
        model,
        if cuda {
            selected
        } else {
            TrainingProfile::Current
        },
    )
}

thread_local! {
    static JOB: Cell<Option<JobTrainingPolicy>> = const { Cell::new(None) };
}

fn scoped<T>(policy: Option<JobTrainingPolicy>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<JobTrainingPolicy>);
    impl Drop for Restore {
        fn drop(&mut self) {
            JOB.set(self.0);
        }
    }
    let _restore = Restore(JOB.replace(policy));
    // Explicit proof stages own attention selection, including PlusStreaming.
    if super::comparison::active_stage().is_some() {
        f()
    } else {
        crate::training_attention::with_training_attention_mode(
            crate::training_attention::TrainingAttentionMode::Reference,
            f,
        )
    }
}

/// Bind this immutable selection for synchronous preparation/training/checkpoint
/// replay. Nested calls and unwinding restore the previous job's selection.
pub fn with_job_policy<T>(policy: JobTrainingPolicy, f: impl FnOnce() -> T) -> T {
    eprintln!("training execution profile: version={VERSION} model={} arm={} resume={} proof_override={:?}",
        policy.model, policy.profile.measured_arm(), policy.resumed, super::comparison::active_stage());
    scoped(Some(policy), f)
}

/// Prevent a job's arithmetic/preparation choices leaking into user callbacks or
/// preview inference. Explicit proof controls remain authoritative.
pub fn without_production_profile<T>(f: impl FnOnce() -> T) -> T {
    scoped(None, f)
}

pub fn active_profile() -> TrainingProfile {
    JOB.get()
        .map_or(TrainingProfile::Current, |job| job.profile)
}

fn strategy(stage: Option<ComparisonStage>, profile: TrainingProfile) -> (bool, bool, bool, bool) {
    if let Some(stage) = stage {
        return (
            matches!(stage, ComparisonStage::Legacy | ComparisonStage::MinusE1),
            !matches!(stage, ComparisonStage::Legacy | ComparisonStage::MinusE2),
            !matches!(stage, ComparisonStage::Legacy | ComparisonStage::MinusE3),
            !matches!(
                stage,
                ComparisonStage::Legacy | ComparisonStage::MinusReferenceOwner
            ),
        );
    }
    (
        matches!(
            profile,
            TrainingProfile::Legacy | TrainingProfile::WithoutE1
        ),
        !matches!(
            profile,
            TrainingProfile::Legacy | TrainingProfile::WithoutE2
        ),
        !matches!(
            profile,
            TrainingProfile::Legacy | TrainingProfile::WithoutE3
        ),
        profile != TrainingProfile::Legacy,
    )
}

fn resolved() -> (bool, bool, bool, bool) {
    strategy(super::comparison::active_stage(), active_profile())
}
pub fn legacy_e1() -> bool {
    resolved().0
}
pub fn cache_enabled() -> bool {
    resolved().1
}
pub fn projection_enabled() -> bool {
    resolved().2
}
pub fn reference_owner_enabled() -> bool {
    resolved().3
}

pub(super) fn resume_profile(meta: &HashMap<String, String>) -> Result<TrainingProfile> {
    match meta.get(RESUME_KEY) {
        None => Ok(TrainingProfile::Current),
        Some(value) => value
            .strip_prefix(&format!("{VERSION}/"))
            .and_then(TrainingProfile::from_arm)
            .ok_or_else(|| CandleError::Msg("resume: unsupported execution profile".into())),
    }
}

/// Current retains the historical fingerprint byte-for-byte. Profile binding in
/// the existing config fingerprint makes old loaders reject non-Current bundles.
pub(super) fn bind_resume_fingerprint(base: &str, profile: TrainingProfile) -> String {
    if profile == TrainingProfile::Current {
        base.into()
    } else {
        format!(
            "{base};execution_profile={VERSION}/{}",
            profile.measured_arm()
        )
    }
}

pub(super) fn record_resume_profile(meta: &mut HashMap<String, String>) {
    let profile = active_profile();
    if profile != TrainingProfile::Current {
        meta.insert(
            RESUME_KEY.into(),
            format!("{VERSION}/{}", profile.measured_arm()),
        );
    }
}

#[cfg(test)]
mod selector_metadata_tests {
    use super::*;

    fn request(dir: &std::path::Path) -> TrainingRequest {
        TrainingRequest {
            items: vec![],
            config: Default::default(),
            output_dir: dir.into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        }
    }

    fn job(profile: TrainingProfile) -> JobTrainingPolicy {
        JobTrainingPolicy {
            model: "metadata-test",
            profile,
            resumed: false,
        }
    }

    #[test]
    fn exact_descriptors_and_non_cuda_defaults() {
        for (id, profile) in [
            ("wan2_2_ti2v_5b", TrainingProfile::WithoutE1),
            ("kolors", TrainingProfile::WithoutE3),
            ("z_image_turbo", TrainingProfile::WithoutE3),
            ("ltx_2_5_distilled", TrainingProfile::WithoutE2),
            ("sd3_5_medium", TrainingProfile::Legacy),
        ] {
            assert_eq!(selected_profile(id, true).1, profile);
            assert_eq!(selected_profile(id, false).1, TrainingProfile::Current);
        }
        for id in [
            "krea",
            "krea_raw",
            "wan2_2_t2v_14b",
            "sd3_5_large",
            "ltx_2_3",
            "unknown",
        ] {
            assert_eq!(selected_profile(id, true).1, TrainingProfile::Current);
        }
    }

    #[test]
    fn complete_arms_and_all_proof_stages_override() {
        for (profile, bits) in [
            (TrainingProfile::Current, (false, true, true, true)),
            (TrainingProfile::Legacy, (true, false, false, false)),
            (TrainingProfile::WithoutE1, (true, true, true, true)),
            (TrainingProfile::WithoutE2, (false, false, true, true)),
            (TrainingProfile::WithoutE3, (false, true, false, true)),
        ] {
            assert_eq!(strategy(None, profile), bits);
            for stage in ComparisonStage::ALL {
                assert_eq!(
                    strategy(Some(stage), profile),
                    strategy(Some(stage), TrainingProfile::Current)
                );
            }
        }
        assert_eq!(
            strategy(
                Some(ComparisonStage::MinusReferenceOwner),
                TrainingProfile::Legacy
            ),
            (false, true, true, false)
        );
        assert_eq!(
            strategy(
                Some(ComparisonStage::PlusStreaming),
                TrainingProfile::Legacy
            ),
            (false, true, true, true)
        );
    }

    #[test]
    fn scopes_suspend_restore_unwind_and_isolate_threads() {
        assert_eq!(active_profile(), TrainingProfile::Current);
        with_job_policy(job(TrainingProfile::Legacy), || {
            assert!(legacy_e1());
            without_production_profile(|| {
                assert_eq!(active_profile(), TrainingProfile::Current);
                assert!(!legacy_e1());
                with_job_policy(
                    job(TrainingProfile::WithoutE2),
                    || assert!(!cache_enabled()),
                );
                assert_eq!(active_profile(), TrainingProfile::Current);
            });
            assert!(legacy_e1());
            let handle = std::thread::spawn(|| {
                assert_eq!(active_profile(), TrainingProfile::Current);
                with_job_policy(job(TrainingProfile::WithoutE3), || {
                    assert!(!projection_enabled())
                });
                assert!(projection_enabled());
            });
            handle.join().unwrap();
            assert!(legacy_e1());
            let panic = std::panic::catch_unwind(|| {
                with_job_policy(job(TrainingProfile::WithoutE1), || {
                    panic!("metadata unwind")
                })
            });
            assert!(panic.is_err());
            assert_eq!(active_profile(), TrainingProfile::Legacy);
            let panic = std::panic::catch_unwind(|| {
                without_production_profile(|| panic!("callback unwind"))
            });
            assert!(panic.is_err());
            assert_eq!(active_profile(), TrainingProfile::Legacy);
        });
        assert_eq!(active_profile(), TrainingProfile::Current);
        fn send_sync<T: Send + Sync>() {}
        send_sync::<JobTrainingPolicy>();
    }

    // Valid safetensors header only; no tensor payload, device or model creation.
    fn write_header(dir: &std::path::Path, metadata: serde_json::Value) {
        let header = serde_json::to_vec(&serde_json::json!({"__metadata__":metadata})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        std::fs::write(dir.join("adapter-step000004.resume.safetensors"), bytes).unwrap();
    }

    #[test]
    fn resume_header_selection_preserves_historical_and_rejects_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let mut req = request(dir.path());
        assert_eq!(
            JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::WithoutE2
        );
        req.config.resume = true;
        write_header(dir.path(), serde_json::json!({}));
        assert_eq!(
            JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        write_header(
            dir.path(),
            serde_json::json!({RESUME_KEY: "measured-reference-v1/minus-e2"}),
        );
        assert_eq!(
            JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::WithoutE2
        );
        assert_eq!(
            JobTrainingPolicy::resolve("ltx_2_5_distilled", false, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        assert_eq!(
            JobTrainingPolicy::resolve("sd3_5_medium", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Legacy
        );
        write_header(
            dir.path(),
            serde_json::json!({RESUME_KEY: "measured-reference-v1/plus-streaming"}),
        );
        assert!(JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req).is_err());
        write_header(
            dir.path(),
            serde_json::json!({RESUME_KEY: "future-v2/minus-e2"}),
        );
        assert!(JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req).is_err());
    }

    #[test]
    fn native_and_non_cuda_resume_owners_do_not_peek_shared_headers() {
        let dir = tempfile::tempdir().unwrap();
        let mut req = request(dir.path());
        req.config.resume = true;
        // A matching shared snapshot name with an invalid header discriminates
        // a real read from the required no-interception path. No tensors exist.
        std::fs::write(
            dir.path().join("adapter-step000004.resume.safetensors"),
            b"invalid shared header",
        )
        .unwrap();
        for model in ["moss_image_base", "moss_image_sft", "unknown", "krea"] {
            for cuda in [false, true] {
                assert_eq!(
                    JobTrainingPolicy::resolve(model, cuda, &req)
                        .unwrap()
                        .profile(),
                    TrainingProfile::Current
                );
            }
        }
        for (model, profile) in [
            ("wan2_2_ti2v_5b", TrainingProfile::WithoutE1),
            ("kolors", TrainingProfile::WithoutE3),
            ("sd3_5_medium", TrainingProfile::Legacy),
        ] {
            assert_eq!(
                JobTrainingPolicy::resolve(model, true, &req)
                    .unwrap()
                    .profile(),
                profile
            );
            assert_eq!(
                JobTrainingPolicy::resolve(model, false, &req)
                    .unwrap()
                    .profile(),
                TrainingProfile::Current
            );
        }
        assert_eq!(
            JobTrainingPolicy::resolve("ltx_2_5_distilled", false, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        assert!(JobTrainingPolicy::resolve("ltx_2_5_distilled", true, &req).is_err());
        assert!(JobTrainingPolicy::resolve("z_image_turbo", true, &req).is_err());
        assert_eq!(
            JobTrainingPolicy::resolve("z_image_turbo", false, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        req.output_dir = dir.path().join("nonexistent-native-resume-path");
        for model in ["moss_image_base", "moss_image_sft", "unknown"] {
            assert_eq!(
                JobTrainingPolicy::resolve(model, true, &req)
                    .unwrap()
                    .profile(),
                TrainingProfile::Current
            );
        }
    }

    #[test]
    fn z_shared_resume_preserves_historical_current_and_tagged_measured_arm() {
        let dir = tempfile::tempdir().unwrap();
        let mut req = request(dir.path());
        req.config.resume = true;
        write_header(dir.path(), serde_json::json!({}));
        assert_eq!(
            JobTrainingPolicy::resolve("z_image_turbo", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        write_header(
            dir.path(),
            serde_json::json!({RESUME_KEY: "measured-reference-v1/minus-e3"}),
        );
        assert_eq!(
            JobTrainingPolicy::resolve("z_image_turbo", true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::WithoutE3
        );
        assert_eq!(
            JobTrainingPolicy::resolve("z_image_turbo", false, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        write_header(
            dir.path(),
            serde_json::json!({RESUME_KEY: "measured-reference-v1/minus-e2"}),
        );
        assert!(JobTrainingPolicy::resolve("z_image_turbo", true, &req).is_err());
    }
}
