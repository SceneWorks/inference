//! Source-owned controls for manual, fresh-process training comparisons.
//! No environment variable changes production math: controls require `with_stage`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonStage {
    Legacy,
    Current,
    MinusE1,
    MinusE2,
    MinusE3,
    MinusReferenceOwner,
    PlusStreaming,
}

impl ComparisonStage {
    pub const ALL: [Self; 7] = [
        Self::Legacy,
        Self::Current,
        Self::MinusE1,
        Self::MinusE2,
        Self::MinusE3,
        Self::MinusReferenceOwner,
        Self::PlusStreaming,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Current => "current",
            Self::MinusE1 => "minus-e1",
            Self::MinusE2 => "minus-e2",
            Self::MinusE3 => "minus-e3",
            Self::MinusReferenceOwner => "minus-reference-owner",
            Self::PlusStreaming => "plus-streaming",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|stage| stage.id() == value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonCadence {
    Ordinary,
    WanDual,
    IrisRestoration,
    /// Failed-only MOSS recovery: one counted warmup and two timed updates.
    MossRecovery,
    OrdinaryShort,
    DepthRecovery,
    RestRecovery,
}

impl ComparisonCadence {
    pub fn updates(self) -> u32 {
        match self {
            Self::Ordinary => 4,
            Self::WanDual => 6,
            Self::IrisRestoration => 8,
            Self::MossRecovery => 3,
            Self::OrdinaryShort | Self::DepthRecovery => 2,
            Self::RestRecovery => 6,
        }
    }
    pub fn warmups(self) -> usize {
        match self {
            Self::Ordinary | Self::MossRecovery | Self::OrdinaryShort | Self::DepthRecovery => 1,
            Self::WanDual | Self::IrisRestoration | Self::RestRecovery => 2,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::WanDual => "wan-dual",
            Self::IrisRestoration => "iris-restoration",
            Self::MossRecovery => "moss_recovery",
            Self::OrdinaryShort => "ordinary_short",
            Self::DepthRecovery => "depth_recovery",
            Self::RestRecovery => "rest_recovery",
        }
    }

    /// The shorter recovery cannot reduce another model's registered cadence.
    pub fn allows_model(self, model_id: &str) -> bool {
        match self {
            Self::MossRecovery => model_id == "moss",
            Self::OrdinaryShort => matches!(model_id, "anima_base" | "flux3"),
            Self::DepthRecovery => model_id == "iris-depth",
            Self::RestRecovery => model_id == "iris-restoration",
            _ => true,
        }
    }

    /// Recovery owner sequences are source-owned physical optimizer bodies,
    /// including counted warmups. Ordinary historical cadence behavior is intact.
    pub fn expected_recovery_owners(self, model_id: &str) -> Option<&'static [&'static str]> {
        match (self, model_id) {
            (Self::MossRecovery, "moss") => Some(&["moss-image"; 3]),
            (Self::OrdinaryShort, "anima_base") => Some(&["anima trainer"; 2]),
            (Self::OrdinaryShort, "flux3") => Some(&["flux3"; 2]),
            (Self::DepthRecovery, "iris-depth") => Some(&["iris-depth"; 2]),
            (Self::RestRecovery, "iris-restoration") => Some(&[
                "iris-restoration-generator",
                "iris-restoration-critic",
                "iris-restoration-generator",
                "iris-restoration-critic",
                "iris-restoration-generator",
                "iris-restoration-critic",
            ]),
            _ => None,
        }
    }

    pub fn recovery_owners_valid(self, model_id: &str, observed: &[&str]) -> bool {
        self.allows_model(model_id)
            && self
                .expected_recovery_owners(model_id)
                .is_none_or(|expected| expected == observed)
    }
}

#[cfg(feature = "model-inference-proof")]
thread_local! {
    static STAGE: std::cell::Cell<Option<ComparisonStage>> = const { std::cell::Cell::new(None) };
}

/// Read only in a registered manual test, before constructing its fresh native Run.
pub fn requested_stage() -> Option<ComparisonStage> {
    #[cfg(feature = "model-inference-proof")]
    {
        std::env::var("MODEL_PROOF_COMPARISON_STAGE")
            .ok()
            .map(|value| {
                ComparisonStage::parse(&value).expect("unknown source-owned comparison stage")
            })
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        None
    }
}

pub fn active_stage() -> Option<ComparisonStage> {
    #[cfg(feature = "model-inference-proof")]
    {
        STAGE.get()
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        None
    }
}

pub fn legacy_e1() -> bool {
    matches!(
        active_stage(),
        Some(ComparisonStage::Legacy | ComparisonStage::MinusE1)
    )
}
pub fn cache_enabled() -> bool {
    !matches!(
        active_stage(),
        Some(ComparisonStage::Legacy | ComparisonStage::MinusE2)
    )
}
pub fn projection_enabled() -> bool {
    !matches!(
        active_stage(),
        Some(ComparisonStage::Legacy | ComparisonStage::MinusE3)
    )
}
pub fn reference_owner_enabled() -> bool {
    !matches!(
        active_stage(),
        Some(ComparisonStage::Legacy | ComparisonStage::MinusReferenceOwner)
    )
}

#[cfg(all(test, feature = "model-inference-proof"))]
pub(crate) fn with_metadata_stage_for_test<T>(stage: ComparisonStage, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<ComparisonStage>);
    impl Drop for Reset {
        fn drop(&mut self) {
            STAGE.set(self.0);
        }
    }
    let _reset = Reset(STAGE.replace(Some(stage)));
    f()
}

/// Physical index for the new comparison recipe only. CUDA ordinal remains zero
/// under its exact one-UUID visibility mapping.
pub fn required_physical_index() -> Option<usize> {
    active_stage().map(
        |_| match std::env::var("MODEL_PROOF_COMPARISON_BINDING").as_deref() {
            Ok("gpu0") => 0,
            Ok("gpu1") => 1,
            _ => panic!("source-owned comparison physical binding required"),
        },
    )
}

/// Append actual canonical configuration/factory facts before any update.
/// Do not include stage-specific output directories in this cross-stage identity.
pub fn record_initialization(value: &serde_json::Value) {
    #[cfg(feature = "model-inference-proof")]
    if active_stage().is_some() {
        super::profile::comparison_initialization(value);
    }
    let _ = value;
}

/// Authenticate private optimizer/EMA/buffer state without publishing tensor arrays.
/// Call after its fresh constructor and before training, using stable tensor names.
pub fn record_initial_tensors<'a>(
    owner: &str,
    tensors: impl IntoIterator<Item = (&'a str, &'a candle_core::Tensor)>,
) -> candle_core::Result<()> {
    #[cfg(feature = "model-inference-proof")]
    if active_stage().is_some() {
        return super::profile::comparison_named_initial_tensors(owner, tensors);
    }
    let _ = (owner, tensors);
    Ok(())
}

/// One child process owns one stage. The closure MUST construct all native mutable
/// training state afresh; it must not reuse a Run, optimizer, EMA, or packed bases.
/// Model quality sampling is deliberately outside this timing-only supplement.
#[cfg(feature = "model-inference-proof")]
pub fn with_stage<T>(
    output: &std::path::Path,
    model_id: &str,
    cadence: ComparisonCadence,
    f: impl FnOnce() -> T,
) -> T {
    let stage = requested_stage().expect("manual comparison stage required");
    assert!(
        !model_id.is_empty() && !model_id.to_ascii_lowercase().contains("krea"),
        "Krea budget is excluded"
    );
    assert!(
        cadence.allows_model(model_id),
        "source-owned recovery model/cadence"
    );
    assert!(
        active_stage().is_none() && !super::profile::active() && !super::profile::cohort_active(),
        "nested comparison scope"
    );
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STAGE.set(None);
        }
    }
    STAGE.set(Some(stage));
    let _reset = Reset;
    let physical = required_physical_index().expect("comparison physical index");
    let uuid = if physical == 0 {
        "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
    } else {
        "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
    };
    assert_eq!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref(),
        Ok(uuid),
        "comparison requires exact selected physical UUID visibility"
    );
    assert_eq!(
        std::env::var("SCENEWORKS_GPU_ID").as_deref(),
        Ok(if physical == 0 { "0" } else { "1" }),
        "comparison sampler physical mapping"
    );
    let mode = if stage == ComparisonStage::PlusStreaming {
        crate::training_attention::TrainingAttentionMode::StreamingCandidate
    } else {
        crate::training_attention::TrainingAttentionMode::Reference
    };
    eprintln!(
        "TRAINING_COMPARISON_STAGE model={model_id} stage={} cadence={} updates={}",
        stage.id(),
        cadence.id(),
        cadence.updates()
    );
    crate::training_attention::with_training_attention_mode(mode, || {
        super::profile::with_comparison_observation(output, model_id, stage, cadence, f)
    })
}

#[cfg(test)]
mod cadence_metadata_tests {
    use super::ComparisonCadence;

    #[test]
    fn recovery_is_exact_three_one_two_without_changing_existing_cadences() {
        for (cadence, id, updates, warmups, timed) in [
            (ComparisonCadence::Ordinary, "ordinary", 4, 1, 3),
            (ComparisonCadence::WanDual, "wan-dual", 6, 2, 4),
            (
                ComparisonCadence::IrisRestoration,
                "iris-restoration",
                8,
                2,
                6,
            ),
            (ComparisonCadence::MossRecovery, "moss_recovery", 3, 1, 2),
            (ComparisonCadence::OrdinaryShort, "ordinary_short", 2, 1, 1),
            (ComparisonCadence::DepthRecovery, "depth_recovery", 2, 1, 1),
            (ComparisonCadence::RestRecovery, "rest_recovery", 6, 2, 4),
        ] {
            assert_eq!(cadence.id(), id);
            assert_eq!(cadence.updates(), updates);
            assert_eq!(cadence.warmups(), warmups);
            assert_eq!(cadence.updates() as usize - cadence.warmups(), timed);
        }
        let recovery = ComparisonCadence::MossRecovery;
        assert_eq!(6 * recovery.updates(), 18);
        // The original failed arm stays charged four; no refund or transfer.
        assert_eq!(4 + 6 * recovery.updates(), 22);
        assert_eq!(24 - (4 + 6 * recovery.updates()), 2);
    }

    #[test]
    fn recovery_owner_order_counts_and_family_caps_are_exact() {
        for (cadence, model) in [
            (ComparisonCadence::MossRecovery, "moss"),
            (ComparisonCadence::OrdinaryShort, "anima_base"),
            (ComparisonCadence::OrdinaryShort, "flux3"),
            (ComparisonCadence::DepthRecovery, "iris-depth"),
            (ComparisonCadence::RestRecovery, "iris-restoration"),
        ] {
            let expected = cadence.expected_recovery_owners(model).unwrap();
            assert_eq!(expected.len(), cadence.updates() as usize);
            assert!(cadence.recovery_owners_valid(model, expected));
            assert!(!cadence.recovery_owners_valid(model, &expected[1..]));
            assert!(!cadence.recovery_owners_valid("unknown", expected));
            let mut extra = expected.to_vec();
            extra.push(expected[0]);
            assert!(!cadence.recovery_owners_valid(model, &extra));
            let wrong = vec!["borrowed-owner"; expected.len()];
            assert!(!cadence.recovery_owners_valid(model, &wrong));
        }
        let cadence = ComparisonCadence::RestRecovery;
        let mut swapped = cadence
            .expected_recovery_owners("iris-restoration")
            .unwrap()
            .to_vec();
        swapped.swap(0, 1);
        assert!(!cadence.recovery_owners_valid("iris-restoration", &swapped));
        let duplicate = ["iris-restoration-generator"; 6];
        assert!(!cadence.recovery_owners_valid("iris-restoration", &duplicate));
        assert_eq!(6 * ComparisonCadence::OrdinaryShort.updates(), 12);
        assert_eq!(2 * ComparisonCadence::DepthRecovery.updates(), 4);
        assert_eq!(5 * ComparisonCadence::RestRecovery.updates(), 30);
        assert_eq!(316 + 12 + 12 + 4 + 30 + 18, 392);
    }

    #[test]
    fn shortened_cadence_is_moss_only_and_old_moss_cadence_remains_valid() {
        assert!(ComparisonCadence::MossRecovery.allows_model("moss"));
        assert!(ComparisonCadence::Ordinary.allows_model("moss"));
        for model in [
            "mage-adapter",
            "wan2_2_ti2v_5b",
            "iris-restoration",
            "krea",
            "",
            "MOSS",
        ] {
            assert!(!ComparisonCadence::MossRecovery.allows_model(model));
            for cadence in [
                ComparisonCadence::Ordinary,
                ComparisonCadence::WanDual,
                ComparisonCadence::IrisRestoration,
            ] {
                assert!(cadence.allows_model(model));
            }
        }
    }
}

#[cfg(all(test, feature = "model-inference-proof"))]
mod policy_selector_metadata_tests {
    use super::*;

    #[test]
    fn explicit_proof_scope_wins_over_job_and_callback_suspension() {
        use crate::gen_core::train::TrainingRequest;
        use crate::train::policy;
        let req = TrainingRequest {
            items: vec![],
            config: Default::default(),
            output_dir: "unused".into(),
            file_name: "out.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        };
        let job = policy::JobTrainingPolicy::resolve("sd3_5_medium", true, &req).unwrap();
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                STAGE.set(None);
            }
        }
        let _reset = Reset;
        for stage in ComparisonStage::ALL {
            STAGE.set(Some(stage));
            policy::with_job_policy(job, || {
                assert_eq!(policy::legacy_e1(), legacy_e1());
                assert_eq!(policy::cache_enabled(), cache_enabled());
                assert_eq!(policy::projection_enabled(), projection_enabled());
                assert_eq!(policy::reference_owner_enabled(), reference_owner_enabled());
                let identity =
                    policy::bind_state_identity("private-proof-state", policy::active_profile());
                assert!(identity.ends_with(&format!("comparison-proof-v1/{}", stage.id())));
                assert!(
                    policy::profile_from_state_identity(&identity).is_err(),
                    "experimental proof checkpoint cannot become a production resume"
                );
                policy::without_production_profile(|| {
                    assert_eq!(policy::legacy_e1(), legacy_e1());
                    assert_eq!(policy::cache_enabled(), cache_enabled());
                    assert_eq!(policy::projection_enabled(), projection_enabled());
                    assert_eq!(policy::reference_owner_enabled(), reference_owner_enabled());
                });
            });
        }
    }
}
