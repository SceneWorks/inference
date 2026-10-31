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
}

impl ComparisonCadence {
    pub fn updates(self) -> u32 {
        match self {
            Self::Ordinary => 4,
            Self::WanDual => 6,
            Self::IrisRestoration => 8,
        }
    }
    pub fn warmups(self) -> usize {
        match self {
            Self::Ordinary => 1,
            Self::WanDual | Self::IrisRestoration => 2,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::WanDual => "wan-dual",
            Self::IrisRestoration => "iris-restoration",
        }
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
