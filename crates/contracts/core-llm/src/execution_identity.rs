//! Shared tensor-neutral provider-resolved execution identity for media and text generation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// The schema of a provider-resolved execution identity. A future incompatible interpretation
/// needs a new version; a consumer must not silently read it as version 1.
pub const EXECUTION_IDENTITY_VERSION: u32 = 1;

/// A fact that either has an actual resolved value or does not apply to this provider. A missing
/// value is never represented by JSON `null`; it makes the whole report [`ExecutionIdentity::Unavailable`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExecutionFact {
    Resolved { value: Value },
    NotApplicable { reason: String },
}

impl ExecutionFact {
    pub fn resolved(value: impl Into<Value>) -> Self {
        Self::Resolved {
            value: value.into(),
        }
    }

    pub fn not_applicable(reason: impl Into<String>) -> Self {
        Self::NotApplicable {
            reason: reason.into(),
        }
    }

    /// Store a resolved floating-point control by its exact IEEE-754 bits. Decimal JSON may
    /// round or normalize signed zero before an exact-replay comparison.
    pub fn f32_bits(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::Msg("execution f32 fact must be finite".into()));
        }
        Ok(Self::resolved(format!("f32:{:08x}", value.to_bits())))
    }

    pub fn f64_bits(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::Msg("execution f64 fact must be finite".into()));
        }
        Ok(Self::resolved(format!("f64:{:016x}", value.to_bits())))
    }

    fn validate(&self, name: &str) -> Result<()> {
        match self {
            Self::Resolved { value: Value::Null } => Err(Error::Msg(format!(
                "execution fact {name} is null; mark it not_applicable or report it unavailable"
            ))),
            Self::NotApplicable { reason } if reason.trim().is_empty() => Err(Error::Msg(format!(
                "execution fact {name} needs a not_applicable reason"
            ))),
            _ => Ok(()),
        }
    }
}

/// One actual random draw associated with an output, in the order the provider used it. A draw
/// names either its effective seed or a digest of explicit noise; models with several streams
/// (video/audio, stages, tiled refinement) publish a separate row for each purpose.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionDraw {
    pub output_index: usize,
    pub purpose: String,
    pub seed: Option<u64>,
    pub noise_sha256: Option<String>,
}

/// The provider's actual RNG/draw scheme. `NoDraws` is for a deterministic transform, not a
/// substitute for an unobserved seed. The latter is [`ExecutionIdentity::Unavailable`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionRandomness {
    Draws {
        scheme: String,
        draws: Vec<ExecutionDraw>,
    },
    NoDraws {
        reason: String,
    },
}

/// Whether the provider can reproduce the same result from this identity and the frozen input
/// binding. The scope names any runtime/device constraints that exact replay must preserve.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionReplaySupport {
    Deterministic { scope: String },
    NonDeterministic { reason: String },
}

impl ExecutionReplaySupport {
    fn validate(&self) -> Result<()> {
        let description = match self {
            Self::Deterministic { scope } => scope,
            Self::NonDeterministic { reason } => reason,
        };
        if description.trim().is_empty() {
            return Err(Error::Msg(
                "execution replay support needs a scope or reason".into(),
            ));
        }
        Ok(())
    }

    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::Deterministic { .. })
    }
}

/// A platform-independent digest of the exact floating-point schedule consumed by a provider.
/// Hash the little-endian IEEE-754 bits, not formatted decimal text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericScheduleDigest {
    pub element: String,
    pub count: usize,
    pub sha256: String,
}

impl NumericScheduleDigest {
    pub fn f32_values(values: &[f32]) -> Result<Self> {
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg(
                "execution schedule must contain finite f32 values".into(),
            ));
        }
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        Ok(Self {
            element: "f32_le_bits".into(),
            count: values.len(),
            sha256: format!("{:x}", hash.finalize()),
        })
    }

    pub fn f64_values(values: &[f64]) -> Result<Self> {
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg(
                "execution schedule must contain finite f64 values".into(),
            ));
        }
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        Ok(Self {
            element: "f64_le_bits".into(),
            count: values.len(),
            sha256: format!("{:x}", hash.finalize()),
        })
    }
}

/// Complete provider-resolved choices for one attempt. The application supplies source,
/// component and adapter byte identities separately from its frozen admission binding; this
/// engine report deliberately carries no paths, credentials or client-authored source claims.
/// `runtime_id` identifies the provider's numerical implementation, while the consuming app also
/// records the exact inference revision it pinned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedExecutionIdentity {
    pub version: u32,
    pub provider_id: String,
    pub backend: String,
    pub runtime_id: String,
    /// Effective sampler/solver or autoregressive token-sampling settings.
    pub sampling: ExecutionFact,
    /// Exact schedule or a [`NumericScheduleDigest`], including all stage schedules where needed.
    pub schedule: ExecutionFact,
    /// Effective estimator, or a reason that this provider has none.
    pub estimator: ExecutionFact,
    /// Actual seed/draw mapping, not a reconstruction from the submitted request.
    pub randomness: ExecutionRandomness,
    /// Provider-attested deterministic replay support and its scope or refusal reason.
    pub deterministic_replay: ExecutionReplaySupport,
    /// Resolved output geometry or an explicit non-applicability reason.
    pub geometry: ExecutionFact,
    /// Every output-affecting memory, tiling, chunking and selected-rung decision.
    pub quality_policy: ExecutionFact,
    /// Effective output format, channels, duration, frame policy and other output choices.
    pub output_policy: ExecutionFact,
    /// Provider-specific resolved choices not represented by the common fields.
    pub provider_choices: BTreeMap<String, ExecutionFact>,
    /// Ordered output or artifact units to which `randomness` draws refer.
    pub output_units: usize,
}

impl ResolvedExecutionIdentity {
    /// Validate a provider report against the descriptor and the successfully completed output.
    /// Providers call this through their completed-output wrapper before publication.
    pub fn validate(&self, provider_id: &str, backend: &str, output_units: usize) -> Result<()> {
        if self.version != EXECUTION_IDENTITY_VERSION {
            return Err(Error::Msg(format!(
                "unsupported execution identity version {}",
                self.version
            )));
        }
        if self.provider_id != provider_id || self.backend != backend {
            return Err(Error::Msg(
                "execution identity provider/backend differs from the completed generator".into(),
            ));
        }
        if self.runtime_id.trim().is_empty() {
            return Err(Error::Msg("execution runtime identity is empty".into()));
        }
        if self.output_units != output_units {
            return Err(Error::Msg(format!(
                "execution identity declares {} outputs, completed generation has {output_units}",
                self.output_units
            )));
        }
        self.deterministic_replay.validate()?;
        for (name, fact) in [
            ("sampling", &self.sampling),
            ("schedule", &self.schedule),
            ("estimator", &self.estimator),
            ("geometry", &self.geometry),
            ("quality_policy", &self.quality_policy),
            ("output_policy", &self.output_policy),
        ] {
            fact.validate(name)?;
        }
        for (name, fact) in [
            ("quality_policy", &self.quality_policy),
            ("output_policy", &self.output_policy),
        ] {
            if matches!(fact, ExecutionFact::NotApplicable { .. }) {
                return Err(Error::Msg(format!(
                    "execution fact {name} must name the effective policy"
                )));
            }
        }
        for (name, fact) in &self.provider_choices {
            if name.trim().is_empty() {
                return Err(Error::Msg(
                    "execution provider choice has an empty key".into(),
                ));
            }
            fact.validate(name)?;
        }
        match &self.randomness {
            ExecutionRandomness::NoDraws { reason } if reason.trim().is_empty() => {
                Err(Error::Msg("no_draws needs a reason".into()))
            }
            ExecutionRandomness::NoDraws { .. } => Ok(()),
            ExecutionRandomness::Draws { scheme, draws } => {
                if scheme.trim().is_empty() || draws.is_empty() {
                    return Err(Error::Msg(
                        "execution RNG scheme and draws must be nonempty".into(),
                    ));
                }
                let mut covered = BTreeSet::new();
                for draw in draws {
                    if draw.output_index >= output_units
                        || draw.purpose.trim().is_empty()
                        || (draw.seed.is_none() && draw.noise_sha256.is_none())
                        || draw
                            .noise_sha256
                            .as_deref()
                            .is_some_and(|hex| !is_sha256(hex))
                    {
                        return Err(Error::Msg("invalid execution draw mapping".into()));
                    }
                    covered.insert(draw.output_index);
                }
                if covered.len() != output_units {
                    return Err(Error::Msg(
                        "execution draws do not cover every completed output".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// An explicit reason exact replay cannot rely on this attempt. Default trait implementations
/// use `provider_report_unavailable` for source compatibility, but repository providers must
/// publish resolved facts or a concrete, source-backed non-replayable reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionUnavailable {
    pub code: String,
    pub detail: String,
    pub missing_facts: Vec<String>,
}

impl ExecutionUnavailable {
    pub fn provider_default() -> Self {
        Self {
            code: "provider_report_unavailable".into(),
            detail: "provider has not published a completed resolved execution report".into(),
            missing_facts: vec!["provider_resolved_execution".into()],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExecutionIdentity {
    Resolved {
        identity: Box<ResolvedExecutionIdentity>,
    },
    Unavailable {
        reason: ExecutionUnavailable,
    },
}

impl ExecutionIdentity {
    pub fn unavailable(reason: ExecutionUnavailable) -> Self {
        Self::Unavailable { reason }
    }

    pub fn provider_default() -> Self {
        Self::unavailable(ExecutionUnavailable::provider_default())
    }

    pub fn resolved(&self) -> Option<&ResolvedExecutionIdentity> {
        match self {
            Self::Resolved { identity } => Some(identity),
            Self::Unavailable { .. } => None,
        }
    }

    pub fn validate(&self, provider_id: &str, backend: &str, output_units: usize) -> Result<()> {
        match self {
            Self::Resolved { identity } => identity.validate(provider_id, backend, output_units),
            Self::Unavailable { reason }
                if reason.code.trim().is_empty()
                    || reason.detail.trim().is_empty()
                    || reason.missing_facts.is_empty()
                    || reason
                        .missing_facts
                        .iter()
                        .any(|fact| fact.trim().is_empty()) =>
            {
                Err(Error::Msg(
                    "execution unavailability needs a typed reason".into(),
                ))
            }
            Self::Unavailable { .. } => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompletedStarVector, CompletedTextLlm, FinishReason, Message, StarVectorFinishReason,
        StarVectorOutput, StreamEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor,
        TextLlmOutput, TextLlmRequest,
    };

    fn descriptor() -> TextLlmDescriptor {
        TextLlmDescriptor {
            id: "text-test".into(),
            family: "test".into(),
            backend: "mlx".into(),
            capabilities: TextLlmCapabilities::default(),
        }
    }

    fn resolved() -> ExecutionIdentity {
        ExecutionIdentity::Resolved {
            identity: Box::new(ResolvedExecutionIdentity {
                version: EXECUTION_IDENTITY_VERSION,
                provider_id: "text-test".into(),
                backend: "mlx".into(),
                runtime_id: "test-provider@1+numerics-a".into(),
                sampling: ExecutionFact::resolved(serde_json::json!({"temperature":0.7})),
                schedule: ExecutionFact::not_applicable("autoregressive tokens"),
                estimator: ExecutionFact::not_applicable("no diffusion estimator"),
                randomness: ExecutionRandomness::Draws {
                    scheme: "token-stream-v1".into(),
                    draws: vec![ExecutionDraw {
                        output_index: 0,
                        purpose: "tokens".into(),
                        seed: Some(7),
                        noise_sha256: None,
                    }],
                },
                deterministic_replay: ExecutionReplaySupport::Deterministic {
                    scope: "same provider revision and device".into(),
                },
                geometry: ExecutionFact::not_applicable("text output"),
                quality_policy: ExecutionFact::resolved(serde_json::json!({"kv_cache":"dense"})),
                output_policy: ExecutionFact::resolved(serde_json::json!({"max_new_tokens":8})),
                provider_choices: BTreeMap::new(),
                output_units: 1,
            }),
        }
    }

    #[test]
    fn shared_identity_round_trips_and_requires_effective_policy() {
        let identity = resolved();
        identity.validate("text-test", "mlx", 1).unwrap();
        assert_eq!(
            serde_json::from_str::<ExecutionIdentity>(&serde_json::to_string(&identity).unwrap())
                .unwrap(),
            identity
        );
        let ExecutionIdentity::Resolved { mut identity } = identity else {
            unreachable!()
        };
        identity.quality_policy = ExecutionFact::not_applicable("unobserved");
        assert!(identity.validate("text-test", "mlx", 1).is_err());
        identity.quality_policy = ExecutionFact::resolved(serde_json::json!({"kv_cache":"dense"}));
        identity.deterministic_replay = ExecutionReplaySupport::NonDeterministic {
            reason: "this backend does not guarantee deterministic reductions".into(),
        };
        assert!(identity.validate("text-test", "mlx", 1).is_ok());
        assert!(!identity.deterministic_replay.is_deterministic());
        identity.deterministic_replay = ExecutionReplaySupport::Deterministic {
            scope: String::new(),
        };
        assert!(identity.validate("text-test", "mlx", 1).is_err());
    }

    #[test]
    fn scalar_numeric_facts_preserve_signed_zero_and_reject_nan() {
        assert_ne!(
            ExecutionFact::f32_bits(0.0).unwrap(),
            ExecutionFact::f32_bits(-0.0).unwrap()
        );
        assert_ne!(
            ExecutionFact::f64_bits(0.0).unwrap(),
            ExecutionFact::f64_bits(-0.0).unwrap()
        );
        assert!(ExecutionFact::f32_bits(f32::NAN).is_err());
    }

    #[test]
    fn text_and_svg_partial_outputs_cannot_claim_exact_replay() {
        let partial = TextLlmOutput {
            text: "partial".into(),
            finish_reason: Some(FinishReason::Cancelled),
            ..Default::default()
        };
        assert!(CompletedTextLlm::new(partial, resolved(), &descriptor()).is_err());
        let svg = StarVectorOutput {
            svg: None,
            generated_tokens: 2,
            generated_bytes: 4,
            finish_reason: StarVectorFinishReason::TokenLimit,
            kv_cache: None,
        };
        assert!(CompletedStarVector::new(svg, resolved(), &descriptor()).is_err());
    }

    #[test]
    fn legacy_text_provider_reports_missing_facts_without_claiming_request_values() {
        struct Legacy(TextLlmDescriptor);
        impl TextLlm for Legacy {
            fn descriptor(&self) -> &TextLlmDescriptor {
                &self.0
            }
            fn validate(&self, _req: &TextLlmRequest) -> crate::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _req: &TextLlmRequest,
                _on_event: &mut dyn FnMut(StreamEvent),
            ) -> crate::Result<TextLlmOutput> {
                Ok(TextLlmOutput {
                    text: "done".into(),
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                })
            }
        }
        let generator = Legacy(descriptor());
        let request = TextLlmRequest::new(vec![Message::user("hello")], 8);
        assert!(matches!(
            generator.plan_execution(&request).unwrap(),
            ExecutionIdentity::Unavailable { .. }
        ));
        let completed = generator
            .generate_with_execution_report(&request, &mut |_| {})
            .unwrap();
        assert_eq!(completed.output.text, "done");
        assert!(matches!(
            completed.execution,
            ExecutionIdentity::Unavailable { .. }
        ));
        let batch = generator
            .generate_batch_with_execution_report(&[request.clone(), request], &mut |_, _| {});
        assert_eq!(batch.len(), 2);
        assert!(batch.into_iter().all(|result| result.is_ok()));
    }
}
