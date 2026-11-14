//! Attempt-local completed output wrappers for the shared execution identity.

use super::{GenerationOutput, GenerationReport, ModelDescriptor};
use crate::{Error, Result};
pub use core_llm::{
    ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness, ExecutionReplaySupport,
    ExecutionUnavailable, NumericScheduleDigest, ResolvedExecutionIdentity,
    EXECUTION_IDENTITY_VERSION,
};

#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use std::collections::BTreeMap;

/// One successful ordinary or streaming generation and its attempt-local execution facts.
/// An error/cancellation returns no completed report. No mutable "last report" slot is involved.
#[derive(Debug)]
pub struct CompletedGeneration {
    pub generation: GenerationReport,
    pub execution: ExecutionIdentity,
}

impl CompletedGeneration {
    pub fn new(
        generation: GenerationReport,
        execution: ExecutionIdentity,
        descriptor: &ModelDescriptor,
    ) -> Result<Self> {
        let units = generation_output_units(&generation)?;
        execution.validate(descriptor.id, descriptor.backend, units)?;
        Ok(Self {
            generation,
            execution,
        })
    }
}

/// One successful native MOSS edit and its attempt-local execution facts.
#[derive(Debug)]
pub struct CompletedMossImageEdit {
    pub output: crate::moss_image::MossImageAdvancedOutput,
    pub execution: ExecutionIdentity,
}

impl CompletedMossImageEdit {
    pub fn new(
        output: crate::moss_image::MossImageAdvancedOutput,
        execution: ExecutionIdentity,
        descriptor: &ModelDescriptor,
    ) -> Result<Self> {
        let units = match &output {
            crate::moss_image::MossImageAdvancedOutput::Images(images) => images.len(),
            crate::moss_image::MossImageAdvancedOutput::Latents { .. } => 1,
        };
        execution.validate(descriptor.id, descriptor.backend, units)?;
        Ok(Self { output, execution })
    }
}

/// One successful conversational audio turn and its attempt-local facts. A context-only turn
/// remains distinguishable from generated speech in the provider's identity.
#[derive(Debug)]
pub struct CompletedAudioTurn {
    pub audio: crate::AudioTrack,
    pub execution: ExecutionIdentity,
}

impl CompletedAudioTurn {
    pub fn new(
        audio: crate::AudioTrack,
        execution: ExecutionIdentity,
        descriptor: &ModelDescriptor,
    ) -> Result<Self> {
        execution.validate(descriptor.id, descriptor.backend, 1)?;
        Ok(Self { audio, execution })
    }

    pub fn without_identity(audio: crate::AudioTrack) -> Self {
        Self {
            audio,
            execution: ExecutionIdentity::provider_default(),
        }
    }
}

fn generation_output_units(report: &GenerationReport) -> Result<usize> {
    let units = match &report.output {
        Some(GenerationOutput::Images(images)) => images.len(),
        Some(GenerationOutput::ImagesRgba(images)) => images.len(),
        Some(GenerationOutput::Video { .. } | GenerationOutput::Audio(_)) => 1,
        None if report.artifacts.is_some() => 1,
        None => 0,
    };
    if units == 0 {
        return Err(Error::Msg(
            "completed generation has no output or artifact".into(),
        ));
    }
    Ok(units)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AudioChunk, AudioTrack, Capabilities, ConversationRole, ConversationSession,
        ConversationTurn, GenerationRequest, Generator, Modality, Progress,
    };

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "execution-test",
            family: "test",
            backend: "candle",
            modality: Modality::Audio,
            capabilities: Capabilities {
                max_count: 1,
                ..Default::default()
            },
        }
    }

    fn track() -> AudioTrack {
        AudioTrack {
            samples: vec![0.25, -0.25],
            sample_rate: 48_000,
            channels: 1,
            stems: Vec::new(),
        }
    }

    fn resolved() -> ResolvedExecutionIdentity {
        ResolvedExecutionIdentity {
            version: EXECUTION_IDENTITY_VERSION,
            provider_id: "execution-test".into(),
            backend: "candle".into(),
            runtime_id: "test-provider@1+numerics-a".into(),
            sampling: ExecutionFact::resolved(serde_json::json!({"sampler":"categorical"})),
            schedule: ExecutionFact::not_applicable("autoregressive tokens have no noise schedule"),
            estimator: ExecutionFact::not_applicable("no diffusion estimator"),
            randomness: ExecutionRandomness::Draws {
                scheme: "seed-per-turn-v1".into(),
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
            geometry: ExecutionFact::not_applicable("audio output"),
            quality_policy: ExecutionFact::resolved(serde_json::json!({"precision":"f32"})),
            output_policy: ExecutionFact::resolved(
                serde_json::json!({"sample_rate":48000,"channels":1}),
            ),
            provider_choices: BTreeMap::new(),
            output_units: 1,
        }
    }

    #[test]
    fn completed_identity_requires_actual_provider_and_output_mapping() {
        let generation = || GenerationReport::from_output(GenerationOutput::Audio(track()));
        let identity = ExecutionIdentity::Resolved {
            identity: Box::new(resolved()),
        };
        let completed = CompletedGeneration::new(generation(), identity.clone(), &descriptor())
            .expect("complete provider facts");
        assert_eq!(completed.execution.resolved().unwrap().output_units, 1);
        let encoded = serde_json::to_string(&identity).unwrap();
        assert_eq!(
            serde_json::from_str::<ExecutionIdentity>(&encoded).unwrap(),
            identity
        );

        let mut wrong = resolved();
        wrong.output_units = 2;
        assert!(CompletedGeneration::new(
            generation(),
            ExecutionIdentity::Resolved {
                identity: Box::new(wrong)
            },
            &descriptor()
        )
        .is_err());
        let mut wrong = resolved();
        wrong.provider_id = "a different model".into();
        assert!(CompletedGeneration::new(
            generation(),
            ExecutionIdentity::Resolved {
                identity: Box::new(wrong)
            },
            &descriptor()
        )
        .is_err());
    }

    #[test]
    fn incomplete_or_ambiguous_execution_facts_do_not_become_replay_receipts() {
        let mut invalid = resolved();
        invalid.version += 1;
        assert!(invalid.validate("execution-test", "candle", 1).is_err());
        let mut invalid = resolved();
        invalid.sampling = ExecutionFact::resolved(Value::Null);
        assert!(invalid.validate("execution-test", "candle", 1).is_err());
        let mut invalid = resolved();
        invalid.randomness = ExecutionRandomness::Draws {
            scheme: "unknown".into(),
            draws: vec![ExecutionDraw {
                output_index: 0,
                purpose: "tokens".into(),
                seed: None,
                noise_sha256: None,
            }],
        };
        assert!(invalid.validate("execution-test", "candle", 1).is_err());
        let mut invalid = resolved();
        if let ExecutionRandomness::Draws { draws, .. } = &mut invalid.randomness {
            draws.push(draws[0].clone());
        }
        assert!(invalid.validate("execution-test", "candle", 1).is_ok());
    }

    #[test]
    fn schedule_digest_uses_numeric_bits_and_rejects_nonfinite_values() {
        let positive_zero = NumericScheduleDigest::f32_values(&[0.0, 1.0]).unwrap();
        let negative_zero = NumericScheduleDigest::f32_values(&[-0.0, 1.0]).unwrap();
        assert_ne!(positive_zero.sha256, negative_zero.sha256);
        assert_eq!(positive_zero.count, 2);
        assert_eq!(positive_zero.element, "f32_le_bits");
        assert!(NumericScheduleDigest::f32_values(&[]).is_err());
        assert!(NumericScheduleDigest::f64_values(&[f64::NAN]).is_err());
    }

    #[test]
    fn compatibility_default_never_claims_submitted_values_as_resolved() {
        struct Legacy(ModelDescriptor);
        impl Generator for Legacy {
            fn descriptor(&self) -> &ModelDescriptor {
                &self.0
            }
            fn validate(&self, _req: &GenerationRequest) -> Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _req: &GenerationRequest,
                _on_progress: &mut dyn FnMut(Progress),
            ) -> Result<GenerationOutput> {
                Ok(GenerationOutput::Audio(track()))
            }
        }
        let generator = Legacy(descriptor());
        let req = GenerationRequest {
            seed: Some(17),
            sampler: Some("client-claim".into()),
            ..Default::default()
        };
        let plan = generator.plan_execution(&req).unwrap();
        assert!(matches!(plan, ExecutionIdentity::Unavailable { .. }));
        let completed = generator
            .generate_with_execution_report(&req, &mut |_| {})
            .unwrap();
        assert!(matches!(
            completed.execution,
            ExecutionIdentity::Unavailable { .. }
        ));
        let mut chunks = Vec::new();
        let streamed = generator
            .generate_streaming_with_execution_report(
                &req,
                &mut |chunk| chunks.push(chunk),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(
            streamed.execution,
            ExecutionIdentity::Unavailable { .. }
        ));
    }

    #[test]
    fn failed_attempt_has_no_completed_report() {
        struct Failing(ModelDescriptor);
        impl Generator for Failing {
            fn descriptor(&self) -> &ModelDescriptor {
                &self.0
            }
            fn validate(&self, _req: &GenerationRequest) -> Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _req: &GenerationRequest,
                _on_progress: &mut dyn FnMut(Progress),
            ) -> Result<GenerationOutput> {
                Err(Error::Canceled)
            }
        }
        assert!(matches!(
            Failing(descriptor())
                .generate_with_execution_report(&GenerationRequest::default(), &mut |_| {}),
            Err(Error::Canceled)
        ));
    }

    #[test]
    fn direct_edit_and_conversation_turn_have_separate_attempt_local_reports() {
        let edit = CompletedMossImageEdit::new(
            crate::moss_image::MossImageAdvancedOutput::Images(vec![crate::Image {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0],
            }]),
            ExecutionIdentity::provider_default(),
            &descriptor(),
        )
        .unwrap();
        assert!(matches!(
            edit.execution,
            ExecutionIdentity::Unavailable { .. }
        ));

        struct Session;
        impl ConversationSession for Session {
            fn step(
                &mut self,
                _turn: &ConversationTurn,
                _on_chunk: &mut dyn FnMut(AudioChunk),
                _on_progress: &mut dyn FnMut(Progress),
            ) -> Result<AudioTrack> {
                Ok(track())
            }
        }
        let turn = ConversationTurn {
            role: ConversationRole::Assistant,
            text: "hello".into(),
            audio: None,
        };
        let completed = Session
            .step_with_execution_report(&turn, &mut |_| {}, &mut |_| {})
            .unwrap();
        assert!(matches!(
            completed.execution,
            ExecutionIdentity::Unavailable { .. }
        ));
        assert!(CompletedAudioTurn::new(
            track(),
            ExecutionIdentity::Resolved {
                identity: Box::new(resolved())
            },
            &descriptor()
        )
        .is_ok());
    }
}
