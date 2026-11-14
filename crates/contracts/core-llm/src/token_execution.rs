//! Attempt-local observations made by native token decoders, never reconstructed from requests.
//!
//! The synchronous scope is thread-bound, nestable, and cleared on every exit (including panic).
//! Providers return the observation with the output; there is no mutable "last report" accessor.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use serde_json::{json, Value};

use crate::{
    CompletedStarVector, CompletedTextLlm, DecodeReport, ExecutionDraw, ExecutionFact,
    ExecutionIdentity, ExecutionRandomness, ExecutionReplaySupport, ExecutionUnavailable,
    FinishReason, ResolvedExecutionIdentity, Result, Sampling, StarVectorOutput, TextLlmDescriptor,
    TextLlmOutput, EXECUTION_IDENTITY_VERSION,
};

#[derive(Default)]
struct Observations {
    id: usize,
    suspended: bool,
    samplers: Vec<Value>,
    seeds: Vec<u64>,
    stochastic: Vec<bool>,
    policies: Vec<Value>,
    decodes: Vec<String>,
    image_schedules: Vec<Value>,
    document_draws: Vec<ExecutionDraw>,
    image_schemes: Vec<String>,
}
thread_local! {
    static ATTEMPTS: RefCell<Vec<Observations>> = const { RefCell::new(Vec::new()) };
}

/// A scope must be created on the same thread as the synchronous native decode.
#[must_use]
pub struct TokenExecutionScope(usize, PhantomData<Rc<()>>);

impl Default for TokenExecutionScope {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenExecutionScope {
    pub fn new() -> Self {
        let id = ATTEMPTS.with(|stack| {
            let mut stack = stack.borrow_mut();
            let id = stack.last().map_or(0, |frame| frame.id + 1);
            stack.push(Observations {
                id,
                ..Default::default()
            });
            id
        });
        Self(id, PhantomData)
    }

    /// No request fields enter this report. The sampler, seed and budgets come from the decode
    /// configuration actually consumed; decode/cache choices come from the completed output.
    pub fn finish_text(
        self,
        output: TextLlmOutput,
        descriptor: &TextLlmDescriptor,
    ) -> Result<CompletedTextLlm> {
        let complete = matches!(
            output.finish_reason,
            Some(FinishReason::Stop | FinishReason::Length)
        );
        let execution = self.identity(descriptor, output.decode.as_ref(), format!("{:?}", output.kv_cache),
            json!({"format":"text", "finish_reason": format!("{:?}", output.finish_reason),
                "thinking_channel": output.thinking.is_some(), "tool_calls": !output.tool_calls.is_empty()}), complete);
        CompletedTextLlm::new(output, execution, descriptor)
    }

    pub fn finish_svg(
        self,
        output: StarVectorOutput,
        decode: Option<&DecodeReport>,
        descriptor: &TextLlmDescriptor,
    ) -> Result<CompletedStarVector> {
        let execution = self.identity(descriptor, decode, format!("{:?}", output.kv_cache),
            json!({"format":"svg", "root_policy":"exactly_one_complete_root", "finish_reason":format!("{:?}", output.finish_reason)}), output.svg.is_some());
        CompletedStarVector::new(output, execution, descriptor)
    }

    /// Complete one VQA answer or interleaved document. Every internal text/image stream maps
    /// to that single artifact, in execution order; the output policy names its image count.
    pub fn finish_document(
        self,
        provider_id: &str,
        backend: &str,
        output_policy: Value,
    ) -> ExecutionIdentity {
        ATTEMPTS.with(|stack| {
            let stack=stack.borrow(); let observed=stack.iter().find(|frame| frame.id == self.0).expect("owned scope");
            if observed.decodes.is_empty() || observed.samplers.is_empty() {
                return unavailable("document_execution_not_observed", "The direct runtime did not complete an observed token decode", "completed_token_decode");
            }
            let image_count=output_policy.get("image_count").and_then(Value::as_u64).unwrap_or(0) as usize;
            if observed.image_schedules.len()!=image_count || observed.image_schemes.len()!=image_count || observed.samplers.len()!=observed.decodes.len() {
                return unavailable("document_stream_mapping_unavailable", "Completed document segments differ from the observed token/image streams", "document_stream_mapping");
            }
            ExecutionIdentity::Resolved { identity:Box::new(ResolvedExecutionIdentity {
                version:EXECUTION_IDENTITY_VERSION, provider_id:provider_id.into(), backend:backend.into(),
                runtime_id:format!("{backend}/{provider_id}/token-document-v1"),
                sampling:ExecutionFact::resolved(json!({"token_segments":observed.samplers,"image_solver":if image_count > 0 {"flow_euler"} else {"not_run"}})),
                schedule: if observed.image_schedules.is_empty() { ExecutionFact::not_applicable("no image denoise executed") }
                    else { ExecutionFact::resolved(json!(observed.image_schedules)) },
                estimator:if image_count > 0 {ExecutionFact::resolved("flow_velocity_from_predicted_pixels")} else {ExecutionFact::not_applicable("autoregressive token decode")},
                randomness:if observed.document_draws.is_empty() {
                    ExecutionRandomness::NoDraws {reason:"Greedy token decoding; no image noise was consumed".into()}
                } else {ExecutionRandomness::Draws {scheme:format!("splitmix64-token-stream-v1; images:{:?}",observed.image_schemes),draws:observed.document_draws.clone()}},
                deterministic_replay:ExecutionReplaySupport::Deterministic {scope:"Same pinned model/tokenizer bytes, inference revision, device, dtype and effective memory/attention policy; identical conditioning and recorded noise streams".into()},
                geometry:ExecutionFact::resolved(output_policy.clone()),
                quality_policy:ExecutionFact::resolved(json!({"decode":observed.decodes,"execution_policies":observed.policies})),
                output_policy:ExecutionFact::resolved(output_policy), provider_choices:Default::default(),output_units:1,
            }) }
        })
    }

    fn identity(
        &self,
        descriptor: &TextLlmDescriptor,
        decode: Option<&DecodeReport>,
        kv: String,
        output: Value,
        complete: bool,
    ) -> ExecutionIdentity {
        if !complete {
            return unavailable(
                "incomplete_token_output",
                "The attempt was cancelled or stopped before a publishable output",
                "completed_output",
            );
        }
        let Some(decode) = decode else {
            return unavailable(
                "token_decode_not_observed",
                "This execution path did not return a measured decode report",
                "decode_path",
            );
        };
        ATTEMPTS.with(|stack| {
            let stack = stack.borrow();
            let observed = stack.iter().find(|frame| frame.id == self.0).expect("owned token execution scope");
            // Multiple decoder invocations can belong to nested unreported calls or a pipeline.
            // Without an explicit output mapping they cannot be attributed to this output.
            if observed.seeds.len() != 1 {
                return unavailable("token_rng_mapping_unavailable", "Expected one observed token stream; this path has absent or multiple decoder streams without an output mapping", "token_rng_mapping");
            }
            ExecutionIdentity::Resolved { identity: Box::new(ResolvedExecutionIdentity {
                version: EXECUTION_IDENTITY_VERSION,
                provider_id: descriptor.id.clone(), backend: descriptor.backend.clone(),
                runtime_id: format!("{}-llm/token-decode-v1/{}", descriptor.backend, decode.path),
                sampling: ExecutionFact::resolved(json!({"controls": observed.samplers[0], "implementation":decode.sampler})),
                schedule: ExecutionFact::not_applicable("autoregressive token decode"),
                estimator: ExecutionFact::not_applicable("no diffusion estimator"),
                randomness: if !observed.stochastic[0] || decode.sampler == "none" {
                    ExecutionRandomness::NoDraws {reason:"The completed decode used greedy sampling or performed no token draw".into()}
                } else {ExecutionRandomness::Draws { scheme:"splitmix64-token-stream-v1".into(), draws: vec![ExecutionDraw {
                    output_index:0, purpose:"autoregressive_tokens".into(), seed:Some(observed.seeds[0]), noise_sha256:None,
                }] }},
                deterministic_replay: token_replay_support(decode, &observed.policies),
                geometry: ExecutionFact::not_applicable("token source output has no raster geometry"),
                quality_policy: ExecutionFact::resolved(json!({"decode":format!("{decode:?}"), "kv_cache":kv})),
                output_policy: ExecutionFact::resolved(json!({"decoder_policies":observed.policies,"completed":output})),
                provider_choices: Default::default(), output_units:1,
            }) }
        })
    }
}
impl Drop for TokenExecutionScope {
    fn drop(&mut self) {
        ATTEMPTS.with(|stack| {
            stack.borrow_mut().retain(|frame| frame.id != self.0);
        });
    }
}

/// Consumer callbacks can reenter a legacy provider. Keep that unrelated decode out of the
/// current receipt while allowing a nested report-aware call to create its own scope.
pub fn without_observations<T>(operation: impl FnOnce() -> T) -> T {
    struct Resume(Option<(usize, bool)>);
    impl Drop for Resume {
        fn drop(&mut self) {
            if let Some((id, suspended)) = self.0 {
                ATTEMPTS.with(|stack| {
                    if let Some(frame) = stack.borrow_mut().iter_mut().find(|frame| frame.id == id)
                    {
                        frame.suspended = suspended;
                    }
                });
            }
        }
    }
    let previous = ATTEMPTS.with(|stack| {
        stack.borrow_mut().last_mut().map(|frame| {
            let previous = (frame.id, frame.suspended);
            frame.suspended = true;
            previous
        })
    });
    let _resume = Resume(previous);
    operation()
}

/// Record at RNG construction, after resolving the fallback seed, then return that exact seed
/// unchanged to the sampler. Floating point controls retain their IEEE-754 bits.
pub fn observe_token_seed(
    sampling: Sampling,
    seed: u64,
    max_new_tokens: usize,
    stop_tokens: &[i32],
) -> u64 {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack.borrow_mut().last_mut().filter(|frame| !frame.suspended) {
            let bits = |value: f32| format!("f32:{:08x}", value.to_bits());
            attempt.samplers.push(json!({
                "temperature":bits(sampling.temperature), "top_p":bits(sampling.top_p), "top_k":sampling.top_k,
                "presence_penalty":bits(sampling.presence_penalty), "presence_penalty_generated_only":sampling.presence_penalty_generated_only, "presence_penalty_prompt_len":0,
                "repetition_penalty":bits(sampling.repetition_penalty), "repetition_context":sampling.repetition_context,
            }));
            if sampling.temperature > 0.0 {
                attempt.document_draws.push(ExecutionDraw { output_index:0,
                    purpose:format!("text_segment_{}",attempt.seeds.len()),seed:Some(seed),noise_sha256:None });
            }
            attempt.stochastic.push(sampling.temperature > 0.0);
            attempt.seeds.push(seed);
            attempt.policies.push(json!({"max_new_tokens":max_new_tokens,"stop_tokens":stop_tokens}));
        }
    });
    seed
}

/// Observe the actual expanded history boundary when the decode sampler resets it after prefill.
pub fn observe_token_prompt_boundary(prompt_len: usize) {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            if let Some(sampling) = attempt.samplers.last_mut() {
                sampling["presence_penalty_prompt_len"] = json!(prompt_len);
            }
        }
    });
}

/// Record a completed internal token segment on the direct understanding route.
pub fn observe_decode(decode: &DecodeReport) {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            attempt.decodes.push(format!("{decode:?}"));
        }
    });
}

/// Called with effective memory/runtime policy at the model's actual execution seam.
pub fn observe_document_policy(policy: Value) {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            attempt.policies.push(policy);
        }
    });
}

pub fn observe_image_schedule(values: &[f32]) {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            if let Ok(digest) = crate::NumericScheduleDigest::f32_values(values) {
                attempt.image_schedules.push(json!(digest));
            }
        }
    });
}

/// Actual noise producer: seed or digest of the explicit f32 tensor, never an options seed
/// substituted for caller-provided noise. One document can contain multiple image streams.
pub fn observe_document_image(
    index: usize,
    scheme: &str,
    seed: Option<u64>,
    explicit_noise: Option<&[f32]>,
) {
    use sha2::{Digest, Sha256};
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            let noise_sha256 = explicit_noise.map(|values| {
                let mut hash = Sha256::new();
                for value in values {
                    hash.update(value.to_bits().to_le_bytes());
                }
                format!("{:x}", hash.finalize())
            });
            attempt.image_schemes.push(scheme.into());
            attempt.document_draws.push(ExecutionDraw {
                output_index: 0,
                purpose: format!("image_{index}"),
                seed,
                noise_sha256,
            });
        }
    });
}

/// Avoid tensor-to-host copies for explicit-noise hashing outside a report-aware call.
pub fn observing() -> bool {
    ATTEMPTS.with(|stack| stack.borrow().last().is_some_and(|frame| !frame.suspended))
}

/// Record the SVG limits at the bounded stream that enforces them.
pub(crate) fn observe_svg_policy(request: &crate::StarVectorRequest) {
    ATTEMPTS.with(|stack| {
        if let Some(attempt) = stack
            .borrow_mut()
            .last_mut()
            .filter(|frame| !frame.suspended)
        {
            attempt
                .policies
                .push(json!({"max_svg_bytes":request.max_svg_bytes,
                "max_wall_time_ns":request.max_wall_time.as_nanos().to_string(),
                "root_policy":"exactly_one_complete_root"}));
        }
    });
}

fn token_replay_support(decode: &DecodeReport, policies: &[Value]) -> ExecutionReplaySupport {
    let reason = if decode.speculative_monitor.is_some() {
        Some("Adaptive speculation consumed elapsed-time observations; the next run may select a different decode path")
    } else if decode.prefix_hit_tokens > 0 {
        Some("Decode reused a live prefix cache; its numerical state is not contained in this execution report")
    } else if policies
        .iter()
        .any(|policy| policy.get("max_wall_time_ns").is_some())
    {
        Some("SVG's bounded stream consumes elapsed wall time and may terminate at a different token on replay")
    } else {
        None
    };
    match reason {
        Some(reason)=>ExecutionReplaySupport::NonDeterministic {reason:reason.into()},
        None=>ExecutionReplaySupport::Deterministic {scope:"Same pinned model/tokenizer bytes, inference revision, device, dtype and effective decode/cache policy; identical conditioning and recorded RNG stream".into()},
    }
}

pub fn unavailable(code: &str, detail: &str, missing: &str) -> ExecutionIdentity {
    ExecutionIdentity::unavailable(ExecutionUnavailable {
        code: code.into(),
        detail: detail.into(),
        missing_facts: vec![missing.into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn descriptor() -> TextLlmDescriptor {
        TextLlmDescriptor {
            id: "native-test".into(),
            backend: "candle".into(),
            family: "test".into(),
            capabilities: Default::default(),
        }
    }
    fn output() -> TextLlmOutput {
        TextLlmOutput {
            text: "done".into(),
            finish_reason: Some(FinishReason::Stop),
            decode: Some(DecodeReport {
                path: "step_model".into(),
                sampler: "host:penalty".into(),
                kv_cache: "growing".into(),
                fallbacks: vec!["no draft loaded".into()],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    fn observe(seed: u64) {
        assert_eq!(
            observe_token_seed(Sampling::default(), seed, 19, &[2, 3]),
            seed
        );
    }
    #[test]
    fn actual_seed_controls_and_runtime_fallback_are_bound_to_output() {
        let scope = TokenExecutionScope::new();
        observe(123);
        observe_token_prompt_boundary(577);
        let completed = scope.finish_text(output(), &descriptor()).unwrap();
        let identity = completed.execution.resolved().unwrap();
        identity.validate("native-test", "candle", 1).unwrap();
        assert_eq!(
            identity.randomness,
            ExecutionRandomness::Draws {
                scheme: "splitmix64-token-stream-v1".into(),
                draws: vec![ExecutionDraw {
                    output_index: 0,
                    purpose: "autoregressive_tokens".into(),
                    seed: Some(123),
                    noise_sha256: None
                }]
            }
        );
        assert!(format!("{:?}", identity.sampling).contains("f32:3f333333"));
        assert!(format!("{:?}", identity.sampling).contains("577"));
        assert!(format!("{:?}", identity.quality_policy).contains("no draft loaded"));
        assert!(format!("{:?}", identity.output_policy).contains("19"));
    }
    #[test]
    fn cancellation_retry_and_missing_sampler_never_reuse_success() {
        let scope = TokenExecutionScope::new();
        observe(1);
        assert!(scope
            .finish_text(output(), &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_some());
        let scope = TokenExecutionScope::new();
        observe(2);
        let mut cancelled = output();
        cancelled.finish_reason = Some(FinishReason::Cancelled);
        assert!(scope
            .finish_text(cancelled, &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_none());
        let scope = TokenExecutionScope::new();
        assert!(scope
            .finish_text(output(), &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_none());
        let scope = TokenExecutionScope::new();
        observe(3);
        let out = scope.finish_text(output(), &descriptor()).unwrap();
        assert!(format!("{:?}", out.execution.resolved().unwrap().randomness).contains("Some(3)"));
    }
    #[test]
    fn nested_scopes_and_panic_do_not_contaminate_caller() {
        let outer = TokenExecutionScope::new();
        observe(1);
        let _ = std::panic::catch_unwind(|| {
            let _inner = TokenExecutionScope::new();
            observe(2);
            panic!("fixture");
        });
        assert!(outer
            .finish_text(output(), &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_some());
        let fresh = TokenExecutionScope::new();
        assert!(fresh
            .finish_text(output(), &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_none());
    }
    #[test]
    fn ambiguous_streams_and_absent_decode_fail_closed() {
        let scope = TokenExecutionScope::new();
        observe(1);
        observe(2);
        assert!(scope
            .finish_text(output(), &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_none());
        let scope = TokenExecutionScope::new();
        observe(1);
        let mut missing = output();
        missing.decode = None;
        assert!(scope
            .finish_text(missing, &descriptor())
            .unwrap()
            .execution
            .resolved()
            .is_none());
    }
}

#[cfg(test)]
mod document_tests {
    use super::*;

    #[test]
    fn interleave_keeps_draw_order_explicit_noise_and_actual_schedule() {
        let scope = TokenExecutionScope::new();
        observe_token_seed(Sampling::default(), 0, 16, &[9]);
        observe_decode(&DecodeReport {
            path: "step_model".into(),
            ..Default::default()
        });
        observe_document_image(
            0,
            "explicit-f32-noise-sha256",
            None,
            Some(&[0.0, -0.0, 1.0]),
        );
        observe_image_schedule(&[0.0, 0.25, 1.0]);
        observe_token_seed(Sampling::default(), 0, 8, &[9]);
        observe_decode(&DecodeReport {
            path: "step_model".into(),
            ..Default::default()
        });
        let execution = scope.finish_document(
            "sensenova_u1_8b",
            "mlx",
            json!({"format":"interleaved_document","image_count":1}),
        );
        execution.validate("sensenova_u1_8b", "mlx", 1).unwrap();
        let identity = execution.resolved().unwrap();
        let ExecutionRandomness::Draws { draws, .. } = &identity.randomness else {
            panic!("draws expected")
        };
        assert_eq!(
            draws
                .iter()
                .map(|draw| draw.purpose.as_str())
                .collect::<Vec<_>>(),
            ["text_segment_0", "image_0", "text_segment_1"]
        );
        assert!(draws[1].seed.is_none());
        assert_eq!(draws[1].noise_sha256.as_ref().unwrap().len(), 64);
        assert!(format!("{:?}", identity.schedule).contains("f32_le_bits"));
    }

    #[test]
    fn greedy_vqa_has_no_random_draws() {
        let scope = TokenExecutionScope::new();
        observe_token_seed(Sampling::greedy(), 42, 8, &[]);
        observe_decode(&DecodeReport {
            sampler: "device".into(),
            ..Default::default()
        });
        let execution = scope.finish_document("vqa", "candle", json!({"format":"text"}));
        execution.validate("vqa", "candle", 1).unwrap();
        assert!(matches!(
            execution.resolved().unwrap().randomness,
            ExecutionRandomness::NoDraws { .. }
        ));
    }

    #[test]
    fn missing_image_schedule_or_output_mapping_is_unavailable() {
        let scope = TokenExecutionScope::new();
        observe_token_seed(Sampling::default(), 0, 8, &[]);
        observe_decode(&DecodeReport::default());
        observe_document_image(0, "splitmix64", Some(7), None);
        assert!(scope
            .finish_document("test", "mlx", json!({"image_count":1}))
            .resolved()
            .is_none());
    }

    #[test]
    fn out_of_order_scope_completion_keeps_each_attempt_own_facts() {
        let first = TokenExecutionScope::new();
        observe_token_seed(Sampling::default(), 1, 8, &[]);
        observe_decode(&DecodeReport::default());
        let second = TokenExecutionScope::new();
        observe_token_seed(Sampling::default(), 2, 8, &[]);
        observe_decode(&DecodeReport::default());
        let a = first.finish_document("test", "mlx", json!({"format":"text"}));
        let b = second.finish_document("test", "mlx", json!({"format":"text"}));
        let seed = |report: ExecutionIdentity| {
            let ExecutionRandomness::Draws { draws, .. } = &report.resolved().unwrap().randomness
            else {
                panic!()
            };
            draws[0].seed
        };
        assert_eq!(seed(a), Some(1));
        assert_eq!(seed(b), Some(2));
    }

    #[test]
    fn callback_legacy_decode_is_excluded_but_nested_report_retains_its_own_stream() {
        let scope = TokenExecutionScope::new();
        observe_token_seed(Sampling::default(), 1, 8, &[]);
        observe_decode(&DecodeReport::default());
        without_observations(|| {
            assert!(!observing());
            observe_token_seed(Sampling::default(), 9, 8, &[]);
            observe_decode(&DecodeReport::default());
            let nested = TokenExecutionScope::new();
            observe_token_seed(Sampling::default(), 2, 8, &[]);
            observe_decode(&DecodeReport::default());
            assert!(nested
                .finish_document("nested", "mlx", json!({"format":"text"}))
                .resolved()
                .is_some());
        });
        let execution = scope.finish_document("outer", "mlx", json!({"format":"text"}));
        let ExecutionRandomness::Draws { draws, .. } = &execution.resolved().unwrap().randomness
        else {
            panic!()
        };
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].seed, Some(1));
    }

    // Static policy metadata, not a measurement or a clock-dependent assertion.
    fn svg_policy_fixture() -> Value {
        json!({"max_wall_time_ns":"100"})
    }

    #[test]
    fn replay_support_names_actual_dynamic_dependency() {
        let mut decode = DecodeReport::default();
        assert!(token_replay_support(&decode, &[]).is_deterministic());
        decode.prefix_hit_tokens = 4;
        assert!(!token_replay_support(&decode, &[]).is_deterministic());
        decode.prefix_hit_tokens = 0;
        assert!(!token_replay_support(&decode, &[svg_policy_fixture()]).is_deterministic());
    }
}
