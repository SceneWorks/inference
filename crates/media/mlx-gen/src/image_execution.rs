//! Completed image execution observations. The scope is synchronous and attempt-local: nested
//! generation shadows its caller, and errors, cancellation and unwinding discard observations.
//! Nothing here resolves a request or reads a previous generation's report.
use std::{cell::RefCell, collections::BTreeMap};

use crate::gen_core::generator::{
    CompletedGeneration, ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness,
    ExecutionReplaySupport, ExecutionUnavailable, NumericScheduleDigest, ResolvedExecutionIdentity,
    EXECUTION_IDENTITY_VERSION,
};
use crate::{CancelFlag, GenerationOutput, ModelDescriptor};
use serde_json::{json, Value};

#[derive(Default)]
struct Observations {
    sampling: Vec<Value>,
    schedules: Vec<NumericScheduleDigest>,
    seeds: Vec<u64>,
    output_indices: Vec<usize>,
    binding: Binding,
    schemes: Vec<String>,
    decode: Vec<(Value, NumericScheduleDigest, u64, String)>,
    policies: BTreeMap<String, Value>,
    choices: BTreeMap<String, ExecutionFact>,
    unavailable: Option<ExecutionUnavailable>,
}

#[derive(Clone, Default)]
struct Binding {
    output_index: Option<usize>,
    primary_seed: Option<u64>,
    primary_rng: Option<String>,
}

struct BindingScope(Option<Binding>);
impl Drop for BindingScope {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            CURRENT.with(|current| {
                if let Some(observed) = current.borrow_mut().as_mut() {
                    observed.binding = previous;
                }
            });
        }
    }
}
fn bind<T>(change: impl FnOnce(&mut Binding), body: impl FnOnce() -> T) -> T {
    let previous = CURRENT.with(|current| {
        current.borrow_mut().as_mut().map(|observed| {
            let previous = observed.binding.clone();
            change(&mut observed.binding);
            previous
        })
    });
    let _scope = BindingScope(previous);
    body()
}

/// Explicitly bind multiple sampling stages to one output. Without this scope each observed
/// sampler is one image in call order; an unbound extra sampler fails completed-count validation.
pub fn with_output_index<T>(output_index: usize, body: impl FnOnce() -> T) -> T {
    bind(|binding| binding.output_index = Some(output_index), body)
}

/// Carry the actual initial-noise seed from its owner into a low-level native loop that accepts
/// already-created noise. Never populate this from a separate request projection.
pub fn with_primary_seed<T>(seed: u64, body: impl FnOnce() -> T) -> T {
    bind(|binding| binding.primary_seed = Some(seed), body)
}

/// Override the primary stream description at the actual noise owner (for example global MLX
/// initial noise followed by keyed FLOW steps). This does not change any RNG or seed.
pub fn with_primary_rng_scheme<T>(rng_scheme: &str, body: impl FnOnce() -> T) -> T {
    bind(
        |binding| binding.primary_rng = Some(rng_scheme.into()),
        body,
    )
}

/// Native loop observation when the initial-noise owner scoped the actual seed above it.
pub fn sampled_native_from_primary_seed(
    solver: &str,
    convention: &str,
    dtype: mlx_rs::Dtype,
    schedule: &[f32],
    rng_scheme: &str,
) {
    let seed = CURRENT.with(|current| {
        current
            .borrow()
            .as_ref()
            .and_then(|observed| observed.binding.primary_seed)
    });
    if let Some(seed) = seed {
        sampled_native_with_rng(solver, convention, dtype, schedule, seed, rng_scheme);
    } else {
        unavailable(
            "mlx_image_initial_seed_unobserved",
            "native loop received external noise without an observed initial seed",
            "initial_noise_seed",
        );
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Observations>> = const { RefCell::new(None) };
}

struct Scope(Option<Observations>);
impl Scope {
    fn enter(observations: Option<Observations>) -> Self {
        Self(CURRENT.with(|current| current.replace(observations)))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.0.take()));
    }
}

/// Isolate ordinary generation and callbacks from an enclosing reporting attempt. This also
/// avoids retaining metadata when the caller only requested pixels.
pub fn without_capture<T>(body: impl FnOnce() -> T) -> T {
    let _scope = Scope::enter(None);
    body()
}

/// Run exactly one provider invocation, publishing only after output and cancellation validation.
/// The closure must run synchronously on this thread; absent observations fail closed.
pub fn complete_image_attempt(
    descriptor: &ModelDescriptor,
    cancel: &CancelFlag,
    body: impl FnOnce() -> crate::gen_core::Result<GenerationOutput>,
) -> crate::gen_core::Result<CompletedGeneration> {
    complete_image_report_attempt(descriptor, cancel, || {
        body().map(crate::gen_core::GenerationReport::from_output)
    })
}

/// Preserve the provider's artifacts, warnings and other rich output fields in the same attempt.
pub fn complete_image_report_attempt(
    descriptor: &ModelDescriptor,
    cancel: &CancelFlag,
    body: impl FnOnce() -> crate::gen_core::Result<crate::gen_core::GenerationReport>,
) -> crate::gen_core::Result<CompletedGeneration> {
    let _scope = Scope::enter(Some(Observations::default()));
    let generation = body()?;
    if cancel.is_cancelled() {
        return Err(crate::gen_core::Error::Canceled);
    }
    let observed = CURRENT
        .with(|current| current.borrow_mut().take())
        .unwrap_or_default();
    let execution = observed.finish(descriptor, generation.output.as_ref())?;
    CompletedGeneration::new(generation, execution, descriptor)
}

/// Observe values already resolved by the provider and passed to its execution path.
pub fn policy(name: &str, value: Value) {
    CURRENT.with(|current| {
        if let Some(observed) = current.borrow_mut().as_mut() {
            if observed.policies.get(name).is_some_and(|old| old != &value) {
                observed.unavailable = Some(reason(
                    "mlx_image_ambiguous_policy",
                    "multiple different policy observations occurred in one image attempt",
                    name,
                ));
            }
            observed.policies.insert(name.into(), value);
        }
    });
}

/// Debug encodings here describe resolved integer/bool memory plans, not request fields.
pub fn policy_debug(name: &str, value: &impl std::fmt::Debug) {
    policy(name, Value::String(format!("{value:?}")));
}

/// Exact IEEE-754 representation of an effective scalar consumed by the provider.
pub fn scalar(name: &str, value: f32) {
    match ExecutionFact::f32_bits(value) {
        Ok(value) => choice(name, value),
        Err(_) => unavailable(
            "mlx_image_nonfinite_choice",
            "an effective scalar is nonfinite",
            name,
        ),
    }
}

/// Shared FLOW factory fallback is the same function the runtime uses to choose the solver.
pub fn sampled_flow(
    sampler: Option<&str>,
    convention: crate::gen_core::sampling::TimestepConvention,
    dtype: mlx_rs::Dtype,
    sigmas: &[f32],
    seed: u64,
) {
    sampled_native(
        crate::sampler::canonical_solver(sampler, crate::Solver::Euler).name(),
        &format!("{convention:?}"),
        dtype,
        sigmas,
        seed,
    );
}

/// Dedicated non-FLOW solvers supply the actual name and schedule parameterization.
pub fn sampled_native(
    solver: &str,
    convention: &str,
    dtype: mlx_rs::Dtype,
    schedule: &[f32],
    seed: u64,
) {
    sampled_native_with_rng(solver, convention, dtype, schedule, seed,
        "mlx-keyed-normal-v1;initial=output-seed;step=seed+0x9e3779b97f4a7c15*(step+1),wrapping-u64;SCM-step=absolute-schedule-index");
}

/// Native solvers with a different RNG must identify the stream they actually consumed.
pub fn sampled_native_with_rng(
    solver: &str,
    convention: &str,
    dtype: mlx_rs::Dtype,
    schedule: &[f32],
    seed: u64,
    rng_scheme: &str,
) {
    sampled(
        json!({"solver": solver, "timestep_convention": convention, "latent_dtype": format!("{dtype:?}"), "rng_scheme": rng_scheme}),
        schedule,
        seed,
    );
}

/// One completed auxiliary decoder per output, in output order. This stage is distinct from the
/// primary image sampler and therefore does not inflate the number of completed image units.
pub fn sampled_decode(sampling: Value, schedule: &[f32], seed: u64, rng_scheme: &str) {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        let Some(observed) = current.as_mut() else {
            return;
        };
        match NumericScheduleDigest::f32_values(schedule) {
            Ok(digest) if !rng_scheme.trim().is_empty() => {
                observed
                    .decode
                    .push((sampling, digest, seed, rng_scheme.into()))
            }
            _ => {
                observed.unavailable = Some(reason(
                    "mlx_image_invalid_decoder_identity",
                    "decoder schedule or RNG scheme was not observed",
                    "decoder_sampling",
                ))
            }
        }
    });
}

pub fn choice(name: &str, value: ExecutionFact) {
    CURRENT.with(|current| {
        if let Some(observed) = current.borrow_mut().as_mut() {
            if observed.choices.get(name).is_some_and(|old| old != &value) {
                observed.unavailable = Some(reason(
                    "mlx_image_ambiguous_choice",
                    "multiple different resolved choices occurred in one image attempt",
                    name,
                ));
            }
            observed.choices.insert(name.into(), value);
        }
    });
}

/// Explicit refusal for an execution path whose complete facts are not observable at this seam.
pub fn unavailable(code: &str, detail: &str, missing: &str) {
    CURRENT.with(|current| {
        if let Some(observed) = current.borrow_mut().as_mut() {
            observed.unavailable = Some(reason(code, detail, missing));
        }
    });
}

fn reason(code: &str, detail: &str, missing: &str) -> ExecutionUnavailable {
    ExecutionUnavailable {
        code: code.into(),
        detail: detail.into(),
        missing_facts: vec![missing.into()],
    }
}

/// Record the actual factory selection, consumed schedule, latent dtype and seed at the sampler
/// boundary. Called only after that sampler succeeds. SCM records its consumed angle schedule.
pub fn sampled(sampling: Value, schedule: &[f32], seed: u64) {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        let Some(observed) = current.as_mut() else {
            return;
        };
        match NumericScheduleDigest::f32_values(schedule) {
            Ok(digest) => {
                let Some(scheme) = sampling
                    .get("rng_scheme")
                    .and_then(Value::as_str)
                    .filter(|v| !v.trim().is_empty())
                else {
                    observed.unavailable = Some(reason(
                        "mlx_image_rng_unobserved",
                        "sampler did not identify its actual RNG scheme",
                        "randomness",
                    ));
                    return;
                };
                let scheme = observed.binding.primary_rng.as_deref().unwrap_or(scheme);
                if scheme.trim().is_empty() {
                    observed.unavailable = Some(reason(
                        "mlx_image_rng_unobserved",
                        "primary RNG override was empty",
                        "randomness",
                    ));
                    return;
                }
                let scheme = scheme.to_owned();
                observed.schemes.push(scheme.clone());
                let output_index = observed
                    .binding
                    .output_index
                    .unwrap_or(observed.seeds.len());
                observed.output_indices.push(output_index);
                let mut sampling = sampling;
                sampling["rng_scheme"] = scheme.into();
                sampling["output_index"] = output_index.into();
                observed.sampling.push(sampling);
                observed.schedules.push(digest);
                observed.seeds.push(seed);
            }
            Err(_) => {
                observed.unavailable = Some(reason(
                    "mlx_image_invalid_schedule",
                    "the consumed schedule has no finite numeric identity",
                    "schedule",
                ));
            }
        }
    });
}

impl Observations {
    fn finish(
        mut self,
        descriptor: &ModelDescriptor,
        output: Option<&GenerationOutput>,
    ) -> crate::gen_core::Result<ExecutionIdentity> {
        if let Some(reason) = self.unavailable {
            return Ok(ExecutionIdentity::unavailable(reason));
        }
        let Some(GenerationOutput::Images(images)) = output else {
            return Ok(ExecutionIdentity::unavailable(reason(
                "mlx_image_output_unobserved",
                "image capture received a non-image output",
                "output_policy",
            )));
        };
        let covered: std::collections::BTreeSet<_> = self.output_indices.iter().copied().collect();
        if covered.len() != images.len()
            || covered.iter().any(|&index| index >= images.len())
            || self.policies.is_empty()
            || (!self.decode.is_empty() && self.decode.len() != images.len())
        {
            return Ok(ExecutionIdentity::unavailable(reason(
                "mlx_image_incomplete_observations",
                "completed image count does not match observed sampler invocations, or the resolved quality policy was not observed",
                "attempt_sampling_and_quality_policy",
            )));
        }
        let mut draws: Vec<_> = self
            .seeds
            .into_iter()
            .zip(self.output_indices)
            .map(|(seed, output_index)| ExecutionDraw {
                output_index,
                purpose: "initial_latent_and_sampler_step_stream".into(),
                seed: Some(seed),
                noise_sha256: None,
            })
            .collect();
        if !self.decode.is_empty() {
            let mut stages = Vec::with_capacity(self.decode.len());
            for (output_index, (sampling, schedule, seed, scheme)) in
                self.decode.into_iter().enumerate()
            {
                self.sampling.push(json!({"stage": "pid_decode", "output_index": output_index, "sampling": sampling, "rng_scheme": scheme}));
                self.schedules.push(schedule.clone());
                stages.push(json!({"output_index": output_index, "sampling": sampling, "schedule": schedule, "rng_scheme": scheme}));
                self.schemes.push(scheme);
                draws.push(ExecutionDraw {
                    output_index,
                    purpose: "pid_decode_initial_noise_and_epsilon_stream".into(),
                    seed: Some(seed),
                    noise_sha256: None,
                });
            }
            self.choices.insert(
                "auxiliary_decode".into(),
                ExecutionFact::resolved(json!(stages)),
            );
        }
        let schemes: std::collections::BTreeSet<_> = self.schemes.into_iter().collect();
        Ok(ExecutionIdentity::Resolved {
            identity: Box::new(ResolvedExecutionIdentity {
                version: EXECUTION_IDENTITY_VERSION,
                provider_id: descriptor.id.into(),
                backend: descriptor.backend.into(),
                runtime_id: format!(
                    "mlx-gen/{};image-execution-v1;{};{}",
                    env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH,
                ),
                sampling: ExecutionFact::resolved(json!(self.sampling)),
                schedule: ExecutionFact::resolved(json!(self.schedules)),
                estimator: ExecutionFact::not_applicable("image denoising has no separate estimator"),
                randomness: ExecutionRandomness::Draws {
                    scheme: serde_json::to_string(&schemes).expect("string sets serialize"),
                    draws,
                },
                deterministic_replay: ExecutionReplaySupport::NonDeterministic {
                    reason: "seeded draws are identified, but bitwise replay across MLX device/kernel executions is not attested".into(),
                },
                geometry: ExecutionFact::resolved(json!(images.iter().map(|image| json!({
                    "width": image.width, "height": image.height,
                })).collect::<Vec<_>>())),
                quality_policy: ExecutionFact::resolved(json!(self.policies)),
                output_policy: ExecutionFact::resolved(json!({"format": "rgb8", "channels": 3, "units": "ordered_images"})),
                provider_choices: self.choices,
                output_units: images.len(),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Capabilities, Image, Modality};

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "execution_probe",
            family: "probe",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        }
    }
    fn output() -> GenerationOutput {
        GenerationOutput::Images(vec![Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        }])
    }
    fn observe(seed: u64) {
        policy("decode", json!("untiled"));
        sampled_native("euler", "Sigma", mlx_rs::Dtype::Float32, &[1.0, 0.0], seed);
    }
    #[test]
    fn completed_attempt_owns_actual_seeds_and_schedule() {
        let completed = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            observe(u64::MAX);
            Ok(output())
        })
        .unwrap();
        let report = completed.execution.resolved().unwrap();
        assert_eq!(report.output_units, 1);
        assert!(!report.deterministic_replay.is_deterministic());
        let ExecutionRandomness::Draws { draws, .. } = &report.randomness else {
            panic!()
        };
        assert_eq!(draws[0].seed, Some(u64::MAX));
        assert_eq!(
            report.schedule,
            ExecutionFact::resolved(json!([
                NumericScheduleDigest::f32_values(&[1.0, 0.0]).unwrap()
            ]))
        );
    }
    #[test]
    fn failure_cancellation_retry_and_nested_calls_do_not_reuse_reports() {
        let cancel = CancelFlag::new();
        assert!(complete_image_attempt(&descriptor(), &cancel, || {
            observe(1);
            Err(crate::gen_core::Error::Canceled)
        })
        .is_err());
        let completed = complete_image_attempt(&descriptor(), &cancel, || Ok(output())).unwrap();
        assert!(completed.execution.resolved().is_none());
        let completed = complete_image_attempt(&descriptor(), &cancel, || {
            observe(2);
            without_capture(|| observe(3));
            let nested = complete_image_attempt(&descriptor(), &cancel, || {
                observe(4);
                Ok(output())
            })
            .unwrap();
            assert!(nested.execution.resolved().is_some());
            Ok(output())
        })
        .unwrap();
        let ExecutionRandomness::Draws { draws, .. } =
            &completed.execution.resolved().unwrap().randomness
        else {
            panic!()
        };
        assert_eq!(draws[0].seed, Some(2));
        assert!(complete_image_attempt(&descriptor(), &cancel, || {
            observe(5);
            cancel.cancel();
            Ok(output())
        })
        .is_err());
    }
    #[test]
    fn extra_sampling_or_unknown_policy_is_unavailable() {
        let completed = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            observe(1);
            observe(2);
            Ok(output())
        })
        .unwrap();
        assert!(completed.execution.resolved().is_none());
        let completed = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            observe(1);
            policy("decode", json!("tiled"));
            Ok(output())
        })
        .unwrap();
        assert!(
            matches!(completed.execution, ExecutionIdentity::Unavailable { reason } if reason.code == "mlx_image_ambiguous_policy")
        );
    }
    #[test]
    fn flow_factory_defaults_fallbacks_and_consumed_schedule_are_observed() {
        use crate::gen_core::sampling::TimestepConvention;
        for (name, expected) in [
            (None, "euler"),
            (Some("unknown"), "euler"),
            (Some("er_sde"), "er_sde"),
        ] {
            let report = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
                policy("decode", json!("native"));
                sampled_flow(
                    name,
                    TimestepConvention::Sigma,
                    mlx_rs::Dtype::Bfloat16,
                    &[0.5, 0.0],
                    17,
                );
                Ok(output())
            })
            .unwrap();
            let resolved = report.execution.resolved().unwrap();
            let ExecutionFact::Resolved { value } = &resolved.sampling else {
                panic!()
            };
            assert_eq!(value[0]["solver"], expected);
            assert_eq!(
                resolved.schedule,
                ExecutionFact::resolved(json!([
                    NumericScheduleDigest::f32_values(&[0.5, 0.0]).unwrap()
                ]))
            );
        }
    }

    #[test]
    fn decoder_must_cover_all_outputs_and_custom_rng_is_preserved() {
        let report = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            policy("decode", json!("test"));
            sampled_native_with_rng(
                "custom",
                "Sigma",
                mlx_rs::Dtype::Float32,
                &[1.0, 0.0],
                5,
                "custom-test-stream-v1",
            );
            sampled_decode(json!("decoder"), &[1.0, 0.0], 8, "decoder-test-stream-v1");
            Ok(output())
        })
        .unwrap();
        let ExecutionRandomness::Draws { draws, scheme } =
            &report.execution.resolved().unwrap().randomness
        else {
            panic!()
        };
        assert!(
            scheme.contains("custom-test-stream-v1") && scheme.contains("decoder-test-stream-v1")
        );
        assert_eq!(draws[1].seed, Some(8));
        let report = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            observe(1);
            observe(2);
            sampled_decode(json!("decoder"), &[1.0, 0.0], 1, "split");
            let GenerationOutput::Images(mut images) = output() else {
                panic!()
            };
            images.push(images[0].clone());
            Ok(GenerationOutput::Images(images))
        })
        .unwrap();
        assert!(report.execution.resolved().is_none());
    }

    #[test]
    fn unwinding_discards_the_attempt() {
        let _ = std::panic::catch_unwind(|| {
            let _ = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
                observe(4);
                panic!("simulated callback unwind")
            });
        });
        let report =
            complete_image_attempt(&descriptor(), &CancelFlag::new(), || Ok(output())).unwrap();
        assert!(report.execution.resolved().is_none());
    }
    #[test]
    fn explicit_output_stages_and_seed_scopes_restore_after_nested_calls() {
        let report = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            policy("decode", json!("native"));
            with_output_index(0, || {
                with_primary_seed(18, || {
                    with_primary_rng_scheme("global-initial-keyed-steps", || {
                        sampled_native_from_primary_seed(
                            "coarse",
                            "Sigma",
                            mlx_rs::Dtype::Float32,
                            &[1.0, 0.5],
                            "unused-default",
                        );
                    });
                    with_primary_seed(19, || {
                        sampled_native_from_primary_seed(
                            "refine",
                            "Sigma",
                            mlx_rs::Dtype::Float32,
                            &[0.5, 0.0],
                            "refine-keyed",
                        );
                    });
                    sampled_native_from_primary_seed(
                        "finish",
                        "Sigma",
                        mlx_rs::Dtype::Float32,
                        &[0.25, 0.0],
                        "finish-keyed",
                    );
                })
            });
            Ok(output())
        })
        .unwrap();
        let resolved = report.execution.resolved().unwrap();
        let ExecutionRandomness::Draws { draws, scheme } = &resolved.randomness else {
            panic!()
        };
        assert_eq!(
            draws
                .iter()
                .map(|draw| (draw.output_index, draw.seed))
                .collect::<Vec<_>>(),
            vec![(0, Some(18)), (0, Some(19)), (0, Some(18))]
        );
        assert!(
            scheme.contains("global-initial-keyed-steps")
                && scheme.contains("refine-keyed")
                && scheme.contains("finish-keyed")
        );
        let report = complete_image_attempt(&descriptor(), &CancelFlag::new(), || {
            policy("decode", json!("native"));
            sampled_native_from_primary_seed(
                "test",
                "Sigma",
                mlx_rs::Dtype::Float32,
                &[1.0, 0.0],
                "test",
            );
            Ok(output())
        })
        .unwrap();
        assert!(
            matches!(report.execution, ExecutionIdentity::Unavailable { reason } if reason.code == "mlx_image_initial_seed_unobserved")
        );
    }

    #[test]
    fn rich_report_warnings_survive_execution_capture() {
        let warning = crate::gen_core::GenerationWarning {
            code: "actual_warning".into(),
            message: "provider output warning".into(),
        };
        let completed = complete_image_report_attempt(&descriptor(), &CancelFlag::new(), || {
            observe(1);
            let mut report = crate::gen_core::GenerationReport::from_output(output());
            report.warnings.push(warning.clone());
            Ok(report)
        })
        .unwrap();
        assert_eq!(completed.generation.warnings, vec![warning]);
        assert!(completed.execution.resolved().is_some());
    }
}
