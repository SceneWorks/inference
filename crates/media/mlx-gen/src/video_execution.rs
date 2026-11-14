//! Synchronous, attempt-local observations from video and joint-AV execution seams.
//!
//! Providers record values where they consume them, never reconstruct them from a request.
//! The scoped collector is discarded on errors, cancellation and unwinding. User callbacks run
//! with capture suspended, so nested renders cannot contaminate the enclosing attempt.
use gen_core::generator::{
    CompletedGeneration, ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness,
    ExecutionReplaySupport, ExecutionUnavailable, GenerationOutput, GenerationReport,
    ModelDescriptor, NumericScheduleDigest, ResolvedExecutionIdentity, EXECUTION_IDENTITY_VERSION,
};
use gen_core::CancelFlag;
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

#[derive(Default)]
struct Observations {
    sampling: Vec<Value>,
    schedules: Vec<Value>,
    policy: Vec<Value>,
    draws: Vec<ExecutionDraw>,
    missing: Vec<String>,
    output_index: usize,
    devices: BTreeSet<String>,
}
thread_local! {
    static ACTIVE: RefCell<Option<Observations>> = const { RefCell::new(None) };
}
struct Scope(Option<Observations>);
impl Scope {
    fn enter(value: Option<Observations>) -> Self {
        Self(ACTIVE.with(|active| active.replace(value)))
    }
    fn take(&self) -> Observations {
        ACTIVE.with(|active| active.borrow_mut().take().unwrap_or_default())
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.0.take());
        });
    }
}
fn observe(f: impl FnOnce(&mut Observations)) {
    ACTIVE.with(|active| {
        if let Some(state) = active.borrow_mut().as_mut() {
            f(state);
        }
    });
}
/// Isolate user code or an unrelated nested generation from the active attempt.
pub fn without_capture<T>(body: impl FnOnce() -> T) -> T {
    let _scope = Scope::enter(None);
    body()
}
/// Record the actual solver and estimator after resolution, with effective controls.
pub fn sampling(stage: &str, solver: &str, estimator: &str, controls: Value) {
    observe(|state| {
        match mlx_rs::Stream::task_local_or_default()
            .get_device()
            .and_then(|device| device.get_type())
        {
            Ok(kind) => {
                state.devices.insert(format!("{kind:?}"));
            }
            Err(_) => state
                .missing
                .push(format!("{stage}: selected MLX stream device")),
        }
    });
    let solver = gen_core::sampling::Solver::from_name(solver)
        .map(|resolved| format!("{resolved:?}"))
        .unwrap_or_else(|| solver.to_owned());
    observe(|state| {
        state.sampling.push(json!({"stage":stage,"solver":solver,
        "estimator":estimator,"controls":controls}))
    });
}
/// Record the canonical solver including the curated runner's actual unknown-name fallback.
pub fn curated_sampling(stage: &str, name: &str, estimator: &str, controls: Value) {
    let selected = crate::sampler::canonical_solver(Some(name), gen_core::sampling::Solver::Euler);
    sampling(stage, selected.name(), estimator, controls);
}
/// Record an exact numeric schedule consumed by a runtime stage.
pub fn schedule(stage: &str, values: &[f32]) {
    observe(|state| match NumericScheduleDigest::f32_values(values) {
        Ok(digest) => state.schedules.push(json!({"stage":stage,"digest":digest})),
        Err(_) => state
            .missing
            .push(format!("{stage}: finite nonempty schedule")),
    });
}
/// Preserve f64 schedules without rounding them to f32 (Krea's release schedule).
pub fn schedule_f64(stage: &str, values: &[f64]) {
    observe(|state| match NumericScheduleDigest::f64_values(values) {
        Ok(digest) => state.schedules.push(json!({"stage":stage,"digest":digest})),
        Err(_) => state
            .missing
            .push(format!("{stage}: finite nonempty schedule")),
    });
}
/// Record the actual memory, precision, chunking or tiling choice at its use site.
pub fn policy(stage: &str, value: Value) {
    observe(|state| state.policy.push(json!({"stage":stage,"value":value})));
}
/// Record an ordered draw. Split-key streams name their split/stream policy in `purpose`.
pub fn draw(output_index: usize, purpose: &str, seed: u64) {
    observe(|state| {
        state.draws.push(ExecutionDraw {
            output_index,
            purpose: purpose.into(),
            seed: Some(seed),
            noise_sha256: None,
        })
    });
}
/// Bind the actual install outcome to this attempt. Current adapter loaders expose target counts
/// but not a digest of the bytes they consumed. A path (or a later hash of that path) cannot attest
/// those bytes, so adapter-bearing output remains explicitly unavailable for exact identity.
pub fn adapter_stack(stage: &str, outcomes: Value, applied_targets: usize) {
    policy(stage, outcomes);
    if applied_targets > 0 {
        unavailable(&format!(
            "{stage}: consumed adapter content digest unavailable"
        ));
    }
}

/// Select the output of a provider-owned multi-image loop.
pub fn output_index(index: usize) {
    observe(|state| state.output_index = index);
}
/// Record a draw for the provider-owned output currently being produced.
pub fn draw_current(purpose: &str, seed: u64) {
    observe(|state| {
        state.draws.push(ExecutionDraw {
            output_index: state.output_index,
            purpose: purpose.into(),
            seed: Some(seed),
            noise_sha256: None,
        })
    });
}
/// Bind an engine-resolved text choice without copying prompt contents into the report.
pub fn text_digest(stage: &str, text: &str) {
    use sha2::{Digest, Sha256};
    observe(|state| {
        state.policy.push(json!({"stage":stage,
        "utf8_sha256":format!("{:x}",Sha256::digest(text.as_bytes()))}))
    });
}
/// Explicit source-backed limitation; never turn an unobserved fact into a resolved receipt.
pub fn unavailable(fact: &str) {
    observe(|state| state.missing.push(fact.into()));
}
/// Complete a provider invocation with only observations belonging to this successful attempt.
pub fn complete(
    descriptor: &ModelDescriptor,
    cancel: &CancelFlag,
    body: impl FnOnce() -> gen_core::Result<GenerationReport>,
) -> gen_core::Result<CompletedGeneration> {
    let scope = Scope::enter(Some(Observations::default()));
    let generation = body()?;
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let mut seen = scope.take();
    let (units, geometry, output) = match generation.output.as_ref() {
        Some(GenerationOutput::Video { frames, fps, audio }) => (
            1,
            json!({"frames":frames.iter().map(|f| [f.width,f.height]).collect::<Vec<_>>()}),
            json!({"format":"rgb8_video","fps":fps,"frames":frames.len(),
                "audio":audio.as_ref().map(|a| json!({"channels":a.channels,"sample_rate":a.sample_rate,"samples":a.samples.len()}))}),
        ),
        Some(GenerationOutput::Images(images)) => (
            images.len(),
            json!({"images":images.iter().map(|f| [f.width,f.height]).collect::<Vec<_>>()}),
            json!({"format":"rgb8_images","count":images.len()}),
        ),
        _ => {
            seen.missing.push("video/image output policy".into());
            (1, Value::Null, Value::Null)
        }
    };
    if seen.sampling.is_empty() {
        seen.missing.push("runtime sampler and estimator".into());
    }
    if seen.schedules.is_empty() {
        seen.missing.push("runtime numeric schedule".into());
    }
    if seen.policy.is_empty() {
        seen.missing.push("runtime quality policy".into());
    }
    if seen.draws.is_empty() {
        seen.missing.push("runtime random draws".into());
    }
    let execution = if seen.missing.is_empty() {
        ExecutionIdentity::Resolved {
            identity: Box::new(ResolvedExecutionIdentity {
                version: EXECUTION_IDENTITY_VERSION,
                provider_id: descriptor.id.into(),
                backend: descriptor.backend.into(),
                runtime_id: format!(
                    "{}@{}:mlx-video-execution-v1:{:?}",
                    descriptor.id,
                    env!("CARGO_PKG_VERSION"),
                    seen.devices
                ),
                sampling: ExecutionFact::resolved(json!(seen.sampling)),
                schedule: ExecutionFact::resolved(json!(seen.schedules)),
                estimator: ExecutionFact::resolved(json!(seen
                    .sampling
                    .iter()
                    .map(|stage| json!({"stage":stage["stage"],"estimator":stage["estimator"]}))
                    .collect::<Vec<_>>())),
                randomness: ExecutionRandomness::Draws {
                    scheme: "mlx-video-ordered-draws-v1".into(),
                    draws: seen.draws,
                },
                deterministic_replay: ExecutionReplaySupport::NonDeterministic {
                    reason:
                        "Bitwise replay on the recorded MLX device and pinned runtime has not been verified"
                            .into(),
                },
                geometry: ExecutionFact::resolved(geometry),
                quality_policy: ExecutionFact::resolved(json!(seen.policy)),
                output_policy: ExecutionFact::resolved(output),
                provider_choices: BTreeMap::new(),
                output_units: units,
            }),
        }
    } else {
        ExecutionIdentity::unavailable(ExecutionUnavailable {
            code: "video_runtime_observation_unavailable".into(),
            detail: format!(
                "{} execution did not expose all required runtime facts",
                descriptor.id
            ),
            missing_facts: seen.missing,
        })
    };
    CompletedGeneration::new(generation, execution, descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "video_execution_test",
            backend: "mlx",
            family: "test",
            modality: gen_core::Modality::Video,
            capabilities: gen_core::Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        }
    }
    fn output() -> GenerationReport {
        GenerationReport::from_output(GenerationOutput::Video {
            frames: vec![gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![1, 2, 3],
            }],
            fps: 24,
            audio: None,
        })
    }
    fn facts(seed: u64) {
        // Tensor-free observations: the fixture deliberately bypasses the device lookup used by
        // the production sampling seam; this tests bookkeeping without opening an MLX stream.
        observe(|state| {
            state
                .sampling
                .push(json!({"stage":"test","solver":"euler","estimator":"velocity"}));
            state.devices.insert("fixture-device".into());
        });
        schedule("video", &[1.0, 0.5, 0.0]);
        policy("decode", json!({"chunk_frames":8}));
        draw(0, "video", seed);
        draw(0, "audio", seed.wrapping_add(1));
    }
    #[test]
    fn completed_output_binds_ordered_video_and_audio_draws_and_numeric_schedule() {
        let completed = complete(&descriptor(), &CancelFlag::default(), || {
            facts(u64::MAX);
            Ok(output())
        })
        .unwrap();
        let identity = completed.execution.resolved().unwrap();
        let ExecutionRandomness::Draws { draws, .. } = &identity.randomness else {
            panic!()
        };
        assert_eq!(
            draws.iter().map(|draw| draw.seed).collect::<Vec<_>>(),
            [Some(u64::MAX), Some(0)]
        );
        assert_eq!(draws[0].purpose, "video");
        assert_eq!(draws[1].purpose, "audio");
        assert_eq!(
            identity.schedule,
            ExecutionFact::resolved(json!([{"stage":"video",
            "digest":NumericScheduleDigest::f32_values(&[1.0,0.5,0.0]).unwrap()}]))
        );
        assert_eq!(identity.output_units, 1);
    }
    #[test]
    fn missing_or_nonfinite_runtime_facts_are_typed_unavailable() {
        for invalid_schedule in [false, true] {
            let completed = complete(&descriptor(), &CancelFlag::default(), || {
                if invalid_schedule {
                    facts(5);
                    schedule("broken", &[f32::NAN]);
                }
                Ok(output())
            })
            .unwrap();
            let ExecutionIdentity::Unavailable { reason } = completed.execution else {
                panic!()
            };
            assert_eq!(reason.code, "video_runtime_observation_unavailable");
            assert!(!reason.missing_facts.is_empty());
        }
    }
    #[test]
    fn failed_and_cancelled_attempts_cannot_publish_or_leak_prior_facts() {
        let error = complete(&descriptor(), &CancelFlag::default(), || {
            facts(5);
            Err(gen_core::Error::Msg("failed decode".into()))
        });
        assert!(error.is_err());
        let cancel = CancelFlag::default();
        let cancelled = complete(&descriptor(), &cancel, || {
            facts(6);
            cancel.cancel();
            Ok(output())
        });
        assert!(matches!(cancelled, Err(gen_core::Error::Canceled)));
        let retry = complete(&descriptor(), &CancelFlag::default(), || Ok(output())).unwrap();
        assert!(matches!(
            retry.execution,
            ExecutionIdentity::Unavailable { .. }
        ));
    }
    #[test]
    fn callbacks_nested_attempts_and_unwinding_restore_the_parent_scope() {
        let completed = complete(&descriptor(), &CancelFlag::default(), || {
            facts(7);
            without_capture(|| {
                facts(999);
                let nested = complete(&descriptor(), &CancelFlag::default(), || {
                    facts(12);
                    Ok(output())
                })
                .unwrap();
                assert!(nested.execution.resolved().is_some());
            });
            let _ = std::panic::catch_unwind(|| {
                let _ = complete(&descriptor(), &CancelFlag::default(), || {
                    facts(888);
                    panic!("fixture callback panic")
                });
            });
            Ok(output())
        })
        .unwrap();
        let ExecutionRandomness::Draws { draws, .. } =
            &completed.execution.resolved().unwrap().randomness
        else {
            panic!()
        };
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].seed, Some(7));
    }
    #[test]
    fn seedvr2_image_batches_keep_draws_on_the_actual_output() {
        let result = complete(&descriptor(), &CancelFlag::default(), || {
            facts(7);
            output_index(1);
            draw_current("tile0", 8);
            draw_current("tile1", 8);
            Ok(GenerationReport::from_output(GenerationOutput::Images(
                vec![
                    gen_core::Image {
                        width: 1,
                        height: 1,
                        pixels: vec![0; 3]
                    };
                    2
                ],
            )))
        })
        .unwrap();
        let identity = result.execution.resolved().unwrap();
        assert_eq!(identity.output_units, 2);
        let ExecutionRandomness::Draws { draws, .. } = &identity.randomness else {
            panic!()
        };
        assert_eq!(
            draws
                .iter()
                .map(|draw| draw.output_index)
                .collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
    }
    #[test]
    fn f64_schedule_identity_preserves_values_that_f32_would_collapse() {
        let identity = |middle| {
            complete(&descriptor(), &CancelFlag::default(), || {
                facts(7);
                schedule_f64("release_schedule", &[1.0, middle, 0.0]);
                Ok(output())
            })
            .unwrap()
            .execution
        };
        let a = 0.5_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(a as f32, b as f32);
        assert_ne!(
            identity(a).resolved().unwrap().schedule,
            identity(b).resolved().unwrap().schedule
        );
    }
    #[test]
    fn curated_defaults_and_unknown_names_use_the_same_resolver_as_execution() {
        use gen_core::sampling::Solver;
        assert_eq!(
            crate::sampler::canonical_solver(None, Solver::Euler),
            Solver::Euler
        );
        assert_eq!(
            crate::sampler::canonical_solver(Some("unknown"), Solver::Euler),
            Solver::Euler
        );
        assert_eq!(
            crate::sampler::canonical_solver(Some("heun"), Solver::Euler),
            Solver::Heun
        );
    }
    #[test]
    fn applied_adapter_content_is_unavailable_and_cannot_leak_to_next_attempt() {
        for applied in [0, 2, 0] {
            let completed = complete(&descriptor(), &CancelFlag::default(), || {
                facts(7);
                adapter_stack("provider_install", json!({"applied":applied}), applied);
                Ok(output())
            })
            .unwrap();
            if applied == 0 {
                assert!(completed.execution.resolved().is_some());
            } else {
                let ExecutionIdentity::Unavailable { reason } = completed.execution else {
                    panic!()
                };
                assert_eq!(
                    reason.missing_facts,
                    ["provider_install: consumed adapter content digest unavailable"]
                );
            }
        }
    }

    #[test]
    fn every_video_provider_overrides_the_external_compatibility_default() {
        for (source, expected) in [
            (include_str!("../mlx-gen-minimax-h3/src/model.rs"), 1),
            (include_str!("../mlx-gen-ltx/src/model/generator.rs"), 2),
            (include_str!("../mlx-gen-wan/src/model/ti2v.rs"), 1),
            (include_str!("../mlx-gen-wan/src/model/a14b_generate.rs"), 1),
            (include_str!("../mlx-gen-wan/src/model_vace.rs"), 2),
            (include_str!("../mlx-gen-svd/src/model.rs"), 1),
            (include_str!("../mlx-gen-scail2/src/pipeline.rs"), 1),
            (include_str!("../mlx-gen-krea-realtime/src/pipeline.rs"), 1),
            (include_str!("../mlx-gen-bernini/src/pipeline.rs"), 1),
            (include_str!("../mlx-gen-bernini/src/bernini.rs"), 1),
            (include_str!("../mlx-gen-mochi/src/model.rs"), 1),
            (include_str!("../mlx-gen-seedvr2/src/registry.rs"), 1),
        ] {
            assert_eq!(
                source.matches("fn generate_with_execution_report(").count(),
                expected
            );
            assert_eq!(
                source.matches("video_execution::complete(").count(),
                expected
            );
        }
    }
}
