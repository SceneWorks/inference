//! Observations from the execution that produced an output, never from submitted options.
//!
//! Candle providers execute synchronously. A stack scoped to that calling thread lets the shared
//! sampler and decoder report their resolved decisions without changing the public request ABI.
//! Each invocation owns a fresh frame; nested generation, errors and unwinding cannot reuse it.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    marker::PhantomData,
    rc::Rc,
};

use candle_core::{DType, Device};
use gen_core::generator::{
    CompletedGeneration, ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness,
    ExecutionReplaySupport, ExecutionUnavailable, GenerationOutput, GenerationRequest, Generator,
    NumericScheduleDigest, ResolvedExecutionIdentity, EXECUTION_IDENTITY_VERSION,
};
use gen_core::Progress;
use serde_json::{json, Value};

#[derive(Default)]
struct Observations {
    non_deterministic: Option<String>,
    rng_schemes: Vec<String>,
    current_output: usize,
    draws: Vec<ExecutionDraw>,
    schedules: Vec<Value>,
    policies: BTreeMap<String, Value>,
    runtimes: Vec<Value>,
    missing: Vec<String>,
}

struct Frame {
    id: u64,
    observations: Observations,
}

thread_local! {
    static NEXT_ATTEMPT: Cell<u64> = const { Cell::new(0) };
    static ATTEMPTS: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
    static SUSPENDED_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub struct Attempt {
    id: u64,
    _thread: PhantomData<Rc<()>>,
    active: bool,
}
impl Attempt {
    pub fn begin() -> Self {
        let id = NEXT_ATTEMPT.get();
        NEXT_ATTEMPT.set(id.wrapping_add(1));
        ATTEMPTS.with_borrow_mut(|stack| {
            stack.push(Frame {
                id,
                observations: Observations::default(),
            })
        });
        Self {
            id,
            active: true,
            _thread: PhantomData,
        }
    }
    fn finish(mut self) -> Observations {
        let result = ATTEMPTS.with_borrow_mut(|stack| {
            let index = stack
                .iter()
                .position(|frame| frame.id == self.id)
                .expect("live attempt");
            stack.remove(index).observations
        });
        // Drop must not pop the enclosing attempt.
        self.disarm();
        result
    }
    fn disarm(&mut self) {
        self.active = false;
    }
}
impl Drop for Attempt {
    fn drop(&mut self) {
        if self.active {
            ATTEMPTS.with_borrow_mut(|stack| {
                stack.retain(|frame| frame.id != self.id);
            });
        }
    }
}

fn observe(f: impl FnOnce(&mut Observations)) {
    ATTEMPTS.with_borrow_mut(|stack| {
        if stack.len() <= SUSPENDED_DEPTH.get() {
            return;
        }
        if let Some(attempt) = stack.last_mut() {
            f(&mut attempt.observations);
        }
    });
}

/// Select the output currently being decoded after a staged batch denoise.
pub fn output_index(index: usize) {
    observe(|attempt| attempt.current_output = index);
}

/// Report a source-backed limitation of exact replay despite complete observed execution facts.
pub fn non_deterministic(reason: &str) {
    observe(|attempt| attempt.non_deterministic = Some(reason.into()));
}

/// Record an actual RNG implementation used by a stream.
pub fn rng_scheme(scheme: &str) {
    observe(|attempt| {
        if !attempt.rng_schemes.iter().any(|known| known == scheme) {
            attempt.rng_schemes.push(scheme.into());
        }
    });
}

/// Observe an actual seed constructor without changing its value or RNG implementation.
pub fn seed(purpose: &str, seed: u64) -> u64 {
    stream(purpose, seed);
    seed
}

/// Record the seed actually handed to a random stream, in completed output order.
pub fn draw(purpose: &str, seed: u64, output_index: usize) {
    observe(|attempt| {
        attempt.current_output = output_index;
        attempt.draws.push(ExecutionDraw {
            output_index,
            purpose: purpose.into(),
            seed: Some(seed),
            noise_sha256: None,
        });
    });
}

/// Bind explicitly supplied noise to its output without inventing a seed.
pub fn noise(purpose: &str, values: &[f32], output_index: usize) {
    observe(|attempt| match NumericScheduleDigest::f32_values(values) {
        Ok(digest) => {
            attempt.current_output = output_index;
            attempt.draws.push(ExecutionDraw {
                output_index,
                purpose: purpose.into(),
                seed: None,
                noise_sha256: Some(digest.sha256),
            });
        }
        Err(error) => attempt.missing.push(error.to_string()),
    });
}

/// Record an additional stream for the currently executing output.
pub fn stream(purpose: &str, seed: u64) {
    observe(|attempt| {
        attempt.draws.push(ExecutionDraw {
            output_index: attempt.current_output,
            purpose: purpose.into(),
            seed: Some(seed),
            noise_sha256: None,
        })
    });
}

/// Record the exact schedule consumed by a selected sampler, after default/fallback resolution.
pub fn schedule(sampler: &str, values: &[f32], estimator: &str) {
    record_schedule(
        sampler,
        NumericScheduleDigest::f32_values(values),
        estimator,
    );
}

/// Record an executed operation which has no diffusion schedule, for example a source roundtrip.
pub fn without_schedule(operation: &str, reason: &str, estimator: &str) {
    observe(|attempt| {
        attempt.schedules.push(json!({
        "sampler": operation, "schedule": {"state":"not_applicable","reason":reason}, "estimator": estimator,
    }))
    });
}

/// The f64 form preserves native schedules without an f32 round trip.
pub fn schedule_f64(sampler: &str, values: &[f64], estimator: &str) {
    record_schedule(
        sampler,
        NumericScheduleDigest::f64_values(values),
        estimator,
    );
}

fn record_schedule(
    sampler: &str,
    digest: gen_core::core_llm::Result<NumericScheduleDigest>,
    estimator: &str,
) {
    observe(|attempt| match digest {
        Ok(digest) => attempt.schedules.push(json!({
            "sampler": sampler, "schedule": digest, "estimator": estimator,
        })),
        Err(error) => attempt.missing.push(error.to_string()),
    });
}

/// Record a resolved quality/memory/decoder decision at the operation consuming it.
/// Repeated stages retain their order rather than overwriting an earlier decision.
pub fn policy(key: &str, value: Value) {
    observe(|attempt| {
        let values = attempt
            .policies
            .entry(key.into())
            .or_insert_with(|| json!([]));
        let values = values.as_array_mut().expect("policy list");
        if let Some(last) = values.last_mut().filter(|last| last["value"] == value) {
            last["repetitions"] = json!(last["repetitions"].as_u64().unwrap_or(1) + 1);
        } else {
            values.push(json!({"value": value, "repetitions": 1}));
        }
    });
}

/// Record actual compute storage, not the requested or default device/precision.
pub fn runtime(device: &Device, dtype: DType) {
    observe(|attempt| {
        let value =
            json!({"device": format!("{:?}", device.location()), "dtype": format!("{dtype:?}")});
        if attempt.runtimes.last() != Some(&value) {
            attempt.runtimes.push(value);
        }
    });
}

/// A route with an unobservable decision must explicitly refuse exact replay.
pub fn unavailable(detail: &str) {
    observe(|attempt| attempt.missing.push(detail.into()));
}

/// Invoke a consumer callback without treating its work as this provider's execution.
/// Nested completed-report calls start a separate active frame and remain observable.
pub fn outside_attempt<T>(body: impl FnOnce() -> T) -> T {
    let depth = ATTEMPTS.with_borrow(|stack| stack.len());
    struct Resume(usize);
    impl Drop for Resume {
        fn drop(&mut self) {
            SUSPENDED_DEPTH.set(self.0);
        }
    }
    let _resume = Resume(SUSPENDED_DEPTH.replace(depth));
    body()
}

/// Run a standalone image provider entry point with the same attempt-local contract.
pub fn capture_direct(
    descriptor: &gen_core::ModelDescriptor,
    cancel: &gen_core::CancelFlag,
    body: impl FnOnce() -> gen_core::Result<GenerationOutput>,
) -> gen_core::Result<CompletedGeneration> {
    let attempt = Attempt::begin();
    let output = body()?;
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let execution = attempt.finish_image(descriptor, &output);
    CompletedGeneration::new(
        gen_core::GenerationReport::from_output(output),
        execution,
        descriptor,
    )
}

/// Run one provider attempt and bind only observations from that invocation to its output.
/// The provider overrides `generate_with_execution_report` with this helper. Its normal/direct
/// implementation remains the execution owner. The trait's streaming report forwards here too.
pub fn capture<G: Generator + ?Sized>(
    generator: &G,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
) -> gen_core::Result<CompletedGeneration> {
    capture_with(generator, req, on_progress, || {})
}

/// Capture provider state selected when the model was loaded, before executing the attempt.
pub fn capture_with<G: Generator + ?Sized>(
    generator: &G,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
    loaded_policy: impl FnOnce(),
) -> gen_core::Result<CompletedGeneration> {
    let attempt = Attempt::begin();
    loaded_policy();
    let generation = generator.generate_with_report(req, &mut |progress| {
        outside_attempt(|| on_progress(progress));
    })?;
    if req.cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let observations = attempt.finish();
    let images = output_facts(generation.output.as_ref());
    let execution = observations.identity(generator.descriptor(), &images);
    CompletedGeneration::new(generation, execution, generator.descriptor())
}

fn output_facts(output: Option<&GenerationOutput>) -> Vec<Value> {
    match output {
        Some(GenerationOutput::Images(images)) => images
            .iter()
            .map(|image| {
                json!({
                    "width":image.width, "height":image.height, "channels":3, "format":"rgb8"
                })
            })
            .collect(),
        Some(GenerationOutput::ImagesRgba(images)) => images
            .iter()
            .map(|image| {
                json!({
                    "width":image.width, "height":image.height, "channels":4, "format":"rgba8"
                })
            })
            .collect(),
        Some(GenerationOutput::Video { frames, fps, audio }) => vec![json!({
            "format":"rgb8-video", "fps":fps,
            "frames":frames.iter().map(|frame| json!({"width":frame.width,"height":frame.height})).collect::<Vec<_>>(),
            "audio":audio.as_ref().map(|track| json!({"sample_rate":track.sample_rate,"channels":track.channels,"samples":track.samples.len()})),
        })],
        Some(GenerationOutput::Audio(track)) => vec![
            json!({"format":"f32-audio","sample_rate":track.sample_rate,"channels":track.channels,"samples":track.samples.len()}),
        ],
        None => Vec::new(),
    }
}

impl Attempt {
    /// Bind direct image/video execution to its actual output, using the same completed schema.
    pub fn finish_image(
        self,
        descriptor: &gen_core::ModelDescriptor,
        output: &GenerationOutput,
    ) -> ExecutionIdentity {
        self.finish()
            .identity(descriptor, &output_facts(Some(output)))
    }
}

impl Observations {
    fn identity(
        mut self,
        descriptor: &gen_core::ModelDescriptor,
        images: &[Value],
    ) -> ExecutionIdentity {
        if self.schedules.is_empty() {
            self.missing.push("executed sampler/schedule".into());
        }
        if self.draws.is_empty() {
            self.missing.push("actual output random streams".into());
        }
        if self.rng_schemes.is_empty() {
            self.missing.push("actual RNG algorithm".into());
        }
        if self.policies.is_empty() {
            self.missing.push("resolved quality policy".into());
        }
        if self.runtimes.is_empty() {
            self.missing.push("actual runtime device/precision".into());
        }
        if images.is_empty() {
            self.missing.push("image output geometry/format".into());
        }
        if !self.missing.is_empty() {
            return ExecutionIdentity::Unavailable {
                reason: ExecutionUnavailable {
                    code: "candle_execution_observation_incomplete".into(),
                    detail: format!(
                        "{} completed an execution with unobserved facts: {}",
                        descriptor.id,
                        self.missing.join(", ")
                    ),
                    missing_facts: self.missing,
                },
            };
        }
        let identity = ResolvedExecutionIdentity {
            version: EXECUTION_IDENTITY_VERSION,
            provider_id: descriptor.id.into(),
            backend: descriptor.backend.into(),
            runtime_id: "candle-execution-v1".into(),
            sampling: ExecutionFact::resolved(json!(self.schedules.iter().map(|s| &s["sampler"]).collect::<Vec<_>>())),
            schedule: ExecutionFact::resolved(json!(self.schedules)),
            estimator: ExecutionFact::resolved(json!(self.schedules.iter().map(|s| &s["estimator"]).collect::<Vec<_>>())),
            randomness: ExecutionRandomness::Draws {
                scheme: format!("candle-ordered-provider-streams-v1:{}", self.rng_schemes.join(";")), draws: self.draws,
            },
            deterministic_replay: match self.non_deterministic {
                Some(reason) => ExecutionReplaySupport::NonDeterministic { reason },
                None => ExecutionReplaySupport::Deterministic {
                    scope: "same inference revision, model/component bytes, device, kernels and resolved policies".into(),
                },
            },
            geometry: ExecutionFact::resolved(json!(images)),
            quality_policy: ExecutionFact::resolved(json!({"operations":self.policies,"compute":self.runtimes})),
            output_policy: ExecutionFact::resolved(json!(images)),
            provider_choices: BTreeMap::new(),
            output_units: images.len(),
        };
        match identity.validate(descriptor.id, descriptor.backend, images.len()) {
            Ok(()) => ExecutionIdentity::Resolved {
                identity: Box::new(identity),
            },
            Err(error) => ExecutionIdentity::Unavailable {
                reason: ExecutionUnavailable {
                    code: "candle_execution_observation_incomplete".into(),
                    detail: error.to_string(),
                    missing_facts: vec!["complete ordered output binding".into()],
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{Capabilities, Image, Modality, ModelDescriptor};
    use std::sync::LazyLock;

    static DESCRIPTOR: LazyLock<ModelDescriptor> = LazyLock::new(|| ModelDescriptor {
        id: "execution-fixture",
        family: "fixture",
        backend: "candle",
        modality: Modality::Image,
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        capabilities: Capabilities::default(),
    });

    struct Fixture;
    impl Generator for Fixture {
        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn descriptor(&self) -> &ModelDescriptor {
            &DESCRIPTOR
        }
        fn generate(
            &self,
            req: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            // This is a raw arithmetic fixture, never a learned model forward.
            rng_scheme("fixture-seeded-raw-arithmetic-v1");
            let images = crate::for_each_image_seed(req.seed.unwrap_or(91), req.count, |seed| {
                let tensor = candle_core::Tensor::zeros((1,), DType::F32, &Device::Cpu)
                    .map_err(crate::CandleError::from)?;
                policy(
                    "actual_decoder",
                    json!({"mode":"single_pass","precision":"f32"}),
                );
                crate::run_flow_sampler(
                    req.sampler.as_deref(),
                    gen_core::sampling::TimestepConvention::Sigma,
                    &[1.0, 0.25, 0.0],
                    tensor,
                    seed,
                    &req.cancel,
                    progress,
                    None,
                    |x, _| Ok(x.zeros_like()?),
                )?;
                Ok::<_, crate::CandleError>(Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                })
            })?;
            Ok(GenerationOutput::Images(images))
        }
        fn generate_with_execution_report(
            &self,
            req: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<CompletedGeneration> {
            capture(self, req, progress)
        }
    }

    fn resolved(completed: CompletedGeneration) -> Box<ResolvedExecutionIdentity> {
        match completed.execution {
            ExecutionIdentity::Resolved { identity } => identity,
            other => panic!("expected resolved execution: {other:?}"),
        }
    }

    #[test]
    fn observes_actual_fallback_schedule_and_wrapping_output_seeds() {
        let req = GenerationRequest {
            count: 2,
            seed: Some(u64::MAX),
            sampler: Some("native-unknown-alias".into()),
            ..Default::default()
        };
        let identity = resolved(
            Fixture
                .generate_with_execution_report(&req, &mut |_| {})
                .unwrap(),
        );
        assert_eq!(
            identity.sampling,
            ExecutionFact::resolved(json!(["euler", "euler"]))
        );
        let ExecutionRandomness::Draws { draws, .. } = identity.randomness else {
            panic!("draws")
        };
        assert_eq!(
            draws
                .iter()
                .map(|draw| draw.seed.unwrap())
                .collect::<Vec<_>>(),
            [u64::MAX, 0]
        );
        assert_eq!(
            draws
                .iter()
                .map(|draw| draw.output_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let ExecutionFact::Resolved { value } = identity.schedule else {
            panic!("schedule")
        };
        assert_eq!(
            value[0]["schedule"],
            json!(NumericScheduleDigest::f32_values(&[1.0, 0.25, 0.0]).unwrap())
        );
    }

    #[test]
    fn omitted_seed_reports_provider_draw_and_streaming_uses_same_attempt() {
        let req = GenerationRequest {
            sampler: Some("heun".into()),
            ..Default::default()
        };
        let identity = resolved(
            Fixture
                .generate_streaming_with_execution_report(
                    &req,
                    &mut |_| panic!("image has no audio chunks"),
                    &mut |_| {},
                )
                .unwrap(),
        );
        assert_eq!(identity.sampling, ExecutionFact::resolved(json!(["heun"])));
        let ExecutionRandomness::Draws { draws, .. } = identity.randomness else {
            panic!("draws")
        };
        assert_eq!(draws[0].seed, Some(91));
    }

    #[test]
    fn callbacks_cannot_contaminate_outer_execution_and_nested_attempts_are_isolated() {
        let outer = resolved(
            Fixture
                .generate_with_execution_report(&GenerationRequest::default(), &mut |_| {
                    draw("foreign_callback", 100, 99);
                    let inner = resolved(
                        Fixture
                            .generate_with_execution_report(
                                &GenerationRequest {
                                    seed: Some(7),
                                    ..Default::default()
                                },
                                &mut |_| {},
                            )
                            .unwrap(),
                    );
                    let ExecutionRandomness::Draws { draws, .. } = inner.randomness else {
                        panic!("draws")
                    };
                    assert_eq!(draws[0].seed, Some(7));
                })
                .unwrap(),
        );
        let ExecutionRandomness::Draws { draws, .. } = outer.randomness else {
            panic!("draws")
        };
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].seed, Some(91));
    }

    #[test]
    fn cancellation_failure_and_panic_never_publish_stale_facts() {
        let req = GenerationRequest::default();
        let result = Fixture.generate_with_execution_report(&req, &mut |_| {
            req.cancel.cancel();
        });
        assert!(result.is_err());
        let _ = std::panic::catch_unwind(|| {
            let _attempt = Attempt::begin();
            draw("abandoned", 42, 0);
            panic!("failed attempt");
        });
        let identity = resolved(
            Fixture
                .generate_with_execution_report(
                    &GenerationRequest {
                        seed: Some(2),
                        ..Default::default()
                    },
                    &mut |_| {},
                )
                .unwrap(),
        );
        let ExecutionRandomness::Draws { draws, .. } = identity.randomness else {
            panic!("draws")
        };
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].seed, Some(2));
    }

    #[test]
    fn cancellation_at_the_terminal_callback_and_direct_return_discards_the_report() {
        let req = GenerationRequest::default();
        assert!(Fixture
            .generate_with_execution_report(&req, &mut |progress| {
                if matches!(progress, Progress::Step { current, total } if current == total) {
                    req.cancel.cancel();
                }
            })
            .is_err());
        let cancel = gen_core::CancelFlag::default();
        assert!(capture_direct(&DESCRIPTOR, &cancel, || {
            cancel.cancel();
            Ok(GenerationOutput::Images(vec![Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            }]))
        })
        .is_err());
        assert!(ATTEMPTS.with_borrow(|stack| stack.is_empty()));
    }

    #[test]
    fn staged_decoder_draws_bind_to_their_outputs_and_explicit_noise_keeps_its_digest() {
        let attempt = Attempt::begin();
        draw("image_initial", 7, 0);
        draw("image_initial", 8, 1);
        output_index(0);
        stream("pid_initial_then_epsilon", 9);
        output_index(1);
        stream("pid_initial_then_epsilon", 9);
        noise("supplied", &[0.0, -0.0], 1);
        non_deterministic("independent decoder seed cannot be supplied for replay");
        runtime(&Device::Cpu, DType::F32);
        runtime(&Device::Cpu, DType::F32);
        runtime(&Device::Cpu, DType::F16);
        runtime(&Device::Cpu, DType::F32);
        let facts = attempt.finish();
        assert_eq!(facts.runtimes.len(), 3);
        assert_eq!(facts.runtimes[0], facts.runtimes[2]);
        assert_eq!(
            facts
                .draws
                .iter()
                .map(|draw| draw.output_index)
                .collect::<Vec<_>>(),
            [0, 1, 0, 1, 1]
        );
        assert_eq!(facts.draws[4].seed, None);
        assert_eq!(
            facts.draws[4].noise_sha256.as_deref(),
            Some(
                NumericScheduleDigest::f32_values(&[0.0, -0.0])
                    .unwrap()
                    .sha256
                    .as_str()
            )
        );
        assert!(facts.non_deterministic.is_some());
    }

    #[test]
    fn dropping_an_outer_direct_scope_does_not_remove_the_inner_scope() {
        let outer = Attempt::begin();
        draw("outer", 1, 0);
        let inner = Attempt::begin();
        draw("inner", 2, 0);
        drop(outer);
        let facts = inner.finish();
        assert_eq!(facts.draws.len(), 1);
        assert_eq!(facts.draws[0].seed, Some(2));
        assert!(ATTEMPTS.with_borrow(|stack| stack.is_empty()));
    }

    #[test]
    fn incomplete_direct_observations_return_typed_unavailable() {
        let output = GenerationOutput::Images(vec![Image {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        }]);
        let attempt = Attempt::begin();
        let identity = attempt.finish_image(&DESCRIPTOR, &output);
        let ExecutionIdentity::Unavailable { reason } = identity else {
            panic!("must refuse")
        };
        assert_eq!(reason.code, "candle_execution_observation_incomplete");
        assert!(reason
            .missing_facts
            .iter()
            .any(|fact| fact.contains("sampler")));
    }

    #[test]
    fn attempts_on_concurrent_threads_do_not_share_seeds() {
        let threads: Vec<_> = [3, 9]
            .into_iter()
            .map(|seed| {
                std::thread::spawn(move || {
                    let identity = resolved(
                        Fixture
                            .generate_with_execution_report(
                                &GenerationRequest {
                                    seed: Some(seed),
                                    ..Default::default()
                                },
                                &mut |_| {},
                            )
                            .unwrap(),
                    );
                    let ExecutionRandomness::Draws { draws, .. } = identity.randomness else {
                        panic!("draws")
                    };
                    assert_eq!(draws[0].seed, Some(seed));
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
