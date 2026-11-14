//! Attempt-local observations from audio execution. Request values are never used as evidence.
//! The synchronous provider owns a thread-local frame. Callbacks are suspended, nested attempts
//! are isolated, and errors/panics discard their frame rather than leaving a last-report slot.
//!
//! Generators expose these observations through `generate_with_execution_report` and the matching
//! streaming method. Native pipeline callers can wrap their final audio operation in
//! [`capture_direct`]; invoke external sinks through [`callback`] to exclude consumer work.
//! An intermediate token API is not a completed audio generation. Opaque caller RNGs, legacy
//! resumed stages, device-global entropy and sessions without a frozen warm-cache identity
//! explicitly report unavailability instead of claiming deterministic regeneration.
use crate::{
    candle_core::{DType, Device},
    gen_core,
};
use gen_core::generator::{
    CompletedGeneration, ExecutionDraw, ExecutionFact, ExecutionIdentity, ExecutionRandomness,
    ExecutionReplaySupport, ExecutionUnavailable, NumericScheduleDigest, ResolvedExecutionIdentity,
    EXECUTION_IDENTITY_VERSION,
};
use gen_core::{
    AudioTrack, GenerationOutput, GenerationReport, GenerationRequest, Generator, ModelDescriptor,
    Progress,
};
pub use serde_json::json;
use serde_json::Value;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    marker::PhantomData,
    rc::Rc,
};

#[derive(Default)]
struct Facts {
    sampling: Vec<Value>,
    schedules: Vec<Value>,
    policies: BTreeMap<String, Vec<Value>>,
    draws: Vec<ExecutionDraw>,
    consumed_seeds: std::collections::BTreeSet<usize>,
    runtimes: Vec<Value>,
    schemes: Vec<String>,
    missing: Vec<String>,
    no_draws: Option<String>,
    nondeterministic: Option<String>,
}
thread_local! {
    static STACK: RefCell<Vec<(u64, Facts)>> = const { RefCell::new(Vec::new()) };
    static NEXT: Cell<u64> = const { Cell::new(0) };
    static SUSPENDED: Cell<usize> = const { Cell::new(0) };
}
/// A synchronous, non-transferable attempt scope, also available to native pipeline callers.
pub struct Attempt {
    id: u64,
    _thread: PhantomData<Rc<()>>,
}
impl Attempt {
    pub fn begin() -> Self {
        let id = NEXT.get();
        NEXT.set(id.wrapping_add(1));
        STACK.with_borrow_mut(|s| s.push((id, Facts::default())));
        Self {
            id,
            _thread: PhantomData,
        }
    }
    pub fn finish(
        self,
        descriptor: &ModelDescriptor,
        audio: Option<&AudioTrack>,
    ) -> ExecutionIdentity {
        let facts = STACK.with_borrow_mut(|s| {
            let i = s
                .iter()
                .position(|(id, _)| *id == self.id)
                .expect("active audio attempt");
            s.remove(i).1
        });
        facts.identity(descriptor, audio)
    }
}
impl Drop for Attempt {
    fn drop(&mut self) {
        STACK.with_borrow_mut(|s| s.retain(|(id, _)| *id != self.id));
    }
}
fn observe(f: impl FnOnce(&mut Facts)) {
    STACK.with_borrow_mut(|s| {
        if s.len() > SUSPENDED.get() {
            if let Some((_, facts)) = s.last_mut() {
                f(facts);
            }
        }
    });
}
/// Run user callbacks outside the provider observation scope, including during unwinding.
pub fn callback<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            SUSPENDED.set(self.0);
        }
    }
    let depth = STACK.with_borrow(|s| s.len());
    let _restore = Restore(SUSPENDED.replace(depth));
    f()
}
pub fn sampling(stage: &str, value: Value) {
    observe(|f| f.sampling.push(json!({"stage":stage,"settings":value})));
}
pub fn policy(key: &str, value: Value) {
    observe(|f| {
        let values = f.policies.entry(key.into()).or_default();
        if values.last() != Some(&value) {
            values.push(value);
        }
    });
}
pub fn runtime(device: &Device, dtype: DType) {
    observe(|f| {
        let v = json!({"device":format!("{:?}",device.location()),"dtype":format!("{dtype:?}")});
        if !f.runtimes.contains(&v) {
            f.runtimes.push(v);
        }
    });
}
/// Identify a completed host-only symbolic operation without inventing a tensor dtype.
pub fn host_runtime(operation: &str) {
    observe(|f| {
        f.runtimes.push(json!({"operation":operation,"execution":"host-symbolic","tensor_dtype":"not_applicable"}))
    });
}
pub fn seed(purpose: &str, scheme: &str, seed: u64) -> u64 {
    observe(|f| {
        if !f.schemes.iter().any(|s| s == scheme) {
            f.schemes.push(scheme.into());
        }
        f.draws.push(ExecutionDraw {
            output_index: 0,
            purpose: purpose.into(),
            seed: Some(seed),
            noise_sha256: None,
        });
    });
    seed
}
pub fn schedule(stage: &str, values: &[f64], estimator: &str) {
    match NumericScheduleDigest::f64_values(values) {
        Ok(digest) => observe(|f| {
            f.schedules
                .push(json!({"stage":stage,"digest":digest,"estimator":estimator}))
        }),
        Err(e) => unavailable(&e.to_string()),
    }
}
pub fn schedule_f32(stage: &str, values: &[f32], estimator: &str) {
    match NumericScheduleDigest::f32_values(values) {
        Ok(digest) => observe(|f| {
            f.schedules
                .push(json!({"stage":stage,"digest":digest,"estimator":estimator}))
        }),
        Err(e) => unavailable(&e.to_string()),
    }
}
pub fn no_draws(reason: &str) {
    observe(|f| f.no_draws = Some(reason.into()));
}
pub fn unavailable(reason: &str) {
    observe(|f| f.missing.push(reason.into()));
}
pub fn nondeterministic(reason: &str) {
    observe(|f| f.nondeterministic = Some(reason.into()));
}
impl Facts {
    fn identity(mut self, d: &ModelDescriptor, audio: Option<&AudioTrack>) -> ExecutionIdentity {
        if self.sampling.is_empty() {
            self.missing
                .push("executed audio sampler or deterministic transform was not observed".into());
        }
        if self.runtimes.is_empty() {
            self.missing
                .push("actual audio compute device/dtype was not observed".into());
        }
        if self.policies.is_empty() {
            self.missing
                .push("executed audio quality policy was not observed".into());
        }
        if self.draws.is_empty() && self.no_draws.is_none() {
            self.missing
                .push("audio random stream or explicit no-draw path was not observed".into());
        }
        if !self.missing.is_empty() {
            return ExecutionIdentity::unavailable(ExecutionUnavailable {
                code: "audio_execution_unavailable".into(),
                detail: format!("{}: {}", d.id, self.missing.join("; ")),
                missing_facts: self.missing,
            });
        }
        let schedules = if self.schedules.is_empty() {
            ExecutionFact::not_applicable(
                "autoregressive or feed-forward audio has no diffusion schedule",
            )
        } else {
            ExecutionFact::resolved(json!(self.schedules))
        };
        let randomness = if self.draws.is_empty() {
            ExecutionRandomness::NoDraws {
                reason: self.no_draws.unwrap(),
            }
        } else {
            ExecutionRandomness::Draws {
                scheme: self.schemes.join(";"),
                draws: self.draws,
            }
        };
        let output = match audio {
            Some(a) => {
                json!({"sample_rate":a.sample_rate,"channels":a.channels,"samples":a.samples.len(),"stems":a.stems.iter().map(|s|json!({"name":s.name,"samples":s.samples.len()})).collect::<Vec<_>>(),"format":"f32_pcm"})
            }
            None => json!({"artifact":"symbolic_plan","audio_rendered":false}),
        };
        let estimator = if self.schedules.is_empty() {
            ExecutionFact::not_applicable("no diffusion estimator on the executed audio path")
        } else {
            ExecutionFact::resolved(json!({"stages": self.schedules}))
        };
        let deterministic_replay = match self.nondeterministic {
            Some(reason) => ExecutionReplaySupport::NonDeterministic { reason },
            None => ExecutionReplaySupport::Deterministic {
                scope: "same inference revision, component bytes, frozen conditioning, ordered RNG streams, device and dtype".into(),
            },
        };
        let identity = ResolvedExecutionIdentity {
            version: EXECUTION_IDENTITY_VERSION,
            provider_id: d.id.into(),
            backend: d.backend.into(),
            runtime_id: "candle-audio-execution-v1".into(),
            sampling: ExecutionFact::resolved(json!(self.sampling)),
            schedule: schedules,
            estimator,
            randomness,
            deterministic_replay,
            geometry: ExecutionFact::not_applicable("audio has no image geometry"),
            quality_policy: ExecutionFact::resolved(
                json!({"runtime":self.runtimes,"policies":self.policies}),
            ),
            output_policy: ExecutionFact::resolved(output),
            provider_choices: BTreeMap::new(),
            output_units: 1,
        };
        ExecutionIdentity::Resolved {
            identity: Box::new(identity),
        }
    }
}
pub fn capture<G: Generator + ?Sized>(
    g: &G,
    r: &GenerationRequest,
    p: &mut dyn FnMut(Progress),
) -> gen_core::Result<CompletedGeneration> {
    capture_report(g.descriptor(), &r.cancel, || {
        g.generate_with_report(r, &mut |v| callback(|| p(v)))
    })
}
/// Default audio streaming emits a terminal chunk while preserving ordinary report artifacts.
/// A cancellation from the terminal sink must still prevent a completed report from escaping.
pub fn capture_terminal_streaming<G: Generator + ?Sized>(
    g: &G,
    r: &GenerationRequest,
    c: &mut dyn FnMut(gen_core::AudioChunk),
    p: &mut dyn FnMut(Progress),
) -> gen_core::Result<CompletedGeneration> {
    let completed = capture(g, r, p)?;
    if let Some(GenerationOutput::Audio(track)) = &completed.generation.output {
        callback(|| {
            c(gen_core::AudioChunk {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                index: 0,
            })
        });
    }
    if r.cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    Ok(completed)
}
pub fn capture_streaming<G: Generator + ?Sized>(
    g: &G,
    r: &GenerationRequest,
    c: &mut dyn FnMut(gen_core::AudioChunk),
    p: &mut dyn FnMut(Progress),
) -> gen_core::Result<CompletedGeneration> {
    capture_report(g.descriptor(), &r.cancel, || {
        g.generate_streaming(r, &mut |v| callback(|| c(v)), &mut |v| callback(|| p(v)))
            .map(GenerationReport::from_output)
    })
}
pub fn capture_report(
    d: &ModelDescriptor,
    cancel: &gen_core::CancelFlag,
    body: impl FnOnce() -> gen_core::Result<GenerationReport>,
) -> gen_core::Result<CompletedGeneration> {
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let attempt = Attempt::begin();
    let generation = body()?;
    if cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    let audio = match &generation.output {
        Some(GenerationOutput::Audio(a)) => Some(a),
        None if generation.artifacts.is_some() => None,
        _ => {
            return Err(gen_core::Error::Msg(
                "audio execution capture requires completed audio or an artifact".into(),
            ))
        }
    };
    let identity = attempt.finish(d, audio);
    CompletedGeneration::new(generation, identity, d)
}

/// Bind explicit or provider-owned host noise in its actual consumed order.
pub fn noise(purpose: &str, scheme: &str, digest: String) {
    observe(|f| {
        if !f.schemes.iter().any(|s| s == scheme) {
            f.schemes.push(scheme.into());
        }
        f.draws.push(ExecutionDraw {
            output_index: 0,
            purpose: purpose.into(),
            seed: None,
            noise_sha256: Some(digest),
        });
    });
}

/// A low-level entry accepting an opaque external RNG must have its constructor observed
/// for this stage. Consume that evidence once so another decode cannot reuse a stale seed.
pub fn require_seed(purpose: &str) {
    observe(|f| {
        let found = f
            .draws
            .iter()
            .enumerate()
            .find(|(i, d)| {
                d.purpose == purpose && d.seed.is_some() && !f.consumed_seeds.contains(i)
            })
            .map(|(i, _)| i);
        match found {
            Some(i) => {
                f.consumed_seeds.insert(i);
            }
            None => f.missing.push(format!(
                "{purpose}: externally supplied token RNG has no same-call constructor identity"
            )),
        }
    });
}
/// Capture a native provider entry point and perform the same final cancellation/output checks.
pub fn capture_direct(
    d: &ModelDescriptor,
    cancel: &gen_core::CancelFlag,
    body: impl FnOnce() -> gen_core::Result<GenerationOutput>,
) -> gen_core::Result<CompletedGeneration> {
    capture_report(d, cancel, || body().map(GenerationReport::from_output))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "audio-test",
            family: "test",
            backend: "candle",
            modality: gen_core::Modality::Audio,
            capabilities: gen_core::Capabilities {
                max_count: 1,
                ..Default::default()
            },
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        }
    }
    fn track() -> AudioTrack {
        AudioTrack {
            samples: vec![0.1, 0.2],
            sample_rate: 24000,
            channels: 1,
            ..Default::default()
        }
    }
    fn facts(seed_value: u64) {
        runtime(&Device::Cpu, DType::F32);
        sampling("effective", json!({"temperature_bits":0.9f32.to_bits()}));
        policy("chunking", json!({"frames":32}));
        seed("tokens", "splitmix64", seed_value);
    }
    struct Mock;
    static DESCRIPTOR: std::sync::LazyLock<ModelDescriptor> = std::sync::LazyLock::new(descriptor);
    impl Generator for Mock {
        fn descriptor(&self) -> &ModelDescriptor {
            &DESCRIPTOR
        }
        fn validate(&self, _: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            r: &GenerationRequest,
            p: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            // An effective provider fallback; deliberately different from the submitted seed.
            facts(13392);
            p(Progress::Decoding);
            if r.prompt == "fail" {
                return Err(gen_core::Error::Msg("failed".into()));
            }
            Ok(GenerationOutput::Audio(track()))
        }
    }
    #[test]
    fn completed_identity_uses_effective_defaults_not_submitted_options() {
        let req = GenerationRequest {
            seed: Some(999),
            guidance: Some(10.0),
            ..Default::default()
        };
        let result = capture(&Mock, &req, &mut |_| {}).unwrap();
        let resolved = result.execution.resolved().unwrap();
        assert_eq!(
            resolved.sampling,
            ExecutionFact::resolved(
                json!([{"stage":"effective","settings":{"temperature_bits":0.9f32.to_bits()}}])
            )
        );
        let ExecutionRandomness::Draws { draws, .. } = &resolved.randomness else {
            panic!("draws")
        };
        assert_eq!(draws[0].seed, Some(13392));
        assert_eq!(
            resolved.output_policy,
            ExecutionFact::resolved(
                json!({"sample_rate":24000,"channels":1,"samples":2,"stems":[],"format":"f32_pcm"})
            )
        );
    }
    #[test]
    fn nested_callbacks_and_other_threads_cannot_pollute_an_attempt() {
        let result = capture(&Mock, &GenerationRequest::default(), &mut |_| {
            seed("callback", "incorrect", 17);
            let inner = Attempt::begin();
            facts(23);
            assert!(inner
                .finish(&descriptor(), Some(&track()))
                .resolved()
                .is_some());
            std::thread::spawn(|| facts(91)).join().unwrap();
        })
        .unwrap();
        let ExecutionRandomness::Draws { draws, .. } =
            &result.execution.resolved().unwrap().randomness
        else {
            panic!("draws")
        };
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].seed, Some(13392));
    }
    #[test]
    fn errors_cancel_and_panic_never_leak_to_retries() {
        let mut r = GenerationRequest {
            prompt: "fail".into(),
            ..Default::default()
        };
        assert!(capture(&Mock, &r, &mut |_| {}).is_err());
        r.prompt.clear();
        let cancel = r.cancel.clone();
        assert!(matches!(
            capture(&Mock, &r, &mut |_| cancel.cancel()),
            Err(gen_core::Error::Canceled)
        ));
        let _ = std::panic::catch_unwind(|| {
            let _attempt = Attempt::begin();
            facts(100);
            panic!("abort");
        });
        let empty = Attempt::begin();
        assert!(matches!(
            empty.finish(&descriptor(), Some(&track())),
            ExecutionIdentity::Unavailable { .. }
        ));
        assert!(capture(&Mock, &GenerationRequest::default(), &mut |_| {})
            .unwrap()
            .execution
            .resolved()
            .is_some());
    }
    #[test]
    fn schedule_digest_preserves_exact_numbers_and_multiple_stages() {
        let scope = Attempt::begin();
        facts(2);
        schedule("stage1", &[1.0, 0.5, 0.0], "velocity");
        schedule("stage2", &[1.0, -0.0], "velocity");
        let result = scope.finish(&descriptor(), Some(&track()));
        let ExecutionFact::Resolved { value } = &result.resolved().unwrap().schedule else {
            panic!("schedule")
        };
        assert_eq!(
            value[0]["digest"],
            json!(NumericScheduleDigest::f64_values(&[1.0, 0.5, 0.0]).unwrap())
        );
        assert_ne!(
            value[1]["digest"],
            json!(NumericScheduleDigest::f64_values(&[1.0, 0.0]).unwrap())
        );
    }
    #[test]
    fn unavailable_overrides_partial_facts_and_plan_remains_artifact_only() {
        let scope = Attempt::begin();
        facts(5);
        unavailable("warm KV state is not exported");
        let ExecutionIdentity::Unavailable { reason } = scope.finish(&descriptor(), Some(&track()))
        else {
            panic!("unavailable")
        };
        assert!(reason.detail.contains("warm KV"));
        let scope = Attempt::begin();
        runtime(&Device::Cpu, DType::F32);
        sampling("plan", json!({"mode":"external-score"}));
        policy("plan", json!({"compute":false}));
        no_draws("external score");
        let result = scope.finish(&descriptor(), None);
        assert!(matches!(
            result.resolved().unwrap().randomness,
            ExecutionRandomness::NoDraws { .. }
        ));
        assert_eq!(
            result.resolved().unwrap().output_policy,
            ExecutionFact::resolved(json!({"artifact":"symbolic_plan","audio_rendered":false}))
        );
    }
    #[test]
    fn ordered_streams_keep_wrapping_seeds_and_explicit_noise() {
        let scope = Attempt::begin();
        facts(u64::MAX);
        seed("vocoder", "std", u64::MAX.wrapping_add(1));
        noise("acoustic", "explicit", "a".repeat(64));
        let result = scope.finish(&descriptor(), Some(&track()));
        let ExecutionRandomness::Draws { draws, .. } = &result.resolved().unwrap().randomness
        else {
            panic!("draws")
        };
        assert_eq!(
            draws.iter().map(|v| v.seed).collect::<Vec<_>>(),
            vec![Some(u64::MAX), Some(0), None]
        );
        result.validate("audio-test", "candle", 1).unwrap();
    }
    #[test]
    fn streaming_and_native_capture_bind_only_the_completed_output() {
        let req = GenerationRequest::default();
        let regular = capture(&Mock, &req, &mut |_| {}).unwrap();
        let mut chunks = 0;
        let streamed = capture_streaming(
            &Mock,
            &req,
            &mut |_| {
                chunks += 1;
                seed("callback", "bad", 0);
            },
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(regular.execution, streamed.execution);
        assert_eq!(chunks, 1);
        let direct = capture_direct(&descriptor(), &req.cancel, || {
            facts(13392);
            Ok(GenerationOutput::Audio(track()))
        })
        .unwrap();
        assert_eq!(regular.execution, direct.execution);
        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            capture_direct(&descriptor(), &cancel, || panic!(
                "must not execute canceled body"
            )),
            Err(gen_core::Error::Canceled)
        ));
    }
    #[test]
    fn external_rng_and_reused_constructor_evidence_fail_closed() {
        let scope = Attempt::begin();
        facts(1);
        require_seed("tokens");
        require_seed("tokens");
        assert!(matches!(
            scope.finish(&descriptor(), Some(&track())),
            ExecutionIdentity::Unavailable { .. }
        ));
        let scope = Attempt::begin();
        facts(1);
        require_seed("external-rng");
        no_draws("decoder itself is deterministic");
        assert!(matches!(
            scope.finish(&descriptor(), Some(&track())),
            ExecutionIdentity::Unavailable { .. }
        ));
    }
    #[test]
    fn terminal_chunk_cancellation_does_not_publish_a_report() {
        let req = GenerationRequest::default();
        let cancel = req.cancel.clone();
        assert!(matches!(
            capture_terminal_streaming(&Mock, &req, &mut |_| cancel.cancel(), &mut |_| {}),
            Err(gen_core::Error::Canceled)
        ));
        let fresh = capture_terminal_streaming(
            &Mock,
            &GenerationRequest::default(),
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert!(fresh.execution.resolved().is_some());
    }
}
