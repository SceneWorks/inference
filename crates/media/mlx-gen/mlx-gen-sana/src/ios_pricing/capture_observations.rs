//! Actual scalar source marks for isolated CPU endpoint execution. No replay
//! creates observations and no arrays are retained. The non-measuring recorder
//! does not query memory counters or invoke the expensive measurement sentinel.
use super::*;
use mlx_gen::ios_pricing::{
    IosAnchorObservation, IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck,
    IosObservedMetadata,
};
use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};

type Stored =
    std::sync::Arc<std::sync::Mutex<(IosAnchorObservation, BTreeMap<(String, String), u32>)>>;

pub(super) fn measure<T>(
    body: impl FnOnce() -> Result<T>,
) -> (Result<(T, MeasuredRun)>, IosAnchorObservation) {
    collect(|stored| {
        {
            let mut state = stored.lock().expect("Sana actual source mark lock");
            state.0.domain = IosMemoryDomain::NativeAllocatorGrowth;
            state.0.coordinate = IosMemoryCoordinate::BeforeModelLoad;
        }
        let samples = stored.clone();
        mlx_gen::ios_anchor_law::measure_phases_with_samples(
            ios_quantum_phase,
            move |sample| {
                // Already queried by the original measurement; no new native read.
                samples
                    .lock()
                    .expect("Sana actual sample lock")
                    .0
                    .baseline_bytes = sample.baseline;
            },
            body,
        )
    })
}

/// Real execution/receipt observations ONLY. No measurement, native byte bound,
/// process baseline or physical-memory evidence is created by this recorder.
pub(super) fn record<T>(body: impl FnOnce() -> Result<T>) -> (Result<T>, IosAnchorObservation) {
    collect(|_| body())
}

fn collect<T>(body: impl FnOnce(&Stored) -> Result<T>) -> (Result<T>, IosAnchorObservation) {
    let stored = std::sync::Arc::new(std::sync::Mutex::new((
        IosAnchorObservation::default(),
        BTreeMap::<(String, String), u32>::new(),
    )));
    let sink = stored.clone();
    struct Remove;
    impl Drop for Remove {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    let previous = set_quantum_listener(Some(Box::new(move |mark| {
        let mut state = sink.lock().expect("Sana actual source mark lock");
        let (observation, latest) = &mut *state;
        match mark {
            QuantumMark::Metadata {
                provider,
                kind,
                fields,
            } => {
                let index =
                    u32::try_from(observation.metadata.len()).expect("bounded fixture mark count");
                observation.metadata.push(IosObservedMetadata {
                    provider: provider.into(),
                    kind: kind.into(),
                    fields: fields.iter().map(|&(k, v)| (k.into(), v)).collect(),
                });
                latest.insert((provider.into(), kind.into()), index);
            }
            QuantumMark::Checked { label, .. } => observation.checks.push(IosObservedCheck {
                phase: ios_quantum_phase(label),
                label: label.into(),
                context: latest.values().copied().collect(),
            }),
            QuantumMark::Synchronized { .. } => {}
        }
    })));
    let guard = Remove;
    assert!(
        !previous,
        "isolated Sana capture must not replace a foreign listener"
    );
    let result = with_cpu_diagnostic_headroom(AMPLE, || body(&stored));
    drop(guard);
    let mut observation = stored
        .lock()
        .expect("Sana actual source mark lock")
        .0
        .clone();
    // Every actual event was retained (overflow is a hard failure), and the
    // body must complete successfully. Receipt validity is checked separately.
    observation.telemetry_complete = result.is_ok();
    (result, observation)
}
