//! Actual scalar source marks for the isolated CPU endpoint captures. No replay
//! creates observations, no arrays are retained, and native sampling/reset points
//! remain those of measure_phases_with_samples.
use super::*;
use mlx_gen::ios_pricing::{
    IosAnchorObservation, IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck,
    IosObservedMetadata,
};
use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};

pub(super) fn measure<T>(
    body: impl FnOnce() -> Result<T>,
) -> (Result<(T, MeasuredRun)>, IosAnchorObservation) {
    let stored = std::sync::Arc::new(std::sync::Mutex::new((
        IosAnchorObservation {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            ..Default::default()
        },
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
    let samples = stored.clone();
    let result = with_cpu_diagnostic_headroom(AMPLE, || {
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
    });
    drop(guard);
    let observation = stored
        .lock()
        .expect("Sana actual source mark lock")
        .0
        .clone();
    (result, observation)
}
