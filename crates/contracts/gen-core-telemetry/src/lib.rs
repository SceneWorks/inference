//! Opt-in execution telemetry (`PIPE-17`, `OPT-8`).
//!
//! Provides:
//! 1. The per-step host/device telemetry [`Collector`] used by MiniMax-H3 terminal telemetry,
//!    with strict field-name stability (`OPT-8`) so existing telemetry consumers see unchanged JSON
//!    keys.
//! 2. Typed stage, phase, and lifecycle telemetry events ([`TelemetryStage`],
//!    [`TelemetryLifecycleKind`], [`TelemetryPhaseTransition`], [`TelemetryEvent`]) and an
//!    object-safe [`TelemetrySink`] contract ([`NoopTelemetrySink`], [`VecTelemetrySink`]) so
//!    stage/phase/lifecycle hooks flow through `gen-core-telemetry`.
//!
//! Samples are end-of-step or hook-time observations, not isolated step allocation peaks or an
//! attribution of Veda workspace bytes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Canonical execution stage observed by `gen-core-telemetry` hooks (`PIPE-17`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryStage {
    PromptEncode,
    ConditioningEncode,
    Denoise,
    Upsample,
    Decode,
    AudioSynthesize,
    VectorSanitize,
    Custom(String),
}

/// Lifecycle hook category emitted through [`TelemetrySink`] and [`Collector`] (`PIPE-17`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryLifecycleKind {
    RunStart,
    RunComplete,
    RunFailed,
    RunCanceled,
    StageStart,
    StageEnd,
    PhaseTransition,
    MemorySnapshot,
    StepComplete,
}

/// Typed phase-boundary transition payload (`PIPE-17`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryPhaseTransition {
    pub phase_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_phase: Option<String>,
    pub to_phase: String,
    pub boundary_step: usize,
    pub wall_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_process_rss_sample_bytes: Option<u64>,
}

/// One typed stage, phase, step, or lifecycle telemetry event (`PIPE-17`, `OPT-8`).
///
/// Field names (`index`, `wallNs`, `hostProcessRssSampleBytes`, `device`) match the existing
/// [`Collector`] schema verbatim so downstream consumers never observe field-name drift.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryEvent {
    pub kind: TelemetryLifecycleKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<TelemetryStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_process_rss_sample_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition: Option<TelemetryPhaseTransition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl TelemetryEvent {
    /// Construct a lifecycle event (run start/complete/failed/canceled or memory snapshot).
    pub fn lifecycle(kind: TelemetryLifecycleKind, detail: Option<String>) -> Self {
        Self {
            kind,
            stage: None,
            phase: None,
            index: None,
            wall_ns: None,
            host_process_rss_sample_bytes: host_process_rss_bytes(),
            device: None,
            transition: None,
            detail,
        }
    }

    /// Construct a stage-start event.
    pub fn stage_start(stage: TelemetryStage) -> Self {
        Self {
            kind: TelemetryLifecycleKind::StageStart,
            stage: Some(stage),
            phase: None,
            index: None,
            wall_ns: None,
            host_process_rss_sample_bytes: host_process_rss_bytes(),
            device: None,
            transition: None,
            detail: None,
        }
    }

    /// Construct a stage-end event with injected elapsed duration and device telemetry payload.
    pub fn stage_end(
        stage: TelemetryStage,
        elapsed: Duration,
        host_rss_bytes: Option<u64>,
        device: serde_json::Value,
    ) -> Self {
        Self {
            kind: TelemetryLifecycleKind::StageEnd,
            stage: Some(stage),
            phase: None,
            index: None,
            wall_ns: Some(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)),
            host_process_rss_sample_bytes: host_rss_bytes,
            device: Some(device),
            transition: None,
            detail: None,
        }
    }

    /// Construct a phase-transition event.
    pub fn phase_transition(transition: TelemetryPhaseTransition) -> Self {
        Self {
            kind: TelemetryLifecycleKind::PhaseTransition,
            stage: Some(TelemetryStage::Denoise),
            phase: Some(transition.to_phase.clone()),
            index: Some(transition.boundary_step),
            wall_ns: Some(transition.wall_ns),
            host_process_rss_sample_bytes: transition.host_process_rss_sample_bytes,
            device: None,
            transition: Some(transition),
            detail: None,
        }
    }
}

/// Object-safe receiver for stage, phase, and lifecycle telemetry events (`PIPE-17`).
pub trait TelemetrySink: Send + Sync {
    fn emit(&self, event: &TelemetryEvent);
}

/// Zero-cost no-op [`TelemetrySink`] for runs with telemetry disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopTelemetrySink;

impl TelemetrySink for NoopTelemetrySink {
    fn emit(&self, _event: &TelemetryEvent) {}
}

/// Thread-safe in-memory [`TelemetrySink`] that records emitted events in order.
#[derive(Clone, Debug, Default)]
pub struct VecTelemetrySink {
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
}

impl VecTelemetrySink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<TelemetryEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl TelemetrySink for VecTelemetrySink {
    fn emit(&self, event: &TelemetryEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event.clone());
    }
}

pub struct Collector {
    steps: Vec<serde_json::Value>,
    lifecycle_events: Vec<TelemetryEvent>,
    sink: Option<Arc<dyn TelemetrySink>>,
    samples: Arc<Mutex<Vec<(Instant, u64)>>>,
    sampled_max: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    sampler: Option<std::thread::JoinHandle<()>>,
}

impl Default for Collector {
    fn default() -> Self {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let sampled_max = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_samples = samples.clone();
        let thread_max = sampled_max.clone();
        let thread_stop = stop.clone();
        let sampler = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                if let Some(bytes) = host_process_rss_bytes() {
                    thread_max.fetch_max(bytes, Ordering::Relaxed);
                    thread_samples.lock().unwrap().push((Instant::now(), bytes));
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        });
        Self {
            steps: Vec::new(),
            lifecycle_events: Vec::new(),
            sink: None,
            samples,
            sampled_max,
            stop,
            sampler: Some(sampler),
        }
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.join();
        }
    }
}

impl Collector {
    /// Attach an external [`TelemetrySink`] to mirror stage, phase, and lifecycle hooks (`PIPE-17`).
    pub fn with_sink(mut self, sink: Arc<dyn TelemetrySink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Record one typed lifecycle event (`PIPE-17`).
    pub fn record_lifecycle(&mut self, kind: TelemetryLifecycleKind, detail: Option<&str>) {
        let event = TelemetryEvent::lifecycle(kind, detail.map(str::to_owned));
        self.push_event(event);
    }

    /// Record entry into a pipeline stage (`PIPE-17`).
    pub fn record_stage_start(&mut self, stage: TelemetryStage) {
        let event = TelemetryEvent::stage_start(stage);
        self.push_event(event);
    }

    /// Record completion of a pipeline stage with elapsed duration and device counters (`PIPE-17`).
    pub fn record_stage_end(
        &mut self,
        stage: TelemetryStage,
        elapsed: Duration,
        device: serde_json::Value,
    ) {
        let host = host_process_rss_bytes();
        if let Some(bytes) = host {
            self.sampled_max.fetch_max(bytes, Ordering::Relaxed);
        }
        let event = TelemetryEvent::stage_end(stage, elapsed, host, device);
        self.push_event(event);
    }

    /// Record a denoise phase transition (`PIPE-17`).
    pub fn record_phase_transition(&mut self, transition: TelemetryPhaseTransition) {
        if let Some(bytes) = transition.host_process_rss_sample_bytes {
            self.sampled_max.fetch_max(bytes, Ordering::Relaxed);
        }
        let event = TelemetryEvent::phase_transition(transition);
        self.push_event(event);
    }

    /// Borrow all stage, phase, and lifecycle events recorded so far.
    pub fn events(&self) -> &[TelemetryEvent] {
        &self.lifecycle_events
    }

    fn push_event(&mut self, event: TelemetryEvent) {
        if let Some(sink) = &self.sink {
            sink.emit(&event);
        }
        self.lifecycle_events.push(event);
    }

    pub fn record(&mut self, index: usize, elapsed: Duration, device: serde_json::Value) {
        let ended = Instant::now();
        let started = ended.checked_sub(elapsed).unwrap_or(ended);
        let host = host_process_rss_bytes();
        if let Some(bytes) = host {
            self.sampled_max.fetch_max(bytes, Ordering::Relaxed);
            self.samples.lock().unwrap().push((ended, bytes));
        }
        let mut samples = self.samples.lock().unwrap();
        let window_max = samples
            .iter()
            .filter(|(at, _)| *at >= started && *at <= ended)
            .map(|(_, bytes)| *bytes)
            .max();
        let window_count = samples
            .iter()
            .filter(|(at, _)| *at >= started && *at <= ended)
            .count();
        samples.retain(|(at, _)| *at >= ended);
        if let Some(sink) = &self.sink {
            sink.emit(&TelemetryEvent {
                kind: TelemetryLifecycleKind::StepComplete,
                stage: Some(TelemetryStage::Denoise),
                phase: None,
                index: Some(index),
                wall_ns: Some(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)),
                host_process_rss_sample_bytes: host,
                device: Some(device.clone()),
                transition: None,
                detail: None,
            });
        }
        self.steps.push(step_value(
            index,
            elapsed,
            host,
            window_max,
            window_count,
            device,
        ));
    }

    pub fn finish(mut self, backend: &str) -> serde_json::Value {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.join();
        }
        let max = self.sampled_max.load(Ordering::Relaxed);
        let mut value = serde_json::json!({
            "schemaVersion": 1,
            "backend": backend,
            "timingMethod": "steady_wall_clock_inside_denoise_loop_through_existing_device_sync_before_progress",
            "hostMethod": "current_process_rss_independent_40ms_poll_plus_completed_step_sample",
            "hostScope": "current process during H3 denoise observer lifetime only; sampled maxima can miss sub-40ms transients; Apple unified-memory RSS overlaps MLX allocation",
            "hostWindowMethod": "40ms process samples in approximately elapsed step wall window, plus end-of-step sample; callback counter read shifts the reconstructed window slightly",
            "hostSampleIntervalMs": 40,
            "hostSampleMaxBytes": (max > 0).then_some(max),
            "deviceScope": "backend allocator/pool cumulative high-water read during H3 denoise; earlier lifetime use can dominate; no per-step reset or Veda-only attribution",
            "workspaceComparison": "unavailable: full denoise allocations are not attributable to Veda workspace",
            "steps": std::mem::take(&mut self.steps),
        });
        if !self.lifecycle_events.is_empty() {
            if let Some(map) = value.as_object_mut() {
                map.insert(
                    "lifecycleEvents".to_owned(),
                    serde_json::to_value(&self.lifecycle_events).unwrap_or(serde_json::Value::Null),
                );
            }
        }
        value
    }
}

fn step_value(
    index: usize,
    elapsed: Duration,
    host: Option<u64>,
    window_max: Option<u64>,
    window_count: usize,
    device: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "wallNs": u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        "hostProcessRssSampleBytes": host,
        "hostWindowSampleMaxBytes": window_max,
        "hostWindowSampleCount": window_count,
        "device": device,
    })
}

#[cfg(target_os = "macos")]
pub fn host_process_rss_bytes() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: `info` is a live, zeroed buffer of exactly `size` bytes for PROC_PIDTASKINFO, which
    // writes at most `size` bytes; getpid takes no arguments.
    let returned = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    // SAFETY: proc_taskinfo is plain-old-data and `info` was zero-initialised (a valid value);
    // `returned == size` means the kernel filled the whole struct.
    (returned == size).then(|| unsafe { info.assume_init().pti_resident_size })
}

#[cfg(target_os = "linux")]
pub fn host_process_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    // SAFETY: sysconf takes no pointers and is thread-safe; a non-positive result is rejected below.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (size > 0).then(|| pages.saturating_mul(size as u64))
}

#[cfg(windows)]
pub fn host_process_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    // SAFETY: PROCESS_MEMORY_COUNTERS is a plain-old-data C struct of integers; all-zero is valid.
    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no close; `counters` is a live
    // struct whose `cb` (passed as the size) is its exact byte size, so the call writes within it.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some(counters.WorkingSetSize as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
pub fn host_process_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn completed_steps_keep_injected_wall_time_and_nullable_device_counters() {
        // Pure serialization of an injected duration: no machine clock is sampled here.
        let first = step_value(
            1,
            Duration::from_nanos(7),
            Some(11),
            Some(13),
            2,
            serde_json::json!({"usedBytes": null}),
        );
        let first_expected = serde_json::json!({"index": 1, "wallNs": 7, "hostProcessRssSampleBytes": 11, "hostWindowSampleMaxBytes": 13, "hostWindowSampleCount": 2, "device": {"usedBytes": null}});
        assert_eq!(first, first_expected);
        let second = step_value(
            2,
            Duration::from_nanos(13),
            None,
            None,
            0,
            serde_json::json!({"usedBytes": 23}),
        );
        let second_expected = serde_json::json!({"index": 2, "wallNs": 13, "hostProcessRssSampleBytes": null, "hostWindowSampleMaxBytes": null, "hostWindowSampleCount": 0, "device": {"usedBytes": 23}});
        assert_eq!(second, second_expected);
        let mut probe = Collector::default();
        probe.record(
            1,
            Duration::from_nanos(7),
            serde_json::json!({"usedBytes": null}),
        );
        probe.record(
            2,
            Duration::from_nanos(13),
            serde_json::json!({"usedBytes": 23}),
        );
        let value = probe.finish("fake");
        assert_eq!(value["steps"][0]["index"], 1);
        assert!(value["steps"][0]["device"]["usedBytes"].is_null());
        assert_eq!(value["steps"][1]["device"]["usedBytes"], 23);
        assert_eq!(
            value["workspaceComparison"],
            "unavailable: full denoise allocations are not attributable to Veda workspace"
        );
    }

    #[test]
    fn opt8_existing_telemetry_field_names_stay_stable() {
        let mut probe = Collector::default();
        probe.record(
            0,
            Duration::from_nanos(5),
            serde_json::json!({"usedBytes": 42}),
        );
        let report = probe.finish("mlx");
        let top_keys: BTreeSet<&str> = report
            .as_object()
            .expect("top-level JSON object")
            .keys()
            .map(String::as_str)
            .collect();
        let expected_top: BTreeSet<&str> = [
            "schemaVersion",
            "backend",
            "timingMethod",
            "hostMethod",
            "hostScope",
            "hostWindowMethod",
            "hostSampleIntervalMs",
            "hostSampleMaxBytes",
            "deviceScope",
            "workspaceComparison",
            "steps",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            top_keys, expected_top,
            "OPT-8: existing telemetry top-level keys must stay unchanged when only step hooks run"
        );
        let step_keys: BTreeSet<&str> = report["steps"][0]
            .as_object()
            .expect("step JSON object")
            .keys()
            .map(String::as_str)
            .collect();
        let expected_step: BTreeSet<&str> = [
            "index",
            "wallNs",
            "hostProcessRssSampleBytes",
            "hostWindowSampleMaxBytes",
            "hostWindowSampleCount",
            "device",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            step_keys, expected_step,
            "OPT-8: existing telemetry per-step keys must stay unchanged"
        );
    }

    #[test]
    fn pipe17_stage_phase_and_lifecycle_hooks_flow_through_collector_and_sink() {
        let sink = Arc::new(VecTelemetrySink::new());
        let mut collector = Collector::default().with_sink(sink.clone());
        collector.record_lifecycle(TelemetryLifecycleKind::RunStart, Some("unit-test"));
        collector.record_stage_start(TelemetryStage::PromptEncode);
        collector.record_stage_end(
            TelemetryStage::PromptEncode,
            Duration::from_nanos(120),
            serde_json::json!({"allocatedBytes": 1024}),
        );
        collector.record_phase_transition(TelemetryPhaseTransition {
            phase_index: 1,
            from_phase: Some("low_res".to_owned()),
            to_phase: "full_res".to_owned(),
            boundary_step: 4,
            wall_ns: 99,
            host_process_rss_sample_bytes: Some(2048),
        });
        collector.record(
            4,
            Duration::from_nanos(50),
            serde_json::json!({"usedBytes": 4096}),
        );
        collector.record_lifecycle(TelemetryLifecycleKind::RunComplete, None);

        assert_eq!(collector.events().len(), 5);
        let sink_events = sink.snapshot();
        assert_eq!(sink_events.len(), 6);
        assert_eq!(sink_events[0].kind, TelemetryLifecycleKind::RunStart);
        assert_eq!(sink_events[1].kind, TelemetryLifecycleKind::StageStart);
        assert_eq!(sink_events[1].stage, Some(TelemetryStage::PromptEncode));
        let expected_stage_end = TelemetryEvent::stage_end(
            TelemetryStage::PromptEncode,
            Duration::from_nanos(120),
            sink_events[2].host_process_rss_sample_bytes,
            serde_json::json!({"allocatedBytes": 1024}),
        );
        assert_eq!(sink_events[2], expected_stage_end);
        assert_eq!(sink_events[3].kind, TelemetryLifecycleKind::PhaseTransition);
        assert_eq!(sink_events[3].phase.as_deref(), Some("full_res"));
        assert_eq!(sink_events[4].kind, TelemetryLifecycleKind::StepComplete);
        assert_eq!(sink_events[4].index, Some(4));
        assert_eq!(sink_events[5].kind, TelemetryLifecycleKind::RunComplete);

        let report = collector.finish("candle");
        assert_eq!(report["schemaVersion"], 1);
        assert_eq!(report["steps"].as_array().unwrap().len(), 1);
        let lifecycle = report["lifecycleEvents"].as_array().unwrap();
        assert_eq!(lifecycle.len(), 5);
        assert_eq!(
            lifecycle[2],
            serde_json::to_value(&expected_stage_end).unwrap()
        );
        assert_eq!(lifecycle[3]["transition"]["toPhase"], "full_res");
    }
}
