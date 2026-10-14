//! Opt-in H3 producer measurements. Samples are end-of-step observations, not
//! isolated step allocation peaks or an attribution of Veda workspace bytes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Collector {
    steps: Vec<serde_json::Value>,
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
        serde_json::json!({
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
        })
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
    let returned = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    (returned == size).then(|| unsafe { info.assume_init().pti_resident_size })
}

#[cfg(target_os = "linux")]
pub fn host_process_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (size > 0).then(|| pages.saturating_mul(size as u64))
}

#[cfg(windows)]
pub fn host_process_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
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
}
