//! Bounded, fail-closed child execution for the SC-20671 and SC-20676 campaigns.
//!
//! This module deliberately uses only the standard library and serde so its process-control tests
//! can run without linking MLX or starting a model. Callers must supply a frozen policy for every run;
//! this module does not choose a host reserve or silently shorten a context band.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
#[cfg(target_os = "macos")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Identity of the macOS admission measure recorded with every measurement.
pub const DARWIN_AVAILABLE_METRIC: &str = "darwin-vm-stat-available-v1";
/// The supervisor hands the worker the host measurement it was admitted on, so the worker's own
/// receipt records the decision's inputs.
pub const HOST_MEMORY_ADMISSION_ENV: &str = "SCENEWORKS_CAMPAIGN_HOST_MEMORY_ADMISSION";

/// One vm_stat host-memory measurement and every component of the admission measure.
///
/// `available_bytes = (free + speculative + purgeable + reclaimable_file) * page_size`, where
/// `reclaimable_file = min(max(0, inactive - purgeable), max(0, file_backed - speculative))`.
/// vm_stat's counters partition as follows: free and speculative are disjoint lists (vm_stat prints
/// the kernel's free_count minus speculative); `File-backed + Anonymous = active + inactive +
/// speculative + throttled`, so speculative read-ahead sits inside File-backed and is removed
/// before the credit; purgeable (volatile) pages are anonymous and can sit on the inactive list,
/// so they are removed from inactive. The min is therefore an upper bound on inactive, clean
/// file-backed pages -- the cache the pageout daemon frees without the compressor or swap -- and
/// file-backed pages on the active list (mapped, in use) are never credited beyond the inactive
/// list's size. Free plus speculative alone counts none of it, which made every admission after a
/// large file read (model hashing) pessimistic by the size of the read.
///
/// Mirrors `scripts/media_campaign_supervisor.py` `darwin_host_memory` and
/// `.github/kv-poc/common.sh` `host_memory_from_vm_stat`; all three are pinned by
/// `testdata/darwin-host-memory-cases.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostMemory {
    pub metric: String,
    pub page_size_bytes: u64,
    pub free_pages: u64,
    pub speculative_pages: u64,
    pub purgeable_pages: u64,
    pub inactive_pages: u64,
    pub file_backed_pages: u64,
    pub reclaimable_file_pages: u64,
    pub available_bytes: u64,
}

impl HostMemory {
    /// Derive the admission measure from raw vm_stat page counts; `None` on overflow.
    pub fn from_pages(
        page_size_bytes: u64,
        free_pages: u64,
        speculative_pages: u64,
        purgeable_pages: u64,
        inactive_pages: u64,
        file_backed_pages: u64,
    ) -> Option<Self> {
        let reclaimable_file_pages = inactive_pages
            .saturating_sub(purgeable_pages)
            .min(file_backed_pages.saturating_sub(speculative_pages));
        let available_bytes = free_pages
            .checked_add(speculative_pages)?
            .checked_add(purgeable_pages)?
            .checked_add(reclaimable_file_pages)?
            .checked_mul(page_size_bytes)?;
        Some(Self {
            metric: DARWIN_AVAILABLE_METRIC.into(),
            page_size_bytes,
            free_pages,
            speculative_pages,
            purgeable_pages,
            inactive_pages,
            file_backed_pages,
            reclaimable_file_pages,
            available_bytes,
        })
    }

    /// Free plus speculative bytes: the live watchdog's measure, unchanged by the admission metric.
    pub fn free_and_speculative_bytes(&self) -> u64 {
        self.free_pages
            .saturating_add(self.speculative_pages)
            .saturating_mul(self.page_size_bytes)
    }

    /// A recorded measurement must name this metric, carry a real page size, and recompute
    /// exactly from its own components.
    pub fn validate(&self) -> Result<(), String> {
        if self.metric != DARWIN_AVAILABLE_METRIC
            || self.page_size_bytes < 4096
            || !self.page_size_bytes.is_power_of_two()
            || Self::from_pages(
                self.page_size_bytes,
                self.free_pages,
                self.speculative_pages,
                self.purgeable_pages,
                self.inactive_pages,
                self.file_backed_pages,
            )
            .as_ref()
                != Some(self)
        {
            return Err("host memory components do not recompute the admission measure".into());
        }
        Ok(())
    }

    /// An admitted row's measurement must also cover cap plus reserve.
    pub fn validate_admits(&self, reserve_bytes: u64, cap_bytes: u64) -> Result<(), String> {
        self.validate()?;
        if reserve_bytes
            .checked_add(cap_bytes)
            .is_none_or(|required| self.available_bytes < required)
        {
            return Err("recorded host available memory is below reserve plus child cap".into());
        }
        Ok(())
    }
}

/// Worker side: the measurement its supervisor admitted it on, checked against the policy the
/// worker itself loaded. A worker started outside the supervisor has none and refuses.
pub fn admitted_host_memory(reserve_bytes: u64, cap_bytes: u64) -> Result<HostMemory, String> {
    let raw = std::env::var(HOST_MEMORY_ADMISSION_ENV).map_err(|_| {
        format!("{HOST_MEMORY_ADMISSION_ENV} is unset: the worker was not admitted by the campaign supervisor")
    })?;
    let host: HostMemory = serde_json::from_str(&raw)
        .map_err(|error| format!("{HOST_MEMORY_ADMISSION_ENV} is malformed: {error}"))?;
    host.validate_admits(reserve_bytes, cap_bytes)?;
    Ok(host)
}

#[derive(Clone, Debug)]
pub struct SafetyPolicy {
    pub deadline: Duration,
    pub poll_interval: Duration,
    pub term_grace: Duration,
    pub host_free_reserve_bytes: u64,
    pub child_footprint_cap_bytes: u64,
    pub max_context_tokens: u64,
    pub max_request_tokens: u64,
    pub stdout_cap_bytes: u64,
    pub stderr_cap_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct RunRequest {
    /// The row's planned context, including its declared band target.
    pub context_tokens: u64,
    /// Largest token count any request in this row can present to the model.
    pub request_tokens: u64,
    /// Fresh paths outside the repository; existing files are never overwritten.
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    InvalidPolicy,
    ContextCeiling,
    PreflightMemory,
    HostMemory,
    ChildFootprint,
    Deadline,
    LogLimit,
    ProbeFailure,
    Io,
    ChildExit,
}

#[derive(Debug)]
pub struct Failure {
    pub reason: StopReason,
    pub detail: String,
    /// Present after spawn, including when the child had to be terminated and reaped.
    pub pid: Option<u32>,
    /// The pre-spawn host measurement, present whenever the probe succeeded (refusals included).
    pub host_memory: Option<Box<HostMemory>>,
}

impl Failure {
    fn new(reason: StopReason, detail: impl Into<String>, pid: Option<u32>) -> Self {
        Self {
            reason,
            detail: detail.into(),
            pid,
            host_memory: None,
        }
    }
}

/// The host probe is separate from MLX allocation counters: a parent process must not report its
/// own MLX counters as though they belonged to the worker PID.
pub trait MemoryProbe {
    /// One host measurement: admission reads `available_bytes`, the live watchdog reads
    /// `free_and_speculative_bytes`.
    fn host_memory(&mut self, deadline: Instant) -> io::Result<HostMemory>;
    fn child_footprint_bytes(&mut self, pid: u32, deadline: Instant) -> io::Result<u64>;
}

fn validate(policy: &SafetyPolicy, request: &RunRequest) -> Result<(), Failure> {
    if policy.deadline.is_zero()
        || policy.poll_interval.is_zero()
        || policy.term_grace.is_zero()
        || policy.host_free_reserve_bytes == 0
        || policy.child_footprint_cap_bytes == 0
        || policy.max_context_tokens == 0
        || policy.max_request_tokens == 0
        || policy.stdout_cap_bytes == 0
        || policy.stderr_cap_bytes == 0
        || policy.poll_interval >= policy.deadline
        || policy.term_grace >= policy.deadline
        || Instant::now().checked_add(policy.deadline).is_none()
        || Instant::now().checked_add(policy.term_grace).is_none()
        || policy
            .host_free_reserve_bytes
            .checked_add(policy.child_footprint_cap_bytes)
            .is_none()
        || request.stdout_path == request.stderr_path
    {
        return Err(Failure::new(
            StopReason::InvalidPolicy,
            "missing, overflowing, or inconsistent mandatory safety policy",
            None,
        ));
    }
    if request.context_tokens == 0
        || request.context_tokens > policy.max_context_tokens
        || request.request_tokens == 0
        || request.request_tokens > request.context_tokens
        || request.request_tokens > policy.max_request_tokens
    {
        return Err(Failure::new(
            StopReason::ContextCeiling,
            "row context or request exceeds its declared ceiling",
            None,
        ));
    }
    Ok(())
}

fn fresh_log(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn drain_bounded(
    mut input: impl Read,
    mut file: File,
    cap: u64,
    exceeded: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut written = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let count = match input.read(&mut buffer) {
            Ok(count) => count,
            Err(error) => {
                failed.store(true, Ordering::Release);
                return Err(error);
            }
        };
        if count == 0 {
            return file.flush();
        }
        let remaining = cap.saturating_sub(written) as usize;
        let keep = remaining.min(count);
        if keep > 0 {
            if let Err(error) = file.write_all(&buffer[..keep]) {
                failed.store(true, Ordering::Release);
                return Err(error);
            }
            written += keep as u64;
        }
        if keep < count {
            exceeded.store(true, Ordering::Release);
            // Keep draining until the supervisor stops the worker, so a full pipe cannot deadlock
            // the child while termination and reaping are in progress.
        }
    }
}

fn terminate_and_reap(
    child: &mut Child,
    grace: Duration,
    poll: Duration,
) -> io::Result<ExitStatus> {
    #[cfg(not(unix))]
    let _ = (grace, poll);
    if let Ok(Some(status)) = child.try_wait() {
        return Ok(status);
    }
    #[cfg(unix)]
    {
        // std has Child::kill (SIGKILL) but no SIGTERM API. /bin/kill keeps this module std-only.
        let _ = Command::new("/bin/kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let until = Instant::now() + grace;
        while Instant::now() < until {
            if let Ok(Some(status)) = child.try_wait() {
                return Ok(status);
            }
            thread::sleep(poll.min(until.saturating_duration_since(Instant::now())));
        }
    }
    // Also used on non-Unix platforms, where the graceful signal is unavailable.
    if !matches!(child.try_wait(), Ok(Some(_))) {
        let _ = child.kill();
    }
    child.wait()
}

/// Execute one row. Every post-spawn failure terminates and reaps the child before returning.
/// Both logs are byte-bounded even when the child writes faster than the polling interval.
pub fn run_guarded<P: MemoryProbe>(
    command: &mut Command,
    request: &RunRequest,
    policy: &SafetyPolicy,
    probe: &mut P,
) -> Result<ExitStatus, Failure> {
    validate(policy, request)?;
    let preflight_deadline = Instant::now() + policy.deadline;
    let host = probe.host_memory(preflight_deadline).map_err(|error| {
        Failure::new(
            StopReason::ProbeFailure,
            format!("host preflight probe: {error}"),
            None,
        )
    })?;
    run_admitted(command, request, policy, probe, &host, preflight_deadline).map_err(
        |mut failure| {
            failure.host_memory = Some(Box::new(host.clone()));
            failure
        },
    )
}

fn run_admitted<P: MemoryProbe>(
    command: &mut Command,
    request: &RunRequest,
    policy: &SafetyPolicy,
    probe: &mut P,
    host: &HostMemory,
    preflight_deadline: Instant,
) -> Result<ExitStatus, Failure> {
    let available = host.available_bytes;
    let admission = policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes;
    if available < admission {
        return Err(Failure::new(
            StopReason::PreflightMemory,
            format!("host available {available} bytes is below admission {admission} bytes"),
            None,
        ));
    }
    if Instant::now() >= preflight_deadline {
        return Err(Failure::new(
            StopReason::Deadline,
            "preflight exceeded the row deadline",
            None,
        ));
    }
    let stdout = fresh_log(&request.stdout_path)
        .map_err(|error| Failure::new(StopReason::Io, format!("open stdout log: {error}"), None))?;
    let stderr = match fresh_log(&request.stderr_path) {
        Ok(file) => file,
        Err(error) => {
            let _ = fs::remove_file(&request.stdout_path);
            return Err(Failure::new(
                StopReason::Io,
                format!("open stderr log: {error}"),
                None,
            ));
        }
    };
    // The files created above reserve the paths. Drain threads own them; the child only owns pipe
    // write ends, so neither an unbounded output() buffer nor a full pipe can stall the parent.
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let host_json = serde_json::to_string(host).map_err(|error| {
        Failure::new(
            StopReason::Io,
            format!("encode host admission: {error}"),
            None,
        )
    })?;
    command.env(HOST_MEMORY_ADMISSION_ENV, host_json);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&request.stdout_path);
            let _ = fs::remove_file(&request.stderr_path);
            return Err(Failure::new(
                StopReason::Io,
                format!("spawn worker: {error}"),
                None,
            ));
        }
    };
    let pid = child.id();
    let exceeded = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let out_reader = child.stdout.take().expect("piped stdout");
    let err_reader = child.stderr.take().expect("piped stderr");
    let out_thread = {
        let exceeded = Arc::clone(&exceeded);
        let failed = Arc::clone(&failed);
        let cap = policy.stdout_cap_bytes;
        thread::Builder::new()
            .name("campaign-stdout".into())
            .spawn(move || drain_bounded(out_reader, stdout, cap, exceeded, failed))
    };
    let out_thread = match out_thread {
        Ok(thread) => thread,
        Err(error) => {
            let _ = terminate_and_reap(&mut child, policy.term_grace, policy.poll_interval);
            return Err(Failure::new(
                StopReason::Io,
                format!("start stdout drain: {error}"),
                Some(pid),
            ));
        }
    };
    let err_thread = {
        let exceeded = Arc::clone(&exceeded);
        let failed = Arc::clone(&failed);
        let cap = policy.stderr_cap_bytes;
        thread::Builder::new()
            .name("campaign-stderr".into())
            .spawn(move || drain_bounded(err_reader, stderr, cap, exceeded, failed))
    };
    let err_thread = match err_thread {
        Ok(thread) => thread,
        Err(error) => {
            let _ = terminate_and_reap(&mut child, policy.term_grace, policy.poll_interval);
            let _ = out_thread.join();
            return Err(Failure::new(
                StopReason::Io,
                format!("start stderr drain: {error}"),
                Some(pid),
            ));
        }
    };
    let deadline = preflight_deadline;
    let result = loop {
        if exceeded.load(Ordering::Acquire) {
            break Err(Failure::new(
                StopReason::LogLimit,
                "worker exceeded a bounded log file",
                Some(pid),
            ));
        }
        if failed.load(Ordering::Acquire) {
            break Err(Failure::new(
                StopReason::Io,
                "worker log drain failed",
                Some(pid),
            ));
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => {
                break Err(Failure::new(
                    StopReason::Io,
                    format!("poll worker: {error}"),
                    Some(pid),
                ));
            }
        }
        if Instant::now() >= deadline {
            break Err(Failure::new(
                StopReason::Deadline,
                "worker exceeded its wall-clock deadline",
                Some(pid),
            ));
        }
        let free = match probe.host_memory(deadline) {
            Ok(host) => host.free_and_speculative_bytes(),
            Err(error) => {
                break Err(Failure::new(
                    StopReason::ProbeFailure,
                    format!("live host probe: {error}"),
                    Some(pid),
                ));
            }
        };
        if free < policy.host_free_reserve_bytes {
            break Err(Failure::new(
                StopReason::HostMemory,
                format!("host free {free} bytes fell below reserve"),
                Some(pid),
            ));
        }
        let footprint = match probe.child_footprint_bytes(pid, deadline) {
            Ok(bytes) => bytes,
            Err(error) => {
                // A fast child can exit between try_wait and the PID probe. Preserve that exit;
                // every other probe failure stops the row.
                if let Ok(Some(status)) = child.try_wait() {
                    break Ok(status);
                }
                break Err(Failure::new(
                    StopReason::ProbeFailure,
                    format!("child footprint probe: {error}"),
                    Some(pid),
                ));
            }
        };
        if footprint > policy.child_footprint_cap_bytes {
            break Err(Failure::new(
                StopReason::ChildFootprint,
                format!("child footprint {footprint} bytes exceeded cap"),
                Some(pid),
            ));
        }
        thread::sleep(
            policy
                .poll_interval
                .min(deadline.saturating_duration_since(Instant::now())),
        );
    };
    let result = match result {
        Ok(status) => Ok(status),
        Err(failure) => {
            match terminate_and_reap(&mut child, policy.term_grace, policy.poll_interval) {
                Ok(_) => Err(failure),
                Err(error) => Err(Failure::new(
                    StopReason::Io,
                    format!("{}; could not reap worker: {error}", failure.detail),
                    Some(pid),
                )),
            }
        }
    };
    let stdout_ok = out_thread.join().ok().and_then(Result::ok).is_some();
    let stderr_ok = err_thread.join().ok().and_then(Result::ok).is_some();
    let drains_ok = stdout_ok && stderr_ok;
    if exceeded.load(Ordering::Acquire) {
        return Err(Failure::new(
            StopReason::LogLimit,
            "worker exceeded a bounded log file",
            Some(pid),
        ));
    }
    if !drains_ok {
        return Err(Failure::new(
            StopReason::Io,
            "worker log drain failed",
            Some(pid),
        ));
    }
    let status = result?;
    if !status.success() {
        return Err(Failure::new(
            StopReason::ChildExit,
            format!("worker exited with {status}"),
            Some(pid),
        ));
    }
    Ok(status)
}

/// Production probe. Pre-spawn admission uses [`HostMemory::available_bytes`] (free, speculative,
/// purgeable, and inactive clean file cache); the live reserve watchdog still uses only free and
/// speculative pages. Compressed and anonymous inactive pages are never counted.
pub struct SystemProbe;

#[cfg(any(target_os = "macos", all(test, unix)))]
fn parse_vm_stat(text: &str) -> io::Result<HostMemory> {
    let page_size = text
        .lines()
        .next()
        .and_then(|line| line.split("page size of ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value >= 4096 && value.is_power_of_two())
        .ok_or_else(|| io::Error::other("vm_stat has no valid page size"))?;
    let pages = |key: &str| -> io::Result<u64> {
        let mut matching = text.lines().skip(1).filter_map(|line| {
            line.trim()
                .strip_prefix(key)
                .and_then(|tail| tail.strip_prefix(':'))
        });
        let value = matching
            .next()
            .and_then(|tail| tail.trim().strip_suffix('.'))
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|digits| digits.parse::<u64>().ok())
            .ok_or_else(|| io::Error::other(format!("vm_stat has no valid {key}")))?;
        if matching.next().is_some() {
            return Err(io::Error::other(format!("vm_stat duplicates {key}")));
        }
        Ok(value)
    };
    HostMemory::from_pages(
        page_size,
        pages("Pages free")?,
        pages("Pages speculative")?,
        pages("Pages purgeable")?,
        pages("Pages inactive")?,
        pages("File-backed pages")?,
    )
    .ok_or_else(|| io::Error::other("vm_stat available-byte count overflows"))
}

#[cfg(any(target_os = "macos", all(test, unix)))]
fn parse_footprint(text: &str) -> io::Result<u64> {
    let mut matching = text
        .lines()
        .filter_map(|line| line.trim().strip_prefix("phys_footprint:"));
    let value = matching
        .next()
        .ok_or_else(|| io::Error::other("footprint has no phys_footprint"))?;
    if matching.next().is_some() {
        return Err(io::Error::other("footprint duplicates phys_footprint"));
    }
    let mut parts = value.split_whitespace();
    let number = parts
        .next()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| io::Error::other("footprint has invalid size"))?;
    let multiplier = match parts.next().unwrap_or("B").to_ascii_uppercase().as_str() {
        "B" => 1.0,
        "K" | "KB" => 1024.0,
        "M" | "MB" => 1024.0 * 1024.0,
        "G" | "GB" => 1024.0 * 1024.0 * 1024.0,
        _ => return Err(io::Error::other("footprint has unknown size unit")),
    };
    if parts.next().is_some() {
        return Err(io::Error::other("footprint has trailing size fields"));
    }
    let bytes = (number * multiplier).ceil();
    if !bytes.is_finite() || bytes > u64::MAX as f64 {
        return Err(io::Error::other("footprint size overflows"));
    }
    Ok(bytes as u64)
}

#[cfg(target_os = "macos")]
fn bounded_system_output(command: &str, args: &[&str], deadline: Instant) -> io::Result<String> {
    // A stalled OS sampler must not consume the whole model-row deadline. Two seconds bounds each
    // sample, including the first preflight sample, while the caller's earlier deadline still wins.
    let deadline = deadline.min(Instant::now() + Duration::from_secs(2));
    if Instant::now() >= deadline {
        return Err(io::Error::other("memory probe deadline elapsed"));
    }
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "sc20671-probe-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = fresh_log(&path)?;
    let mut child = match Command::new(command)
        .args(args)
        .stdout(Stdio::from(file))
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&path);
            return Err(error);
        }
    };
    let status = loop {
        if fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 64 * 1024) {
            let _ = terminate_and_reap(
                &mut child,
                Duration::from_millis(100),
                Duration::from_millis(10),
            );
            let _ = fs::remove_file(&path);
            return Err(io::Error::other("memory probe output exceeded 64 KiB"));
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let _ = terminate_and_reap(
                    &mut child,
                    Duration::from_millis(100),
                    Duration::from_millis(10),
                );
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        }
        if Instant::now() >= deadline {
            let _ = terminate_and_reap(
                &mut child,
                Duration::from_millis(100),
                Duration::from_millis(10),
            );
            let _ = fs::remove_file(&path);
            return Err(io::Error::other("memory probe timed out"));
        }
        thread::sleep(Duration::from_millis(10));
    };
    let bytes = fs::read(&path);
    let _ = fs::remove_file(&path);
    if !status.success() {
        return Err(io::Error::other("memory probe command failed"));
    }
    let bytes = bytes?;
    if bytes.len() > 64 * 1024 {
        return Err(io::Error::other("memory probe output exceeded 64 KiB"));
    }
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

impl MemoryProbe for SystemProbe {
    fn host_memory(&mut self, deadline: Instant) -> io::Result<HostMemory> {
        #[cfg(target_os = "macos")]
        {
            parse_vm_stat(&bounded_system_output("/usr/bin/vm_stat", &[], deadline)?)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = deadline;
            Err(io::Error::other(
                "campaign host memory probe requires macOS",
            ))
        }
    }

    fn child_footprint_bytes(&mut self, pid: u32, deadline: Instant) -> io::Result<u64> {
        #[cfg(target_os = "macos")]
        {
            parse_footprint(&bounded_system_output(
                "/usr/bin/footprint",
                &["--pid", &pid.to_string(), "--noCategories", "--wired"],
                deadline,
            )?)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (pid, deadline);
            Err(io::Error::other(
                "campaign child footprint probe requires macOS",
            ))
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicU64;

    struct FakeProbe {
        host: VecDeque<io::Result<HostMemory>>,
        footprint: VecDeque<io::Result<u64>>,
    }

    /// A one-byte-page measurement whose free pages are the whole available measure.
    fn free_only(bytes: u64) -> HostMemory {
        HostMemory::from_pages(1, bytes, 0, 0, 0, 0).unwrap()
    }

    impl FakeProbe {
        fn new(free: impl IntoIterator<Item = io::Result<u64>>) -> Self {
            Self::hosts(free.into_iter().map(|sample| sample.map(free_only)))
        }

        fn hosts(host: impl IntoIterator<Item = io::Result<HostMemory>>) -> Self {
            Self {
                host: host.into_iter().collect(),
                footprint: VecDeque::new(),
            }
        }
    }

    impl MemoryProbe for FakeProbe {
        fn host_memory(&mut self, _: Instant) -> io::Result<HostMemory> {
            self.host.pop_front().unwrap_or_else(|| Ok(free_only(1000)))
        }
        fn child_footprint_bytes(&mut self, _: u32, _: Instant) -> io::Result<u64> {
            self.footprint.pop_front().unwrap_or(Ok(1))
        }
    }

    fn policy() -> SafetyPolicy {
        SafetyPolicy {
            deadline: Duration::from_millis(250),
            poll_interval: Duration::from_millis(10),
            term_grace: Duration::from_millis(30),
            host_free_reserve_bytes: 100,
            child_footprint_cap_bytes: 200,
            max_context_tokens: 1000,
            max_request_tokens: 1000,
            stdout_cap_bytes: 1024,
            stderr_cap_bytes: 1024,
        }
    }

    fn new_request() -> RunRequest {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "sc20671-supervisor-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        RunRequest {
            context_tokens: 100,
            request_tokens: 80,
            stdout_path: root.join("stdout.log"),
            stderr_path: root.join("stderr.log"),
        }
    }

    fn gone(pid: u32) -> bool {
        !Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    /// The cross-language fixture shared with scripts/tests/test_darwin_host_memory.py (Python
    /// supervisor and the kv-poc workflow precheck).
    fn host_memory_cases() -> Vec<(String, String, Option<HostMemory>)> {
        let fixture: serde_json::Value =
            serde_json::from_slice(include_bytes!("../testdata/darwin-host-memory-cases.json"))
                .unwrap();
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| {
                (
                    case["name"].as_str().unwrap().to_owned(),
                    case["vmStat"].as_str().unwrap().to_owned(),
                    serde_json::from_value(case["expect"].clone()).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn vm_stat_admission_matches_the_shared_fixture_and_fails_closed() {
        let cases = host_memory_cases();
        assert!(cases
            .iter()
            .any(|(name, ..)| name == "heavy-anonymous-inactive"));
        for (name, text, expect) in cases {
            match expect {
                Some(expect) => {
                    let host = parse_vm_stat(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
                    assert_eq!(host, expect, "{name}");
                    host.validate().unwrap();
                    assert_eq!(
                        serde_json::to_value(&host).unwrap()["reclaimableFilePages"],
                        expect.reclaimable_file_pages,
                        "{name}: serialized field names are the shared contract"
                    );
                }
                None => assert!(parse_vm_stat(&text).is_err(), "{name} must fail closed"),
            }
        }
    }

    #[test]
    fn fixture_discriminates_every_plausible_wrong_definition() {
        type Credit = fn(&HostMemory) -> u64;
        let mutants: [(&str, Credit); 6] = [
            ("drop the min", |h| {
                h.file_backed_pages.saturating_sub(h.speculative_pages)
            }),
            ("count all inactive", |h| h.inactive_pages),
            ("drop file-backed", |h| {
                h.inactive_pages.saturating_sub(h.purgeable_pages)
            }),
            ("speculative counted twice", |h| {
                h.inactive_pages
                    .saturating_sub(h.purgeable_pages)
                    .min(h.file_backed_pages)
            }),
            ("purgeable counted twice", |h| {
                h.inactive_pages
                    .min(h.file_backed_pages.saturating_sub(h.speculative_pages))
            }),
            ("no file cache", |_| 0),
        ];
        let valid: Vec<HostMemory> = host_memory_cases()
            .into_iter()
            .filter_map(|(.., expect)| expect)
            .collect();
        for (name, mutant) in mutants {
            assert!(
                valid.iter().any(|h| mutant(h) != h.reclaimable_file_pages),
                "fixture cannot tell `{name}` from the definition"
            );
        }
    }

    #[test]
    fn recorded_measurement_must_recompute_and_cover_the_admission() {
        let host = host_memory_cases()
            .into_iter()
            .find_map(|(name, _, expect)| (name == "heavy-anonymous-inactive").then_some(expect))
            .flatten()
            .unwrap();
        host.validate_admits(host.available_bytes - 1, 1).unwrap();
        assert!(host.validate_admits(host.available_bytes, 1).is_err());
        for tampered in [
            HostMemory {
                available_bytes: host.available_bytes + host.page_size_bytes,
                ..host.clone()
            },
            HostMemory {
                reclaimable_file_pages: host.inactive_pages,
                ..host.clone()
            },
            HostMemory {
                metric: "free-plus-speculative".into(),
                ..host.clone()
            },
            HostMemory {
                page_size_bytes: 1,
                ..host.clone()
            },
        ] {
            assert!(tampered.validate().is_err(), "{tampered:?}");
        }
    }

    #[test]
    fn admission_counts_file_cache_but_the_watchdog_keeps_free_plus_speculative() {
        // 10 free + 290 inactive file-backed pages: available 300 covers cap 200 + reserve 100,
        // while free plus speculative (10) is under the reserve, so the live watchdog trips.
        let cached = HostMemory::from_pages(1, 10, 0, 0, 290, 290).unwrap();
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::hosts([Ok(cached.clone()), Ok(cached.clone())]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::HostMemory, "{}", failure.detail);
        assert_eq!(failure.host_memory.as_deref(), Some(&cached));
        assert!(gone(failure.pid.unwrap()));
        // The same 290 inactive pages as anonymous memory (file-backed only 5) are not credited.
        let anonymous = HostMemory::from_pages(1, 10, 0, 0, 290, 5).unwrap();
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::hosts([Ok(anonymous.clone())]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::PreflightMemory);
        assert!(failure.detail.contains("host available 15 bytes"));
        assert_eq!(failure.host_memory.as_deref(), Some(&anonymous));
        assert!(failure.pid.is_none() && !request.stdout_path.exists());
    }

    #[test]
    fn worker_receives_the_measurement_it_was_admitted_on() {
        let host = HostMemory::from_pages(4096, 1, 0, 0, 0, 0).unwrap();
        let request = new_request();
        let expected = serde_json::to_string(&host).unwrap();
        run_guarded(
            Command::new("/bin/sh").args([
                "-c",
                &format!("test \"${HOST_MEMORY_ADMISSION_ENV}\" = '{expected}'"),
            ]),
            &request,
            &policy(),
            &mut FakeProbe::hosts([Ok(host.clone())]),
        )
        .unwrap();
        let decoded: HostMemory = serde_json::from_str(&expected).unwrap();
        decoded.validate_admits(100, 200).unwrap();
        assert!(decoded.validate_admits(4000, 97).is_err());
    }

    #[test]
    fn footprint_parser_is_conservative_and_fails_closed() {
        assert_eq!(
            parse_footprint("phys_footprint: 1.5 MB\n").unwrap(),
            1_572_864
        );
        assert!(parse_footprint("phys_footprint: unknown GB\n").is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_probe_reads_owned_child_and_host() {
        let mut child = Command::new("/bin/sleep").arg("2").spawn().unwrap();
        let mut probe = SystemProbe;
        let deadline = Instant::now() + Duration::from_secs(2);
        let host = probe.host_memory(deadline);
        let owned_child_bytes = probe.child_footprint_bytes(child.id(), deadline);
        let _ = child.kill();
        child.wait().unwrap();
        // These are probe *contents*, not latency assertions; NonZero makes the check explicit
        // without asking the clock-assertion ratchet to infer that distinction from `deadline`.
        // validate() recomputes available from the parsed components, so it is never below free
        // plus speculative.
        let host = host.unwrap();
        host.validate().unwrap();
        std::num::NonZeroU64::new(host.free_and_speculative_bytes())
            .expect("host probe returned zero");
        std::num::NonZeroU64::new(owned_child_bytes.unwrap()).expect("child probe returned zero");
    }

    #[test]
    fn rejects_policy_context_and_preflight_before_spawn() {
        let request = new_request();
        let mut invalid = request.clone();
        invalid.context_tokens = 1001;
        let result = run_guarded(
            &mut Command::new("/usr/bin/false"),
            &invalid,
            &policy(),
            &mut FakeProbe::new([]),
        );
        assert_eq!(result.unwrap_err().reason, StopReason::ContextCeiling);
        let result = run_guarded(
            &mut Command::new("/usr/bin/false"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(299)]),
        );
        assert_eq!(result.unwrap_err().reason, StopReason::PreflightMemory);
        assert!(!request.stdout_path.exists());
    }

    #[test]
    fn deadline_terminates_and_reaps_child() {
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::Deadline);
        assert!(gone(failure.pid.unwrap()));
    }

    #[test]
    fn deadline_escalates_when_child_ignores_term() {
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sh").args(["-c", "trap '' TERM; while :; do :; done"]),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::Deadline);
        assert!(gone(failure.pid.unwrap()));
    }

    #[test]
    fn live_host_decline_and_probe_failure_stop_and_reap() {
        for (samples, expected) in [
            (vec![Ok(1000), Ok(50)], StopReason::HostMemory),
            (
                vec![Ok(1000), Err(io::Error::other("probe failed"))],
                StopReason::ProbeFailure,
            ),
        ] {
            let request = new_request();
            let failure = run_guarded(
                Command::new("/bin/sleep").arg("2"),
                &request,
                &policy(),
                &mut FakeProbe::new(samples),
            )
            .unwrap_err();
            assert_eq!(failure.reason, expected);
            assert!(gone(failure.pid.unwrap()));
        }
    }

    #[test]
    fn footprint_and_log_limits_stop_and_reap() {
        let request = new_request();
        let mut probe = FakeProbe::new([Ok(1000)]);
        probe.footprint.push_back(Ok(201));
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut probe,
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::ChildFootprint);
        assert!(gone(failure.pid.unwrap()));

        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sh").args(["-c", "while :; do printf 1234567890; done"]),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::LogLimit);
        assert!(fs::metadata(&request.stdout_path).unwrap().len() <= policy().stdout_cap_bytes);
        assert!(gone(failure.pid.unwrap()));
    }

    #[test]
    fn reports_success_and_nonzero_exit_with_bounded_logs() {
        let request = new_request();
        let status = run_guarded(
            Command::new("/bin/sh").args(["-c", "printf ok; printf note >&2"]),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap();
        assert!(status.success());
        assert_eq!(fs::read(&request.stdout_path).unwrap(), b"ok");
        assert_eq!(fs::read(&request.stderr_path).unwrap(), b"note");

        let request = new_request();
        let failure = run_guarded(
            &mut Command::new("/usr/bin/false"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::ChildExit, "{}", failure.detail);
        assert!(gone(failure.pid.unwrap()));
    }
}
