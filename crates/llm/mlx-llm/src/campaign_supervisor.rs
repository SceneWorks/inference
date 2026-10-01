//! Bounded, fail-closed child execution for the SC-20671 and SC-20676 campaigns.
//!
//! This module deliberately uses only the standard library, serde and (on macOS, for
//! `proc_pid_rusage`) libc so its process-control tests can run without linking MLX or starting a
//! model. Callers must supply a frozen policy for every run;
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

/// Identity of the macOS admission and host-reserve watchdog measure recorded with every sample.
pub const DARWIN_AVAILABLE_METRIC: &str = "darwin-vm-stat-available-v3";
/// The supervisor hands the worker the host measurement it was admitted on, so the worker's own
/// receipt records the decision's inputs.
pub const HOST_MEMORY_ADMISSION_ENV: &str = "SCENEWORKS_CAMPAIGN_HOST_MEMORY_ADMISSION";
/// Every implementation (Rust, Python, the kv-poc shell precheck) refuses a measure at or above
/// 2^63 bytes, so all three fail closed on the same input.
const AVAILABLE_BYTES_LIMIT: u64 = 1 << 63;

/// The pre-spawn admission rule of every campaign parent (this supervisor and
/// `scripts/media_campaign_supervisor.py`): a unit (row, arm, role, capture, coordinate) starts
/// only when host available memory covers that unit's estimated peak plus the host reserve. The
/// child footprint cap and the live host-reserve watchdog are unchanged by it: a unit that grows
/// past its cap, or eats into the reserve, is still aborted (runtime catching is the tradeoff).
pub const ESTIMATE_PLUS_RESERVE_RULE: &str = "estimate-plus-reserve-v1";
/// Estimate source of a unit with no static estimate: the child footprint cap itself, which makes
/// its admission exactly the former cap-plus-reserve rule.
pub const CAP_FALLBACK_ESTIMATE_SOURCE: &str = "child-footprint-cap-fallback";
/// Estimate source when an already-completed identical unit measured a peak above the static
/// estimate, so a known underestimate is never reused.
pub const MEASURED_PEAK_ESTIMATE_SOURCE: &str = "measured-peak-of-completed-identical-unit";

/// One unit's admission estimate: the bytes the rule adds to the host reserve and where they came
/// from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionEstimate {
    pub source: String,
    pub bytes: u64,
}

impl AdmissionEstimate {
    /// `max(static estimate, measured peak of an already-completed identical unit)`, or the child
    /// footprint cap when the unit has no static estimate. An estimate above the cap refuses the
    /// unit before spawn: the cap watchdog would abort it anyway.
    pub fn resolve(
        static_estimate: Option<(&str, u64)>,
        measured_peak_bytes: Option<u64>,
        cap_bytes: u64,
    ) -> Result<Self, String> {
        let mut estimate = match static_estimate {
            Some((source, bytes)) => Self {
                source: source.into(),
                bytes,
            },
            None => Self {
                source: CAP_FALLBACK_ESTIMATE_SOURCE.into(),
                bytes: cap_bytes,
            },
        };
        if let Some(measured) = measured_peak_bytes.filter(|measured| *measured > estimate.bytes) {
            estimate = Self {
                source: MEASURED_PEAK_ESTIMATE_SOURCE.into(),
                bytes: measured,
            };
        }
        if estimate.bytes > cap_bytes {
            return Err(format!(
                "admission estimate {} bytes ({}) exceeds the child footprint cap {cap_bytes} bytes; refused before spawn",
                estimate.bytes, estimate.source
            ));
        }
        estimate.validate(cap_bytes)?;
        Ok(estimate)
    }

    /// A recorded estimate names its source and lies in `1..=cap`; the cap fallback is the cap.
    pub fn validate(&self, cap_bytes: u64) -> Result<(), String> {
        if self.source.is_empty()
            || self.bytes == 0
            || self.bytes > cap_bytes
            || (self.source == CAP_FALLBACK_ESTIMATE_SOURCE && self.bytes != cap_bytes)
        {
            return Err("admission estimate lacks a source or lies outside 1..=child cap".into());
        }
        Ok(())
    }
}

/// What the supervisor hands its worker in [`HOST_MEMORY_ADMISSION_ENV`]: the rule, the unit's
/// estimate, and the host measurement it was admitted on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostAdmission {
    pub rule: String,
    pub estimate_source: String,
    pub estimate_bytes: u64,
    pub host_memory: HostMemory,
}

impl HostAdmission {
    pub fn estimate(&self) -> AdmissionEstimate {
        AdmissionEstimate {
            source: self.estimate_source.clone(),
            bytes: self.estimate_bytes,
        }
    }

    /// The decision must be this rule's, with a valid estimate, on a measurement covering
    /// estimate plus reserve.
    pub fn validate(&self, reserve_bytes: u64, cap_bytes: u64) -> Result<(), String> {
        if self.rule != ESTIMATE_PLUS_RESERVE_RULE {
            return Err(format!(
                "host admission rule is not {ESTIMATE_PLUS_RESERVE_RULE}"
            ));
        }
        self.estimate().validate(cap_bytes)?;
        self.host_memory
            .validate_admits(reserve_bytes, self.estimate_bytes)
    }
}

/// The raw vm_stat page counters the measure is computed from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmStatPages {
    pub free: u64,
    pub speculative: u64,
    pub purgeable: u64,
    pub inactive: u64,
    pub file_backed: u64,
    pub anonymous: u64,
    pub throttled: u64,
    pub active: u64,
}

/// One vm_stat host-memory measurement and every component of the available measure, used by the
/// pre-spawn admission and the live host-reserve watchdog alike.
///
/// `available_bytes = (free + speculative + purgeable + R) * page_size` (below 2^63), where
/// `R = max(0, file_backed - speculative)`: Activity Monitor's "Cached Files" model. Every
/// file-backed page is page cache the kernel can drop (clean) or write back (dirty) without the
/// compressor or swap; anonymous pages are never credited. vm_stat prints free (the kernel's
/// free_count minus speculative) and speculative as disjoint lists, and `File-backed + Anonymous =
/// active + inactive + speculative + throttled`, so speculative read-ahead sits inside File-backed
/// and is removed from `R`; purgeable pages are anonymous, so they never overlap `R`. Inactive,
/// anonymous, throttled and active pages are parsed (fail closed) and recorded for audit only.
///
/// Known limitation (Activity Monitor semantics): file-backed pages that another process has
/// actively mapped -- executables, another app's mmapped model weights -- count as available
/// although evicting them makes that process fault them back in. The campaign child's own mapped
/// weights are bounded by its `phys_footprint` cap, not by this measure.
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
    pub anonymous_pages: u64,
    pub throttled_pages: u64,
    pub active_pages: u64,
    pub reclaimable_file_pages: u64,
    pub available_bytes: u64,
}

impl HostMemory {
    /// Derive the measure from raw vm_stat page counts; `None` at or above 2^63 bytes.
    pub fn from_pages(page_size_bytes: u64, pages: VmStatPages) -> Option<Self> {
        let reclaimable_file_pages = pages.file_backed.saturating_sub(pages.speculative);
        let available_bytes = pages
            .free
            .checked_add(pages.speculative)?
            .checked_add(pages.purgeable)?
            .checked_add(reclaimable_file_pages)?
            .checked_mul(page_size_bytes)
            .filter(|bytes| *bytes < AVAILABLE_BYTES_LIMIT)?;
        Some(Self {
            metric: DARWIN_AVAILABLE_METRIC.into(),
            page_size_bytes,
            free_pages: pages.free,
            speculative_pages: pages.speculative,
            purgeable_pages: pages.purgeable,
            inactive_pages: pages.inactive,
            file_backed_pages: pages.file_backed,
            anonymous_pages: pages.anonymous,
            throttled_pages: pages.throttled,
            active_pages: pages.active,
            reclaimable_file_pages,
            available_bytes,
        })
    }

    /// The raw counters this measurement was derived from.
    pub fn pages(&self) -> VmStatPages {
        VmStatPages {
            free: self.free_pages,
            speculative: self.speculative_pages,
            purgeable: self.purgeable_pages,
            inactive: self.inactive_pages,
            file_backed: self.file_backed_pages,
            anonymous: self.anonymous_pages,
            throttled: self.throttled_pages,
            active: self.active_pages,
        }
    }

    /// A recorded measurement must name this metric, carry a real page size, and recompute
    /// exactly from its own components.
    pub fn validate(&self) -> Result<(), String> {
        if self.metric != DARWIN_AVAILABLE_METRIC
            || self.page_size_bytes < 4096
            || !self.page_size_bytes.is_power_of_two()
            || Self::from_pages(self.page_size_bytes, self.pages()).as_ref() != Some(self)
        {
            return Err("host memory components do not recompute the admission measure".into());
        }
        Ok(())
    }

    /// An admitted unit's measurement must also cover its estimate plus the reserve
    /// ([`ESTIMATE_PLUS_RESERVE_RULE`]).
    pub fn validate_admits(&self, reserve_bytes: u64, estimate_bytes: u64) -> Result<(), String> {
        self.validate()?;
        if reserve_bytes
            .checked_add(estimate_bytes)
            .is_none_or(|required| self.available_bytes < required)
        {
            return Err("recorded host available memory is below reserve plus estimate".into());
        }
        Ok(())
    }
}

/// Worker side: the decision its supervisor admitted it on, checked against the policy the worker
/// itself loaded. A worker started outside the supervisor has none and refuses.
pub fn admitted_host(reserve_bytes: u64, cap_bytes: u64) -> Result<HostAdmission, String> {
    let raw = std::env::var(HOST_MEMORY_ADMISSION_ENV).map_err(|_| {
        format!("{HOST_MEMORY_ADMISSION_ENV} is unset: the worker was not admitted by the campaign supervisor")
    })?;
    let admission: HostAdmission = serde_json::from_str(&raw)
        .map_err(|error| format!("{HOST_MEMORY_ADMISSION_ENV} is malformed: {error}"))?;
    admission.validate(reserve_bytes, cap_bytes)?;
    Ok(admission)
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
    /// The row's estimated peak, admitted under [`ESTIMATE_PLUS_RESERVE_RULE`].
    pub estimate: AdmissionEstimate,
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
    /// The live sample that tripped the host-reserve watchdog ([`StopReason::HostMemory`]).
    pub watchdog_host_memory: Option<Box<HostMemory>>,
}

impl Failure {
    fn new(reason: StopReason, detail: impl Into<String>, pid: Option<u32>) -> Self {
        Self {
            reason,
            detail: detail.into(),
            pid,
            host_memory: None,
            watchdog_host_memory: None,
        }
    }
}

/// The host probe is separate from MLX allocation counters: a parent process must not report its
/// own MLX counters as though they belonged to the worker PID.
pub trait MemoryProbe {
    /// One host measurement. Pre-spawn admission and the live host-reserve watchdog both read
    /// [`HostMemory::available_bytes`], so page cache alone never aborts an admitted row.
    fn host_memory(&mut self, deadline: Instant) -> io::Result<HostMemory>;
    /// One read of the child's `phys_footprint`. [`io::ErrorKind::NotFound`] means the PID no
    /// longer exists (`ESRCH`: the child is exiting); the supervisor lets its exit handling
    /// classify that instead of reporting a probe failure.
    fn child_footprint_bytes(&mut self, pid: u32, deadline: Instant) -> io::Result<u64>;
}

/// Reads per watchdog tick: one read plus three immediate retries of a failed read.
const CHILD_FOOTPRINT_READS_PER_TICK: u32 = 4;
/// Host reads per preflight or watchdog tick: one read plus three immediate retries.
const HOST_MEMORY_READS_PER_TICK: u32 = 4;
/// Consecutive ticks whose every read of one probe (host or child footprint) failed before the row
/// stops with [`StopReason::ProbeFailure`]. The other watchdog keeps running on the failed ticks.
const PROBE_FAILED_TICK_LIMIT: u32 = 5;

/// One host measurement, retried immediately; the last error once every read failed.
fn read_host_memory<P: MemoryProbe>(probe: &mut P, deadline: Instant) -> io::Result<HostMemory> {
    let mut read = 1;
    loop {
        match probe.host_memory(deadline) {
            Err(_) if read < HOST_MEMORY_READS_PER_TICK => read += 1,
            result => return result,
        }
    }
}

/// One watchdog tick's child footprint: the first successful read, or the last error once a
/// process-gone read or every retry has failed.
fn read_child_footprint<P: MemoryProbe>(
    probe: &mut P,
    pid: u32,
    deadline: Instant,
) -> io::Result<u64> {
    let mut read = 1;
    loop {
        match probe.child_footprint_bytes(pid, deadline) {
            Err(error)
                if error.kind() != io::ErrorKind::NotFound
                    && read < CHILD_FOOTPRINT_READS_PER_TICK =>
            {
                read += 1;
            }
            result => return result,
        }
    }
}

/// An abort detail that keeps the OS errno and its message.
fn describe_probe_error(error: &io::Error) -> String {
    match error.raw_os_error() {
        Some(errno) => format!("errno {errno}: {error}"),
        None => format!("no errno: {error}"),
    }
}

/// A process's Darwin `phys_footprint` ledger, the metric `/usr/bin/footprint` prints as
/// `phys_footprint` / `phys_footprint_peak`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysFootprint {
    pub current_bytes: u64,
    pub lifetime_peak_bytes: u64,
}

/// Read `pid`'s `phys_footprint` with one `proc_pid_rusage(RUSAGE_INFO_V4)` syscall rather than a
/// `/usr/bin/footprint` subprocess. A PID that no longer exists (`ESRCH`) is
/// [`io::ErrorKind::NotFound`]; every other failure keeps its raw OS errno.
#[cfg(target_os = "macos")]
pub fn phys_footprint(pid: u32) -> io::Result<PhysFootprint> {
    let os_pid = libc::pid_t::try_from(pid)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "PID exceeds pid_t"))?;
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: the RUSAGE_INFO_V4 flavor writes at most one `rusage_info_v4` into the buffer, which
    // is exactly that size and zero-initialized.
    let status = unsafe {
        libc::proc_pid_rusage(
            os_pid,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if status != 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("process {pid} not found (ESRCH)"),
            ));
        }
        return Err(error);
    }
    // SAFETY: the successful call initialized the buffer (and it was zeroed before).
    let info = unsafe { info.assume_init() };
    Ok(PhysFootprint {
        current_bytes: info.ri_phys_footprint,
        lifetime_peak_bytes: info.ri_lifetime_max_phys_footprint,
    })
}

/// Poll interval of a settled footprint read.
pub const FOOTPRINT_SETTLE_POLL: Duration = Duration::from_millis(10);
/// Consecutive unchanged reads (250 ms at the poll interval) that make a footprint settled.
pub const FOOTPRINT_SETTLE_STABLE_READS: u32 = 25;
/// Upper bound on the reads of one settled sample (3 s at the poll interval).
pub const FOOTPRINT_SETTLE_MAX_READS: u32 = 300;
/// Footprint change (bytes) below which two reads count as unchanged.
pub const FOOTPRINT_SETTLE_TOLERANCE_BYTES: u64 = 1024 * 1024;

/// `pid`'s `phys_footprint` after it stops changing. Darwin returns memory MLX just freed (Metal
/// buffers) to the process footprint asynchronously: measured on an 0.9 GB free, the footprint
/// holds its pre-free value for 60-105 ms and then falls in steps. `/usr/bin/footprint` took long
/// enough to launch that it always read after the drop; a bare syscall reads before it, which made
/// a released row look 1.1 GB over its weights-loaded boundary (sc-20671 W1 row 2). A settled read
/// polls until the footprint is unchanged for [`FOOTPRINT_SETTLE_STABLE_READS`] consecutive reads,
/// bounded by [`FOOTPRINT_SETTLE_MAX_READS`] (then the latest read is returned).
pub fn settled_phys_footprint(pid: u32) -> io::Result<PhysFootprint> {
    settle_footprint(
        || phys_footprint(pid),
        || thread::sleep(FOOTPRINT_SETTLE_POLL),
        FOOTPRINT_SETTLE_STABLE_READS,
        FOOTPRINT_SETTLE_MAX_READS,
    )
}

fn settle_footprint(
    mut read: impl FnMut() -> io::Result<PhysFootprint>,
    mut wait: impl FnMut(),
    stable_reads: u32,
    max_reads: u32,
) -> io::Result<PhysFootprint> {
    let mut latest = read()?;
    let mut anchor = latest.current_bytes;
    let mut stable = 0;
    for _ in 1..max_reads {
        wait();
        latest = read()?;
        if latest.current_bytes.abs_diff(anchor) <= FOOTPRINT_SETTLE_TOLERANCE_BYTES {
            stable += 1;
            if stable >= stable_reads {
                break;
            }
        } else {
            anchor = latest.current_bytes;
            stable = 0;
        }
    }
    Ok(latest)
}

#[cfg(not(target_os = "macos"))]
pub fn phys_footprint(_pid: u32) -> io::Result<PhysFootprint> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "phys_footprint requires macOS",
    ))
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
        || request
            .estimate
            .validate(policy.child_footprint_cap_bytes)
            .is_err()
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
    let host = read_host_memory(probe, preflight_deadline).map_err(|error| {
        Failure::new(
            StopReason::ProbeFailure,
            format!(
                "host preflight probe failed {HOST_MEMORY_READS_PER_TICK} reads; last error {}",
                describe_probe_error(&error)
            ),
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
    let estimate = &request.estimate;
    // `validate` bounded the estimate by the cap, and reserve plus cap does not overflow.
    let admission = policy.host_free_reserve_bytes + estimate.bytes;
    if available < admission {
        return Err(Failure::new(
            StopReason::PreflightMemory,
            format!(
                "host available {available} bytes is below admission {admission} bytes \
                 ({ESTIMATE_PLUS_RESERVE_RULE}: estimate {} bytes from {} plus reserve {} bytes)",
                estimate.bytes, estimate.source, policy.host_free_reserve_bytes
            ),
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
    let host_json = serde_json::to_string(&HostAdmission {
        rule: ESTIMATE_PLUS_RESERVE_RULE.into(),
        estimate_source: estimate.source.clone(),
        estimate_bytes: estimate.bytes,
        host_memory: host.clone(),
    })
    .map_err(|error| {
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
    let mut failed_footprint_ticks = 0_u32;
    let mut failed_host_ticks = 0_u32;
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
        match read_host_memory(probe, deadline) {
            Ok(sample) => {
                failed_host_ticks = 0;
                if sample.available_bytes < policy.host_free_reserve_bytes {
                    let mut failure = Failure::new(
                        StopReason::HostMemory,
                        format!(
                            "host available {} bytes ({}) fell below reserve {} bytes",
                            sample.available_bytes, sample.metric, policy.host_free_reserve_bytes
                        ),
                        Some(pid),
                    );
                    failure.watchdog_host_memory = Some(Box::new(sample));
                    break Err(failure);
                }
            }
            // The child footprint watchdog keeps running on a failed host tick.
            Err(error) => {
                failed_host_ticks += 1;
                if failed_host_ticks >= PROBE_FAILED_TICK_LIMIT {
                    break Err(Failure::new(
                        StopReason::ProbeFailure,
                        format!(
                            "live host probe failed on {failed_host_ticks} consecutive ticks \
                             ({HOST_MEMORY_READS_PER_TICK} reads each); last error {}",
                            describe_probe_error(&error)
                        ),
                        Some(pid),
                    ));
                }
            }
        }
        match read_child_footprint(probe, pid, deadline) {
            Ok(footprint) => {
                failed_footprint_ticks = 0;
                if footprint > policy.child_footprint_cap_bytes {
                    break Err(Failure::new(
                        StopReason::ChildFootprint,
                        format!("child footprint {footprint} bytes exceeded cap"),
                        Some(pid),
                    ));
                }
            }
            Err(error) => {
                // A child can exit between try_wait and the PID probe. Preserve that exit.
                if let Ok(Some(status)) = child.try_wait() {
                    break Ok(status);
                }
                // ESRCH: the child is exiting but not yet reapable; the next tick's try_wait
                // classifies the exit. Any other error stops the row only once it is sustained.
                if error.kind() != io::ErrorKind::NotFound {
                    failed_footprint_ticks += 1;
                    if failed_footprint_ticks >= PROBE_FAILED_TICK_LIMIT {
                        break Err(Failure::new(
                            StopReason::ProbeFailure,
                            format!(
                                "child footprint probe failed on {failed_footprint_ticks} \
                                 consecutive ticks ({CHILD_FOOTPRINT_READS_PER_TICK} reads each); \
                                 last error {}",
                                describe_probe_error(&error)
                            ),
                            Some(pid),
                        ));
                    }
                }
            }
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

/// Production probe. Pre-spawn admission and the live reserve watchdog both use
/// [`HostMemory::available_bytes`] (free, speculative, purgeable, and inactive clean file cache);
/// compressed and anonymous inactive pages are never counted. The child cap stays
/// `phys_footprint`.
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
        VmStatPages {
            free: pages("Pages free")?,
            speculative: pages("Pages speculative")?,
            purgeable: pages("Pages purgeable")?,
            inactive: pages("Pages inactive")?,
            file_backed: pages("File-backed pages")?,
            anonymous: pages("Anonymous pages")?,
            throttled: pages("Pages throttled")?,
            active: pages("Pages active")?,
        },
    )
    .ok_or_else(|| io::Error::other("vm_stat available bytes reach 2^63"))
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
        return Err(io::Error::other(format!("{command} exited with {status}")));
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

    fn child_footprint_bytes(&mut self, pid: u32, _deadline: Instant) -> io::Result<u64> {
        // One non-blocking syscall per read; no sampler process can stall past the deadline.
        phys_footprint(pid).map(|footprint| footprint.current_bytes)
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
        /// Answers every footprint read once `footprint` is drained.
        footprint_after: Box<dyn FnMut() -> io::Result<u64>>,
        footprint_reads: usize,
        /// Answers every host read once `host` is drained.
        host_after: Box<dyn FnMut() -> io::Result<HostMemory>>,
        host_reads: usize,
    }

    /// A one-byte-page measurement whose free pages are the whole available measure.
    fn free_only(bytes: u64) -> HostMemory {
        HostMemory::from_pages(
            1,
            VmStatPages {
                free: bytes,
                ..VmStatPages::default()
            },
        )
        .unwrap()
    }

    impl FakeProbe {
        fn new(free: impl IntoIterator<Item = io::Result<u64>>) -> Self {
            Self::hosts(free.into_iter().map(|sample| sample.map(free_only)))
        }

        fn hosts(host: impl IntoIterator<Item = io::Result<HostMemory>>) -> Self {
            Self {
                host: host.into_iter().collect(),
                footprint: VecDeque::new(),
                footprint_after: Box::new(|| Ok(1)),
                footprint_reads: 0,
                host_after: Box::new(|| Ok(free_only(1000))),
                host_reads: 0,
            }
        }
    }

    impl MemoryProbe for FakeProbe {
        fn host_memory(&mut self, _: Instant) -> io::Result<HostMemory> {
            self.host_reads += 1;
            self.host.pop_front().unwrap_or_else(|| (self.host_after)())
        }
        fn child_footprint_bytes(&mut self, _: u32, _: Instant) -> io::Result<u64> {
            self.footprint_reads += 1;
            self.footprint
                .pop_front()
                .unwrap_or_else(|| (self.footprint_after)())
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
            // No static estimate: admitted against the cap (200) plus reserve (100).
            estimate: AdmissionEstimate::resolve(None, None, 200).unwrap(),
            stdout_path: root.join("stdout.log"),
            stderr_path: root.join("stderr.log"),
        }
    }

    /// A request whose unit has a static estimate of `bytes`.
    fn estimated_request(bytes: u64) -> RunRequest {
        RunRequest {
            estimate: AdmissionEstimate::resolve(Some(("test-static-estimate", bytes)), None, 200)
                .unwrap(),
            ..new_request()
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
                    // File-backed + Anonymous = active + inactive + speculative + throttled: the
                    // identity the anonymous bound rests on (exact on the real snapshots).
                    let counter = |key: &str| -> u64 {
                        text.lines()
                            .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))
                            .unwrap()
                            .trim()
                            .trim_end_matches('.')
                            .parse()
                            .unwrap()
                    };
                    assert_eq!(
                        counter("File-backed pages") + counter("Anonymous pages"),
                        counter("Pages active")
                            + counter("Pages inactive")
                            + counter("Pages speculative")
                            + counter("Pages throttled"),
                        "{name}"
                    );
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
            ("count anonymous instead of file-backed", |h| {
                h.anonymous_pages
            }),
            ("count inactive instead of file-backed", |h| {
                h.inactive_pages
            }),
            ("min with inactive (v1)", |h| {
                h.inactive_pages
                    .saturating_sub(h.purgeable_pages)
                    .min(h.file_backed_pages.saturating_sub(h.speculative_pages))
            }),
            ("anonymous bound (v2)", |h| {
                h.inactive_pages
                    .saturating_sub(h.purgeable_pages)
                    .min(h.file_backed_pages.saturating_sub(h.speculative_pages))
                    .min((h.inactive_pages + h.throttled_pages).saturating_sub(h.anonymous_pages))
            }),
            ("speculative counted twice", |h| h.file_backed_pages),
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
        host.validate_admits(1, host.available_bytes - 1).unwrap();
        assert!(host.validate_admits(1, host.available_bytes).is_err());
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
    fn admission_and_watchdog_count_reclaimable_file_cache() {
        // 10 free + 290 inactive file-backed pages: available 300 covers cap 200 + reserve 100
        // and stays above the reserve on every live sample, although free plus speculative (10)
        // is under it throughout. Page cache alone must not abort the row.
        let file_cache = |pages| {
            HostMemory::from_pages(
                1,
                VmStatPages {
                    free: 10,
                    inactive: pages,
                    file_backed: pages,
                    ..VmStatPages::default()
                },
            )
            .unwrap()
        };
        let cached = file_cache(290);
        let request = new_request();
        let status = run_guarded(
            Command::new("/bin/sleep").arg("0.1"),
            &request,
            &policy(),
            &mut FakeProbe::hosts((0..1000).map(|_| Ok(cached.clone()))),
        )
        .unwrap();
        assert!(status.success());
        // A live sample whose available measure is under the reserve aborts, naming the metric.
        let short = file_cache(89);
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::hosts([Ok(cached.clone()), Ok(short.clone())]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::HostMemory, "{}", failure.detail);
        assert!(
            failure.detail.contains("host available 99 bytes"),
            "{}",
            failure.detail
        );
        assert!(
            failure.detail.contains(DARWIN_AVAILABLE_METRIC),
            "{}",
            failure.detail
        );
        assert_eq!(failure.host_memory.as_deref(), Some(&cached));
        assert_eq!(failure.watchdog_host_memory.as_deref(), Some(&short));
        assert!(gone(failure.pid.unwrap()));
        // The same 290 inactive pages as anonymous memory are not credited.
        let anonymous = HostMemory::from_pages(
            1,
            VmStatPages {
                free: 10,
                inactive: 290,
                file_backed: 5,
                anonymous: 290,
                ..VmStatPages::default()
            },
        )
        .unwrap();
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::hosts([Ok(anonymous.clone())]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::PreflightMemory);
        // Only its 5 file-backed pages are credited, never the 290 anonymous inactive pages.
        assert!(failure.detail.contains("host available 15 bytes"));
        assert_eq!(failure.host_memory.as_deref(), Some(&anonymous));
        assert!(failure.pid.is_none() && !request.stdout_path.exists());
    }

    #[test]
    fn worker_receives_the_decision_it_was_admitted_on() {
        let host = HostMemory::from_pages(
            4096,
            VmStatPages {
                free: 1,
                ..VmStatPages::default()
            },
        )
        .unwrap();
        let request = estimated_request(150);
        let decision = HostAdmission {
            rule: ESTIMATE_PLUS_RESERVE_RULE.into(),
            estimate_source: "test-static-estimate".into(),
            estimate_bytes: 150,
            host_memory: host.clone(),
        };
        let expected = serde_json::to_string(&decision).unwrap();
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
        let decoded: HostAdmission = serde_json::from_str(&expected).unwrap();
        assert_eq!(decoded, decision);
        // 4096 available covers estimate 150 + reserve 3946 exactly, not one byte more.
        decoded.validate(3946, 200).unwrap();
        assert!(decoded.validate(3947, 200).is_err());
        // A decision under another rule, without an estimate source, with an estimate above the
        // worker's cap, or claiming the cap fallback below the cap is refused.
        for tampered in [
            HostAdmission {
                rule: "cap-plus-reserve".into(),
                ..decision.clone()
            },
            HostAdmission {
                estimate_source: String::new(),
                ..decision.clone()
            },
            HostAdmission {
                estimate_source: CAP_FALLBACK_ESTIMATE_SOURCE.into(),
                ..decision.clone()
            },
            HostAdmission {
                estimate_bytes: 0,
                ..decision.clone()
            },
        ] {
            assert!(tampered.validate(100, 200).is_err(), "{tampered:?}");
        }
        assert!(decision.validate(100, 149).is_err());
        // The legacy payload (a bare measurement, no estimate) no longer decodes.
        assert!(
            serde_json::from_str::<HostAdmission>(&serde_json::to_string(&host).unwrap()).is_err()
        );
    }

    #[test]
    fn a_unit_is_admitted_on_its_estimate_plus_reserve() {
        // Estimate 50 + reserve 100: admitted at exactly 150 available, refused at 149, although
        // both are far below the former cap (200) plus reserve.
        let request = estimated_request(50);
        let status = run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(150)]),
        )
        .unwrap();
        assert!(status.success());
        let request = estimated_request(50);
        let failure = run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(149)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::PreflightMemory);
        assert!(
            failure.detail.contains(ESTIMATE_PLUS_RESERVE_RULE)
                && failure.detail.contains("test-static-estimate"),
            "{}",
            failure.detail
        );
        assert!(failure.pid.is_none() && !request.stdout_path.exists());
    }

    #[test]
    fn a_unit_without_an_estimate_falls_back_to_the_cap() {
        let fallback = AdmissionEstimate::resolve(None, None, 200).unwrap();
        assert_eq!(
            fallback,
            AdmissionEstimate {
                source: CAP_FALLBACK_ESTIMATE_SOURCE.into(),
                bytes: 200,
            }
        );
        let request = new_request();
        assert_eq!(request.estimate, fallback);
        let failure = run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(299)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::PreflightMemory);
        assert!(failure.detail.contains(CAP_FALLBACK_ESTIMATE_SOURCE));
        let request = new_request();
        assert!(run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(300)]),
        )
        .unwrap()
        .success());
    }

    #[test]
    fn the_estimate_is_the_larger_of_static_and_a_measured_identical_unit() {
        let resolve = |measured| AdmissionEstimate::resolve(Some(("static", 50)), measured, 200);
        assert_eq!(resolve(None).unwrap().bytes, 50);
        assert_eq!(resolve(Some(40)).unwrap().source, "static");
        assert_eq!(
            resolve(Some(60)).unwrap(),
            AdmissionEstimate {
                source: MEASURED_PEAK_ESTIMATE_SOURCE.into(),
                bytes: 60,
            }
        );
        // A measured peak never lowers the cap fallback.
        assert_eq!(
            AdmissionEstimate::resolve(None, Some(60), 200)
                .unwrap()
                .bytes,
            200
        );
        // An estimate above the cap is refused before spawn; a zero one is malformed.
        assert!(resolve(Some(201)).is_err());
        assert!(AdmissionEstimate::resolve(Some(("static", 201)), None, 200).is_err());
        assert!(AdmissionEstimate::resolve(Some(("static", 0)), None, 200).is_err());
        assert!(AdmissionEstimate::resolve(Some(("", 50)), None, 200).is_err());
        // The supervisor refuses a request carrying an out-of-range estimate as invalid policy.
        let mut request = new_request();
        request.estimate.bytes = 201;
        let failure = run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::InvalidPolicy);
    }

    #[test]
    fn an_estimate_admitted_unit_is_still_aborted_at_its_cap_and_reserve() {
        // Admitted on estimate 50 (150 available), the child then grows past the 200 cap: the
        // unchanged footprint watchdog aborts and reaps it.
        let request = estimated_request(50);
        let mut probe = FakeProbe::new([Ok(150)]);
        probe.footprint_after = Box::new(|| Ok(201));
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut probe,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            StopReason::ChildFootprint,
            "{}",
            failure.detail
        );
        assert!(gone(failure.pid.unwrap()));
        // A child at exactly its cap keeps running.
        let request = estimated_request(50);
        let mut probe = FakeProbe::new([Ok(150)]);
        probe.footprint_after = Box::new(|| Ok(200));
        assert!(run_guarded(
            Command::new("/bin/sleep").arg("0.05"),
            &request,
            &policy(),
            &mut probe,
        )
        .unwrap()
        .success());
        // Admitted on its estimate, the host then falls under the reserve: the unchanged
        // host-reserve watchdog aborts it.
        let request = estimated_request(50);
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(150), Ok(99)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::HostMemory, "{}", failure.detail);
        assert!(gone(failure.pid.unwrap()));
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
        let host = host.unwrap();
        host.validate().unwrap();
        std::num::NonZeroU64::new(host.available_bytes).expect("host probe returned zero");
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
        let request = new_request();
        let failure = run_guarded(
            Command::new("/bin/sleep").arg("2"),
            &request,
            &policy(),
            &mut FakeProbe::new([Ok(1000), Ok(50)]),
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::HostMemory);
        assert!(gone(failure.pid.unwrap()));

        let request = new_request();
        let mut probe = FakeProbe::new([Ok(1000)]);
        probe.host_after = Box::new(|| Err(io::Error::from_raw_os_error(35)));
        let failure = run_guarded(
            &mut until_exists(&stop_path(&request)),
            &request,
            &patient_policy(),
            &mut probe,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            StopReason::ProbeFailure,
            "{}",
            failure.detail
        );
        for part in [
            "live host probe failed on 5 consecutive ticks (4 reads each)",
            "errno 35: ",
        ] {
            assert!(failure.detail.contains(part), "{}", failure.detail);
        }
        assert_eq!(probe.host_reads, 1 + 20);
        assert!(gone(failure.pid.unwrap()));
    }

    #[test]
    fn host_preflight_retries_then_refuses_with_the_error() {
        let request = new_request();
        let mut probe = FakeProbe::new([
            Err(io::Error::from_raw_os_error(35)),
            Err(io::Error::from_raw_os_error(35)),
            Err(io::Error::from_raw_os_error(35)),
            Ok(1000),
        ]);
        run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut probe,
        )
        .unwrap_or_else(|failure| panic!("{:?}: {}", failure.reason, failure.detail));

        let request = new_request();
        let mut probe = FakeProbe::new([]);
        probe.host_after = Box::new(|| Err(io::Error::from_raw_os_error(35)));
        let failure = run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request,
            &policy(),
            &mut probe,
        )
        .unwrap_err();
        assert_eq!(failure.reason, StopReason::ProbeFailure);
        assert!(
            failure
                .detail
                .contains("host preflight probe failed 4 reads; last error errno 35: "),
            "{}",
            failure.detail
        );
        assert_eq!(probe.host_reads, 4);
        assert!(failure.pid.is_none() && !request.stdout_path.exists());
    }

    #[test]
    fn transient_host_failures_never_stop_a_row() {
        // Four fully failed ticks, one good read, four more failed ticks: never five in a row.
        let request = new_request();
        let stop = stop_path(&request);
        let eagain = || Err(io::Error::from_raw_os_error(35));
        let mut host = vec![Ok(free_only(1000))];
        host.extend((0..16).map(|_| eagain()));
        host.push(Ok(free_only(1000)));
        host.extend((0..16).map(|_| eagain()));
        let mut probe = FakeProbe::hosts(host);
        let flag = stop.clone();
        probe.host_after = Box::new(move || {
            fs::write(&flag, b"")?;
            Ok(free_only(1000))
        });
        let status = run_guarded(
            &mut until_exists(&stop),
            &request,
            &patient_policy(),
            &mut probe,
        )
        .unwrap_or_else(|failure| panic!("{:?}: {}", failure.reason, failure.detail));
        assert!(status.success());
        assert!(probe.host_reads > 34);
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

    /// A deadline no test below reaches: each is bounded by probe reads, never the clock.
    fn patient_policy() -> SafetyPolicy {
        SafetyPolicy {
            deadline: Duration::from_secs(30),
            ..policy()
        }
    }

    fn eperm() -> io::Result<u64> {
        Err(io::Error::from_raw_os_error(1))
    }

    fn pid_gone() -> io::Result<u64> {
        Err(io::Error::new(io::ErrorKind::NotFound, "ESRCH"))
    }

    fn read_tick(probe: &mut FakeProbe) -> io::Result<u64> {
        read_child_footprint(probe, 1, Instant::now())
    }

    /// A child that runs until `stop` exists, so a test decides when it exits.
    fn until_exists(stop: &Path) -> Command {
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "while [ ! -e \"$0\" ]; do sleep 0.01; done",
            stop.to_str().unwrap(),
        ]);
        command
    }

    fn stop_path(request: &RunRequest) -> PathBuf {
        request.stdout_path.with_file_name("stop")
    }

    #[test]
    fn a_tick_retries_a_failed_read_but_not_a_gone_pid() {
        let mut probe = FakeProbe::new([]);
        probe.footprint.extend([eperm(), eperm(), eperm(), Ok(7)]);
        assert_eq!(read_tick(&mut probe).unwrap(), 7);
        assert_eq!(probe.footprint_reads, 4);

        let mut probe = FakeProbe::new([]);
        probe.footprint_after = Box::new(eperm);
        assert_eq!(read_tick(&mut probe).unwrap_err().raw_os_error(), Some(1));
        assert_eq!(probe.footprint_reads, 4);

        let mut probe = FakeProbe::new([]);
        probe.footprint_after = Box::new(pid_gone);
        assert_eq!(
            read_tick(&mut probe).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(probe.footprint_reads, 1);
    }

    #[test]
    fn transient_footprint_failures_never_stop_a_row() {
        // Four fully failed ticks, one good read, four more failed ticks: never five in a row.
        let request = new_request();
        let stop = stop_path(&request);
        let mut probe = FakeProbe::new([Ok(1000)]);
        probe.footprint.extend((0..16).map(|_| eperm()));
        probe.footprint.push_back(Ok(1));
        probe.footprint.extend((0..16).map(|_| eperm()));
        let flag = stop.clone();
        probe.footprint_after = Box::new(move || {
            fs::write(&flag, b"")?;
            Ok(1)
        });
        let status = run_guarded(
            &mut until_exists(&stop),
            &request,
            &patient_policy(),
            &mut probe,
        )
        .unwrap_or_else(|failure| panic!("{:?}: {}", failure.reason, failure.detail));
        assert!(status.success());
        assert!(probe.footprint_reads > 33);
    }

    #[test]
    fn a_gone_pid_is_left_to_exit_handling() {
        // ESRCH on every read: the child is exiting, so the row ends with the child's own exit.
        let request = new_request();
        let stop = stop_path(&request);
        let mut probe = FakeProbe::new([Ok(1000)]);
        let flag = stop.clone();
        let mut reads = 0;
        probe.footprint_after = Box::new(move || {
            reads += 1;
            if reads == 3 * PROBE_FAILED_TICK_LIMIT {
                fs::write(&flag, b"")?;
            }
            pid_gone()
        });
        let status = run_guarded(
            &mut until_exists(&stop),
            &request,
            &patient_policy(),
            &mut probe,
        )
        .unwrap_or_else(|failure| panic!("{:?}: {}", failure.reason, failure.detail));
        assert!(status.success());
        assert!(probe.footprint_reads >= 3 * PROBE_FAILED_TICK_LIMIT as usize);
    }

    #[test]
    fn sustained_footprint_failure_stops_the_row_with_errno_and_count() {
        let request = new_request();
        let mut probe = FakeProbe::new([Ok(1000)]);
        probe.footprint_after = Box::new(eperm);
        let failure = run_guarded(
            &mut until_exists(&stop_path(&request)),
            &request,
            &patient_policy(),
            &mut probe,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            StopReason::ProbeFailure,
            "{}",
            failure.detail
        );
        for part in [
            "failed on 5 consecutive ticks (4 reads each)",
            "errno 1: ",
            "Operation not permitted",
        ] {
            assert!(failure.detail.contains(part), "{}", failure.detail);
        }
        assert_eq!(probe.footprint_reads, 20);
        assert!(gone(failure.pid.unwrap()));
    }

    /// A settled read waits out a delayed staircase release and returns the settled value;
    /// the read count is bounded.
    #[test]
    fn settled_footprint_waits_for_an_asynchronous_release() {
        let footprint = |current_bytes| PhysFootprint {
            current_bytes,
            lifetime_peak_bytes: 1_000_000_000,
        };
        const MB: u64 = 1024 * 1024;
        // 6 reads at the pre-free value, a two-step drop, then flat.
        let trace = [900, 900, 900, 900, 900, 900, 650, 130]
            .iter()
            .map(|mb| mb * MB)
            .collect::<Vec<_>>();
        let mut reads = 0_usize;
        let settled = settle_footprint(
            || {
                reads += 1;
                Ok(footprint(*trace.get(reads - 1).unwrap_or(&(130 * MB))))
            },
            || {},
            10,
            100,
        )
        .unwrap();
        assert_eq!(settled.current_bytes, 130 * MB);
        assert_eq!(reads, trace.len() + 10);
        // A footprint that never stops moving returns its latest read at the bound.
        let mut moving = 0_u64;
        let bounded = settle_footprint(
            || {
                moving += 2 * MB;
                Ok(footprint(moving))
            },
            || {},
            10,
            50,
        )
        .unwrap();
        assert_eq!(bounded.current_bytes, 100 * MB);
        // Sub-tolerance jitter counts as settled.
        let mut jitter = 0_u64;
        let steady = settle_footprint(
            || {
                jitter += 1;
                Ok(footprint(500 * MB + (jitter % 2) * 4096))
            },
            || {},
            5,
            100,
        )
        .unwrap();
        assert!(steady.current_bytes >= 500 * MB);
        assert_eq!(jitter, 6);
    }

    /// Real hardware: memory MLX frees is visible in a settled read, not an immediate one.
    #[cfg(target_os = "macos")]
    #[test]
    fn settled_footprint_sees_freed_mlx_memory_leave_the_process() {
        let pid = std::process::id();
        let before = settled_phys_footprint(pid).unwrap().current_bytes;
        {
            let block = mlx_rs::Array::ones::<f32>(&[64 * 1024 * 1024]).unwrap();
            let doubled = block.multiply(mlx_rs::Array::from_f32(2.0)).unwrap();
            doubled.eval().unwrap();
        }
        mlx_rs::memory::clear_cache();
        let settled = settled_phys_footprint(pid).unwrap().current_bytes;
        // 512 MiB was allocated and freed; the settled read is back near the pre-allocation
        // footprint (well under half the allocation above it).
        assert!(
            settled < before + 256 * 1024 * 1024,
            "freed MLX memory still counted: before={before} settled={settled}"
        );
    }

    /// The syscall reads the same ledger `/usr/bin/footprint` prints, for a child holding 64 MiB.
    #[cfg(target_os = "macos")]
    #[test]
    fn phys_footprint_reads_a_child_allocation_and_a_reaped_pid_is_gone() {
        const BLOCK: u64 = 64 << 20;
        // dd reads one 64 MiB block from /dev/zero, then blocks writing it into an undrained pipe.
        let mut dd = Command::new("/bin/dd")
            .args(["if=/dev/zero", "bs=64m", "count=1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut first = [0_u8; 1];
        // The block is fully read (resident) before its first byte can reach the pipe.
        dd.stdout.as_mut().unwrap().read_exact(&mut first).unwrap();
        let pid = dd.id();
        let before = phys_footprint(pid).unwrap();
        let tool = Command::new("/usr/bin/footprint")
            .args(["-p", &pid.to_string(), "-f", "bytes"])
            .output()
            .unwrap();
        let after = phys_footprint(pid).unwrap();
        drop(dd.stdout.take());
        let _ = dd.kill();
        dd.wait().unwrap();
        assert!(
            (BLOCK..4 * BLOCK).contains(&before.current_bytes),
            "{before:?}"
        );
        assert!(before.lifetime_peak_bytes >= before.current_bytes);
        assert_eq!(before, after, "dd is blocked, so its footprint is stable");
        let text = String::from_utf8(tool.stdout).unwrap();
        let field = |name: &str| -> u64 {
            text.lines()
                .find_map(|line| line.trim().strip_prefix(name))
                .and_then(|value| value.trim().strip_suffix(" B"))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("footprint output lacks {name}: {text}"))
        };
        assert_eq!(field("phys_footprint:"), before.current_bytes);
        assert_eq!(field("phys_footprint_peak:"), before.lifetime_peak_bytes);
        assert_eq!(
            phys_footprint(pid).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }
}
