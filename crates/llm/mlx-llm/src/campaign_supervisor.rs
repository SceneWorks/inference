//! Bounded, fail-closed child execution for the SC-20671 and SC-20676 campaigns.
//!
//! This module deliberately uses only the standard library so its process-control tests can run
//! without linking MLX or starting a model. Callers must supply a frozen policy for every run;
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
}

impl Failure {
    fn new(reason: StopReason, detail: impl Into<String>, pid: Option<u32>) -> Self {
        Self {
            reason,
            detail: detail.into(),
            pid,
        }
    }
}

/// The host probe is separate from MLX allocation counters: a parent process must not report its
/// own MLX counters as though they belonged to the worker PID.
pub trait MemoryProbe {
    fn host_free_bytes(&mut self, deadline: Instant) -> io::Result<u64>;
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
    let available = probe.host_free_bytes(preflight_deadline).map_err(|error| {
        Failure::new(
            StopReason::ProbeFailure,
            format!("host preflight probe: {error}"),
            None,
        )
    })?;
    let admission = policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes;
    if available < admission {
        return Err(Failure::new(
            StopReason::PreflightMemory,
            format!("host free {available} bytes is below admission {admission} bytes"),
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
        let free = match probe.host_free_bytes(deadline) {
            Ok(bytes) => bytes,
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

/// Production probe. The reserve uses only free and speculative pages; inactive, compressed, and
/// purgeable pages are excluded because reclaimability is not guaranteed at a safe latency.
pub struct SystemProbe;

#[cfg(any(target_os = "macos", all(test, unix)))]
fn parse_vm_stat(text: &str) -> io::Result<u64> {
    let page_size = text
        .lines()
        .next()
        .and_then(|line| line.split("page size of ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value >= 4096 && value.is_power_of_two())
        .ok_or_else(|| io::Error::other("vm_stat has no valid page size"))?;
    let pages = |key: &str| -> io::Result<u64> {
        let mut matching = text.lines().filter_map(|line| {
            line.trim()
                .strip_prefix(key)
                .and_then(|tail| tail.trim().strip_prefix(':'))
        });
        let value = matching
            .next()
            .and_then(|tail| tail.trim().trim_end_matches('.').parse::<u64>().ok())
            .ok_or_else(|| io::Error::other(format!("vm_stat has no valid {key}")))?;
        if matching.next().is_some() {
            return Err(io::Error::other(format!("vm_stat duplicates {key}")));
        }
        Ok(value)
    };
    pages("Pages free")?
        .checked_add(pages("Pages speculative")?)
        .and_then(|count| count.checked_mul(page_size))
        .ok_or_else(|| io::Error::other("vm_stat free-byte count overflows"))
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
    fn host_free_bytes(&mut self, deadline: Instant) -> io::Result<u64> {
        #[cfg(target_os = "macos")]
        {
            return parse_vm_stat(&bounded_system_output("/usr/bin/vm_stat", &[], deadline)?);
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
            return parse_footprint(&bounded_system_output(
                "/usr/bin/footprint",
                &["--pid", &pid.to_string(), "--noCategories", "--wired"],
                deadline,
            )?);
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
        free: VecDeque<io::Result<u64>>,
        footprint: VecDeque<io::Result<u64>>,
    }

    impl FakeProbe {
        fn new(free: impl IntoIterator<Item = io::Result<u64>>) -> Self {
            Self {
                free: free.into_iter().collect(),
                footprint: VecDeque::new(),
            }
        }
    }

    impl MemoryProbe for FakeProbe {
        fn host_free_bytes(&mut self, _: Instant) -> io::Result<u64> {
            self.free.pop_front().unwrap_or(Ok(1000))
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

    #[test]
    fn parsers_are_conservative_and_fail_closed() {
        let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 1000.\nPages speculative: 5.\n";
        assert_eq!(parse_vm_stat(vm).unwrap(), 105 * 16384);
        assert!(parse_vm_stat("Pages free: 100.\n").is_err());
        assert!(parse_vm_stat(&(vm.to_owned() + "Pages free: 1.\n")).is_err());
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
        let free = probe.host_free_bytes(deadline);
        let footprint = probe.child_footprint_bytes(child.id(), deadline);
        let _ = child.kill();
        child.wait().unwrap();
        assert!(free.unwrap() > 0);
        assert!(footprint.unwrap() > 0);
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
