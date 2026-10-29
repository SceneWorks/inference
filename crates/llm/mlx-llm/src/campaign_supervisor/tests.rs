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
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../testdata/darwin-host-memory-cases.json"
    ))
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
    let mut success_policy = policy();
    success_policy.deadline = Duration::from_secs(2);
    let status = run_guarded(
        Command::new("/bin/sleep").arg("0.1"),
        &request,
        &success_policy,
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
    assert!(serde_json::from_str::<HostAdmission>(&serde_json::to_string(&host).unwrap()).is_err());
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
