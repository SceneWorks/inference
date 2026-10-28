"""CPU-only fail-closed process supervision tests; no model or GPU is started."""

import dataclasses
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import media_campaign_supervisor as safety


class FixtureDeadlineClock:
    """Test-only: arm the teardown budget after bounded positive fixture readiness."""
    def __init__(self, now=time.monotonic, pause=time.sleep):
        self.now, self.pause, self.ready_at = now, pause, None

    def __call__(self):
        return 0.0 if self.ready_at is None else self.now() - self.ready_at

    def admit(self, ready, timeout=30.0):
        until = self.now() + timeout
        while not ready():
            if self.now() >= until:
                raise AssertionError("fixture startup readiness deadline exceeded")
            self.pause(.01)
        self.ready_at = self.now()


def published_fixture_pid(path):
    """Read only the closed/renamed PID record, within bounded bootstrap polling."""
    try:
        published = path.read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError):
        # Atomic rename prevents partial publication, but a Windows sharing/access
        # denial can still make the published name temporarily unreadable. This is
        # not readiness; the caller's existing bootstrap deadline remains in force.
        return None
    if not published.isdecimal() or int(published) <= 0:
        raise AssertionError("malformed published fixture PID")
    return int(published)


class FixtureDeadlineClockTests(unittest.TestCase):
    def test_bootstrap_is_bounded_separately_from_three_second_teardown_budget(self):
        now = [0.0]
        clock = FixtureDeadlineClock(lambda: now[0], lambda delay: now.__setitem__(0, now[0] + delay))
        self.assertEqual(clock(), 0)
        now[0] = 7
        self.assertEqual(clock(), 0)  # Slow bootstrap has not armed the teardown clock.
        clock.admit(lambda: now[0] >= 10)
        self.assertEqual(clock(), 0)
        now[0] += 3
        self.assertAlmostEqual(clock(), 3)

    def test_missing_readiness_fails_closed_without_arming_teardown_clock(self):
        now = [0.0]
        clock = FixtureDeadlineClock(lambda: now[0], lambda delay: now.__setitem__(0, now[0] + delay))
        with self.assertRaisesRegex(AssertionError, "fixture startup readiness deadline"):
            clock.admit(lambda: False, timeout=.05)
        self.assertIsNone(clock.ready_at)
        self.assertLess(now[0], .1)

    def test_pid_publication_read_denial_retries_within_bootstrap_bound(self):
        now = [0.0]
        clock = FixtureDeadlineClock(lambda: now[0], lambda delay: now.__setitem__(0, now[0] + delay))
        with patch.object(Path, "read_text", side_effect=[PermissionError(13, "denied"), "123"]):
            clock.admit(lambda: published_fixture_pid(Path("pid")) == 123, timeout=.05)
        self.assertAlmostEqual(clock.ready_at, .01)
        with patch.object(Path, "read_text", side_effect=PermissionError(13, "denied")):
            with self.assertRaisesRegex(AssertionError, "fixture startup readiness deadline"):
                clock.admit(lambda: published_fixture_pid(Path("pid")) == 123, timeout=.05)


class PublishedFixturePidTests(unittest.TestCase):
    def clock(self):
        now = [0.0]
        return FixtureDeadlineClock(lambda: now[0], lambda delay: now.__setitem__(0, now[0] + delay))

    def test_access_denial_is_not_ready_until_a_valid_record_is_read(self):
        clock, observed = self.clock(), []
        def ready():
            pid = published_fixture_pid(Path("grandchild-pid"))
            if pid is not None:
                observed.append(pid)
            return pid is not None
        with patch.object(Path, "read_text", side_effect=[PermissionError(13, "sharing denial"), "1234"]) as read:
            clock.admit(ready)
        self.assertEqual(observed, [1234])
        self.assertEqual(read.call_count, 2)
        self.assertIsNotNone(clock.ready_at)

    def test_permanent_access_denial_fails_at_the_existing_bootstrap_deadline(self):
        clock = self.clock()
        with patch.object(Path, "read_text", side_effect=PermissionError(13, "persistent denial")), \
             self.assertRaisesRegex(AssertionError, "fixture startup readiness deadline"):
            clock.admit(lambda: published_fixture_pid(Path("grandchild-pid")) is not None, timeout=.05)
        self.assertIsNone(clock.ready_at)

    def test_partial_temporary_record_is_never_used_as_published_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grandchild-pid"
            temporary = path.with_suffix(".tmp")
            temporary.write_text("12", encoding="ascii")
            self.assertIsNone(published_fixture_pid(path))
            temporary.write_text("1234", encoding="ascii")
            os.replace(temporary, path)
            self.assertEqual(published_fixture_pid(path), 1234)

    def test_malformed_published_records_and_unrelated_io_errors_remain_terminal(self):
        for record in ("", "0", "-1", "123partial", "123\n"):
            with self.subTest(record=record), patch.object(Path, "read_text", return_value=record), \
                 self.assertRaisesRegex(AssertionError, "malformed published fixture PID"):
                published_fixture_pid(Path("grandchild-pid"))
        with patch.object(Path, "read_text", side_effect=OSError(5, "I/O failure")), self.assertRaises(OSError):
            published_fixture_pid(Path("grandchild-pid"))

    def test_reaping_assertions_use_only_the_validated_invocation_snapshot(self):
        case = WindowsSupervisorTests()
        case.root, case.sampled = Path("fixture"), [1]
        case.data = {"deadlineSeconds": 3}
        case._has_exited = lambda pid: pid == 1234
        observed = []
        case._assert_exited = observed.append
        def run(script, *, on_spawn, clock):
            # Simulate the real Probe's real-clock membership sampling: the supervisor observed the
            # live grandchild as owned (the root is already gone) before the deadline fired.
            case.supervision_start = 0.0
            case.membership_samples = [(1.0, {5678})]
            on_spawn(1234, None)
            raise safety.SupervisionError("deadline", "fixture deadline")
        case._run = run
        with patch.object(Path, "read_text", side_effect=["1234", "5678", AssertionError("reopened PID publication")]) as read:
            case._deadline_reaps_root_and_grandchild_after_root_exit()
        self.assertEqual(read.call_count, 2)
        self.assertEqual(observed, [1234, 5678])


class Probe:
    def __init__(self, *, free=10**12, footprint=1024, gpu_free=10**12):
        self.free = free
        self.footprint = footprint
        self.gpu = gpu_free

    def host_free(self):
        return self.free

    def host_admission(self):
        # Like SystemProbe: the admission measure is the components' availableBytes (whole pages).
        host = safety.darwin_host_memory(4096, {
            "freePages": self.host_free() // 4096, "speculativePages": 0, "purgeablePages": 0,
            "inactivePages": 0, "fileBackedPages": 0,
            "anonymousPages": 0, "throttledPages": 0, "activePages": 0})
        return host["availableBytes"], host

    def tree_footprint(self, _pgid):
        return self.footprint

    def gpu_free(self):
        return self.gpu

    def gpu_free_and_tree_bytes(self, _owner):
        return self.gpu, 1024


@unittest.skipUnless(os.name == "posix", "process-group tests need POSIX")
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        data = {
            "schemaVersion": 1, "backend": "darwin-mlx", "deadlineSeconds": 1,
            "pollMillis": 10, "termGraceMillis": 50,
            "hostFreeReserveBytes": 100, "childFootprintCapBytes": 10**6,
            "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 10**6,
        }
        policy_file = self.root / "policy.json"
        policy_file.write_text(json.dumps(data), encoding="utf-8")
        self.policy = safety.load_policy(policy_file)

    def run_child(self, script, *, probe=None, floor=None, gpu_floor=None, policy=None,
                  event_path=None, on_spawn=None, measured=None):
        return safety.run_guarded(
            [sys.executable, "-c", script], cwd=self.root, env=os.environ.copy(),
            policy=policy or self.policy, stdout_path=self.root / "stdout",
            stderr_path=self.root / "stderr", static_floor_host_bytes=floor,
            static_floor_gpu_bytes=gpu_floor,
            host_estimate_source=None if floor is None else "test-static-estimate",
            gpu_estimate_source=None if gpu_floor is None else "test-static-estimate",
            measured_peak_host_bytes=measured, probe=probe or Probe(),
            event_path=event_path, on_spawn=on_spawn,
        )

    def read_unaccepted(self, error):
        path = safety.write_unaccepted_record(
            self.root / "records" / f"{len(list(self.root.glob('records/*.json')))}.unaccepted.json",
            kind="test-unaccepted-row", coordinate="cell", error=error,
        )
        raw = path.read_bytes()
        self.assertEqual(raw, safety.canonical(json.loads(raw)))
        self.assertEqual((path.parent / f"{path.name}.sha256").read_text(encoding="utf-8"),
                         f"{safety.digest(raw)}  {path.name}\n")
        return json.loads(raw)

    def test_material_row_is_admitted_by_runtime_guards_without_a_peak_proof(self):
        result = self.run_child("open('spawned','w').close()")
        self.assertTrue((self.root / "spawned").exists())
        admission = safety.validate_admission(result.admission, policy_sha256=self.policy.sha256)
        self.assertEqual(admission["mode"], safety.RUNTIME_GUARDED_ADMISSION)
        self.assertEqual(admission["childFootprintCapBytes"], self.policy.child_footprint_cap_bytes)
        self.assertEqual(admission["hostFreeReserveBytes"], self.policy.host_free_reserve_bytes)
        self.assertEqual(admission["policySha256"], self.policy.sha256)
        self.assertIsNone(admission["staticFloorHostBytes"])
        # No static estimate: the row falls back to its cap, recorded with the rule and the
        # measurement the rule compared.
        self.assertEqual(admission["rule"], safety.ESTIMATE_PLUS_RESERVE_RULE)
        self.assertEqual((admission["hostEstimateSource"], admission["hostEstimateBytes"]),
                         (safety.CAP_FALLBACK_ESTIMATE_SOURCE, self.policy.child_footprint_cap_bytes))
        self.assertEqual(admission["hostAvailableBytes"], admission["hostMemoryComponents"]["availableBytes"])
        # The unknown transient peak is a recorded field (null plus reason), not a refusal.
        self.assertIsNone(admission["wholeProcessPeakBoundBytes"])
        self.assertEqual(admission["wholeProcessPeakUnknownReason"], safety.UNKNOWN_PEAK_REASON)
        for mutation in ({"mode": "static-peak-proof"}, {"wholeProcessPeakUnknownReason": ""},
                         {"wholeProcessPeakBoundBytes": 10**6}, {"policySha256": "0" * 64},
                         {"backend": "unknown"}, {"staticFloorHostBytes": 10**6 + 1},
                         {"rule": "cap-plus-reserve"}, {"rule": None}, {"hostEstimateSource": None},
                         {"hostEstimateBytes": None}, {"hostEstimateBytes": 10**6 + 1},
                         {"hostEstimateBytes": 10**6 - 1}, {"hostEstimateSource": "unnamed-static"},
                         {"staticFloorHostBytes": 1024},
                         {"hostAvailableBytes": admission["hostAvailableBytes"] - 1},
                         {"gpuEstimateBytes": 1}):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_admission({**admission, **mutation}, policy_sha256=self.policy.sha256)
        for missing in ("rule", "hostEstimateSource", "hostEstimateBytes", "hostAvailableBytes"):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_admission({key: value for key, value in admission.items() if key != missing},
                                              policy_sha256=self.policy.sha256)
        cuda = safety.runtime_guarded_admission(dataclasses.replace(
            self.policy, backend="linux-cuda", gpu_free_reserve_bytes=1024, child_gpu_cap_bytes=10**6,
            cuda_device_uuid="GPU-12345678-1234-1234-1234-123456789abc"), static_floor_gpu_bytes=1024,
            gpu_estimate_source="test-static-estimate")
        self.assertEqual((cuda["gpuEstimateSource"], cuda["gpuEstimateBytes"]), ("test-static-estimate", 1024))
        # A static admission (no measurement yet) is structurally valid; only a run row must
        # also record available memory covering estimate plus reserve.
        safety.validate_admission(cuda, policy_sha256=self.policy.sha256, admitted=False)
        with self.assertRaisesRegex(safety.SupervisionError, "below reserve plus estimate"):
            safety.validate_admission(cuda, policy_sha256=self.policy.sha256)
        measured = {**cuda, "hostAvailableBytes": 100 + 10**6, "gpuAvailableBytes": 1024 + 1024}
        safety.validate_admission(measured, policy_sha256=self.policy.sha256)
        for mutation in ({"cudaDeviceUuid": None}, {"gpuFreeReserveBytes": None},
                         {"childGpuCapBytes": 0}, {"staticFloorGpuBytes": 10**6 + 1},
                         {"gpuEstimateSource": None}, {"gpuEstimateBytes": 1023},
                         {"gpuAvailableBytes": 1024 + 1023}, {"hostAvailableBytes": 100 + 10**6 - 1}):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_admission({**measured, **mutation}, policy_sha256=self.policy.sha256)

    def test_row_without_runtime_guards_is_refused_before_spawn(self):
        for field, value in (("host_free_reserve_bytes", 0), ("child_footprint_cap_bytes", 0),
                             ("deadline_seconds", 0), ("poll_millis", 0),
                             ("term_grace_millis", 0), ("poll_millis", 1000)):
            with self.subTest(field=field, value=value):
                policy = dataclasses.replace(self.policy, **{field: value})
                with self.assertRaisesRegex(safety.SupervisionError, "unguarded-policy") as caught:
                    self.run_child("open('spawned','w').close()", policy=policy)
                self.assertFalse((self.root / "spawned").exists())
                self.assertIsNone(caught.exception.pid)
                self.assertIsNone(caught.exception.admission)
        cuda = dataclasses.replace(self.policy, backend="linux-cuda",
                                   cuda_device_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                                   gpu_free_reserve_bytes=1024, child_gpu_cap_bytes=10**6)
        for field, value in (("gpu_free_reserve_bytes", None), ("child_gpu_cap_bytes", 0),
                             ("cuda_device_uuid", None)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(safety.SupervisionError, "unguarded-policy"):
                    self.run_child("open('spawned','w').close()",
                                   policy=dataclasses.replace(cuda, **{field: value}))
                self.assertFalse((self.root / "spawned").exists())
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory"):
            safety.runtime_guarded_admission(cuda, static_floor_gpu_bytes=10**6 + 1,
                                             gpu_estimate_source="test-static-estimate")
        # A static estimate without its source (or a source without an estimate) is malformed.
        for floor, source in ((1024, None), (None, "test-static-estimate"), (0, "test-static-estimate")):
            with self.subTest(floor=floor, source=source):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.runtime_guarded_admission(self.policy, static_floor_host_bytes=floor,
                                                     host_estimate_source=source)
        self.assertEqual(safety.runtime_guarded_admission(cuda)["childGpuCapBytes"], 10**6)
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"]), (False, "refused"))

    def test_insufficient_host_ram_is_a_sealed_unaccepted_refusal(self):
        needed = self.policy.host_free_reserve_bytes + self.policy.child_footprint_cap_bytes
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory") as caught:
            self.run_child("open('spawned','w').close()", probe=Probe(free=needed - 1))
        self.assertFalse((self.root / "spawned").exists())
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"], record["pid"]), (False, "refused", None))
        self.assertEqual(record["admission"]["childFootprintCapBytes"], self.policy.child_footprint_cap_bytes)
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory"):
            self.run_child("open('spawned','w').close()",
                           floor=self.policy.child_footprint_cap_bytes + 1)
        self.assertFalse((self.root / "spawned").exists())
        # Without an estimate the row falls back to the cap: covering cap plus reserve (to the next
        # whole page) is admitted.
        result = self.run_child("open('spawned','w').close()", probe=Probe(free=needed + 4095))
        self.assertEqual(result.admission["hostEstimateSource"], safety.CAP_FALLBACK_ESTIMATE_SOURCE)
        self.assertTrue((self.root / "spawned").exists())

    def test_a_row_is_admitted_on_its_estimate_plus_reserve(self):
        # Estimate 8092 + reserve 100: admitted at exactly 8192 available (two pages), refused one
        # byte (here: one page) less, both far below the cap (10**6) plus reserve.
        estimate = 2 * 4096 - self.policy.host_free_reserve_bytes
        needed = self.policy.host_free_reserve_bytes + estimate
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory") as caught:
            self.run_child("open('spawned','w').close()", probe=Probe(free=needed - 1), floor=estimate)
        self.assertFalse((self.root / "spawned").exists())
        self.assertIn(safety.ESTIMATE_PLUS_RESERVE_RULE, caught.exception.detail)
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["outcome"], record["admission"]["hostAvailableBytes"]), ("refused", needed - 4096))
        safety.validate_admission(record["admission"], policy_sha256=self.policy.sha256, admitted=False)
        with self.assertRaisesRegex(safety.SupervisionError, "below reserve plus estimate"):
            safety.validate_admission(record["admission"], policy_sha256=self.policy.sha256)
        result = self.run_child("open('spawned','w').close()", probe=Probe(free=needed), floor=estimate)
        self.assertTrue((self.root / "spawned").exists())
        admission = safety.validate_admission(result.admission, policy_sha256=self.policy.sha256)
        self.assertEqual((admission["staticFloorHostBytes"], admission["hostEstimateSource"],
                          admission["hostEstimateBytes"], admission["hostAvailableBytes"]),
                         (estimate, "test-static-estimate", estimate, needed))
        # A completed identical unit that measured more raises the estimate to its peak.
        (self.root / "spawned").unlink()
        for path in ("stdout", "stderr"):
            (self.root / path).unlink()
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory"):
            self.run_child("open('spawned','w').close()", probe=Probe(free=needed), floor=estimate,
                           measured=estimate + 1)
        self.assertFalse((self.root / "spawned").exists())
        self.assertEqual(safety.admission_estimate(1024, "s", 2048, 10**6, "host"),
                         (safety.MEASURED_PEAK_ESTIMATE_SOURCE, 2048))
        self.assertEqual(safety.admission_estimate(1024, "s", 512, 10**6, "host"), ("s", 1024))
        self.assertEqual(safety.admission_estimate(None, None, 512, 10**6, "host"),
                         (safety.CAP_FALLBACK_ESTIMATE_SOURCE, 10**6))
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory"):
            safety.admission_estimate(1024, "s", 10**6 + 1, 10**6, "host")

    def test_an_estimate_admitted_row_is_still_aborted_at_its_cap(self):
        # Admitted on an 8092-byte estimate (exactly estimate plus reserve available), the child's
        # owned footprint then exceeds the 10**6 cap: the unchanged footprint watchdog aborts and
        # reaps it.
        estimate = 2 * 4096 - self.policy.host_free_reserve_bytes
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint") as caught:
            self.run_child("import time; time.sleep(5)", floor=estimate,
                           probe=Probe(free=2 * 4096, footprint=self.policy.child_footprint_cap_bytes + 1))
        self.assertFalse(safety._group_pids(caught.exception.pid))
        record = self.read_unaccepted(caught.exception)
        self.assertEqual(record["outcome"], "aborted")
        self.assertEqual(record["admission"]["hostEstimateBytes"], estimate)

    def test_insufficient_cuda_free_is_a_sealed_unaccepted_refusal(self):
        cuda = dataclasses.replace(self.policy, backend="linux-cuda",
                                   cuda_device_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                                   gpu_free_reserve_bytes=1024, child_gpu_cap_bytes=10**6)
        needed = cuda.gpu_free_reserve_bytes + cuda.child_gpu_cap_bytes
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory") as caught:
            self.run_child("open('spawned','w').close()", policy=cuda, probe=Probe(gpu_free=needed - 1))
        self.assertFalse((self.root / "spawned").exists())
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"], record["pid"]), (False, "refused", None))
        self.assertEqual(record["admission"]["childGpuCapBytes"], cuda.child_gpu_cap_bytes)
        result = self.run_child("open('spawned','w').close()", policy=cuda, probe=Probe(gpu_free=needed))
        self.assertEqual(result.gpu_free_at_launch, needed)
        self.assertTrue((self.root / "spawned").exists())

    def test_spawn_failure_is_a_sealed_failed_record(self):
        with self.assertRaisesRegex(safety.SupervisionError, "spawn-failure") as caught:
            safety.run_guarded(
                [str(self.root / "missing-entrypoint")], cwd=self.root, env=os.environ.copy(),
                policy=self.policy, stdout_path=self.root / "stdout",
                stderr_path=self.root / "stderr", probe=Probe(),
            )
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"], record["pid"]), (False, "failed", None))
        safety.validate_admission(record["admission"], policy_sha256=self.policy.sha256)
        data = json.loads(self.policy.canonical_bytes)
        data.update({"backend": "windows-cuda",
                     "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
                     "gpuFreeReserveBytes": 100, "childGpuCapBytes": 10**6})
        policy_path = self.root / "windows-policy.json"
        policy_path.write_text(json.dumps(data), encoding="utf-8")
        windows_policy = safety.load_policy(policy_path)
        job = type("FakeJob", (), {"close": lambda self: None})()
        with patch.object(safety.os, "name", "nt"), \
             patch.object(safety.windows, "WindowsJob", return_value=job), \
             patch.object(safety.subprocess, "Popen", side_effect=OSError("not a Win32 application")):
            with self.assertRaisesRegex(safety.SupervisionError, "spawn-failure") as caught:
                safety.run_guarded(
                    ["missing.exe"], cwd=self.root, env=os.environ.copy(), policy=windows_policy,
                    stdout_path=self.root / "win-stdout", stderr_path=self.root / "win-stderr",
                    probe=Probe(),
                )
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"]), (False, "failed"))

    def test_watchdog_abort_and_child_failure_are_unaccepted_records(self):
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint") as caught:
            self.run_child("import time; time.sleep(5)", probe=Probe(footprint=10**7))
        self.assertFalse(safety._group_pids(caught.exception.pid))
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"], record["pid"]),
                         (False, "aborted", caught.exception.pid))
        safety.validate_admission(record["admission"], policy_sha256=self.policy.sha256)
        # The breaching sample is sealed with the abort, so a cap overrun names its size and time.
        breach = record["watchdogChildFootprint"]
        self.assertEqual((breach["footprintBytes"], breach["capBytes"], breach["sampleCount"]),
                         (10**7, self.policy.child_footprint_cap_bytes, 1))
        self.assertEqual(breach["trajectory"], [{"elapsedMillis": breach["elapsedMillis"],
                                                 "footprintBytes": 10**7}])
        self.assertNotIn("processes", breach)  # the fixture probe exposes no per-process split
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaisesRegex(safety.SupervisionError, "child-exit") as caught:
            self.run_child("import sys; sys.exit(7)")
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"]), (False, "failed"))
        self.assertIsInstance(record["pid"], int)

    def test_policy_parsing_and_preflight_refuse_before_spawn(self):
        self.assertEqual(self.policy.sha256, safety.digest(self.policy.canonical_bytes))
        with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
            safety.darwin_available("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2.\nPages speculative: 3.\n")
        self.assertEqual(safety.linux_free_bytes("MemAvailable: 2048 kB\n"), 2048 * 1024)
        with self.assertRaisesRegex(safety.SupervisionError, "preflight-memory"):
            self.run_child("open('spawned','w').close()", probe=Probe(free=10))
        self.assertFalse((self.root / "spawned").exists())

    def test_success_and_nonzero_exit_are_distinct(self):
        result = self.run_child("import time; print('ok'); time.sleep(.05)")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.samples)
        self.assertEqual((self.root / "stdout").read_bytes(), b"ok\n")
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaisesRegex(safety.SupervisionError, "child-exit"):
            self.run_child("import sys; sys.exit(7)")

    def test_production_deadline_still_includes_on_spawn_bootstrap(self):
        now, owned = [0.0], []
        def bootstrap(pid, pgid):
            owned.append(pgid)
            now[0] = 2.0  # The ordinary one-second policy is already expired.
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            safety.run_guarded(
                [sys.executable, "-c", "import threading; threading.Event().wait()"],
                cwd=self.root, env=os.environ.copy(), policy=self.policy, probe=Probe(),
                stdout_path=self.root / "stdout", stderr_path=self.root / "stderr",
                clock=lambda: now[0], on_spawn=bootstrap,
            )
        self.assertEqual(len(owned), 1)
        self.assertFalse(safety._group_pids(owned[0]))

    def test_on_spawn_persists_owned_identity_before_sampling(self):
        status_path = self.root / "spawn-status.json"
        callback_calls = []

        def persist(pid, pgid):
            callback_calls.append((pid, pgid))
            self.assertEqual(pgid, pid)
            self.assertIn(pid, safety._group_pids(pgid))
            temporary = status_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump({"pid": pid, "pgid": pgid}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, status_path)

        case = self

        class PersistedProbe(Probe):
            def tree_footprint(self, pgid):
                self_status = json.loads(status_path.read_text(encoding="utf-8"))
                case.assertEqual(self_status["pgid"], pgid.pgid)
                return super().tree_footprint(pgid)

        result = self.run_child("import time; time.sleep(.15)", probe=PersistedProbe(),
                                on_spawn=persist)
        self.assertEqual(callback_calls, [(result.pid, result.pid)])
        self.assertEqual(json.loads(status_path.read_text(encoding="utf-8")),
                         {"pid": result.pid, "pgid": result.pid})

    def test_on_spawn_exception_reaps_owned_parent_and_descendant(self):
        descendant_path = self.root / "descendant"
        child_source = (
            "import os,subprocess,time; "
            "p=subprocess.Popen(['sleep','5']); "
            "open('descendant','w').write(str(p.pid)); time.sleep(5)"
        )
        ownership = []

        def fail_after_spawn(pid, pgid):
            ownership.append((pid, pgid))
            deadline = time.monotonic() + 1
            while (not descendant_path.exists() or descendant_path.stat().st_size == 0
                   ) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(descendant_path.exists() and descendant_path.stat().st_size > 0)
            raise OSError("status persistence failed")

        with self.assertRaisesRegex(OSError, "status persistence failed"):
            self.run_child(child_source, on_spawn=fail_after_spawn)
        pid, pgid = ownership[0]
        self.assertEqual(pgid, pid)
        self.assertFalse(safety._group_pids(pgid))
        self.assertNotIn(int(descendant_path.read_text(encoding="ascii")), safety._process_table())

    def test_windows_on_spawn_exception_terminates_owned_job(self):
        data = json.loads(self.policy.canonical_bytes)
        data.update({"backend": "windows-cuda",
                     "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
                     "gpuFreeReserveBytes": 100, "childGpuCapBytes": 10**6})
        policy_path = self.root / "windows-policy.json"
        policy_path.write_text(json.dumps(data), encoding="utf-8")
        policy = safety.load_policy(policy_path)

        class FakeJob:
            assigned = False
            terminated = False
            closed = False

            def assign_and_resume(self, _child):
                self.assigned = True

            def terminate_and_reap(self, _child, _grace):
                self.terminated = True

            def close(self):
                self.closed = True

        class FakeChild:
            pid = 43210

            def wait(self, **_kwargs):
                return 0

        class CudaProbe(Probe):
            def gpu_free(self):
                return 10**12

        job = FakeJob()
        observed = []

        def fail_after_owned(pid, pgid):
            observed.append((pid, pgid, job.assigned))
            raise OSError("status persistence failed")

        with patch.object(safety.os, "name", "nt"), \
             patch.object(safety.windows, "WindowsJob", return_value=job), \
             patch.object(safety.subprocess, "Popen", return_value=FakeChild()):
            with self.assertRaisesRegex(OSError, "status persistence failed"):
                self.run_child("unused", policy=policy, probe=CudaProbe(),
                               gpu_floor=1024, on_spawn=fail_after_owned)
        self.assertEqual(observed, [(43210, None, True)])
        self.assertTrue(job.terminated)
        self.assertTrue(job.closed)

    def test_deadline_reaps_owned_process_group(self):
        script = (
            "import os,subprocess,time; "
            "p=subprocess.Popen(['sleep','5']); "
            "open('pid','w').write(str(os.getpid())); "
            "open('descendant','w').write(str(p.pid)); time.sleep(5)"
        )
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            self.run_child(script)
        self.assertEqual(safety._group_pids(int((self.root / "pid").read_text(encoding="ascii"))), set())

    def test_root_exit_does_not_leave_owned_descendant(self):
        script = (
            "import os,subprocess; "
            "p=subprocess.Popen(['sleep','5']); "
            "open('pid','w').write(str(os.getpid())); "
            "open('descendant','w').write(str(p.pid))"
        )
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            self.run_child(script)
        self.assertFalse(safety._group_pids(int((self.root / "pid").read_text(encoding="ascii"))))

    class MembersProbe(Probe):
        """Records every (ppid, pgid) row each owned member showed on any tick."""
        def __init__(self):
            super().__init__()
            self.rows = {}
        def tree_footprint(self, owner):
            table = safety._process_table()
            for pid in owner.members():
                if pid in table:
                    self.rows.setdefault(pid, set()).add(table[pid])
            return super().tree_footprint(owner)

    def test_descendant_in_its_own_group_is_owned_measured_and_reaped(self):
        script = (
            "import os,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import os,time;time.sleep(.1);os.setsid();time.sleep(5)']); "
            "open('escaped-pid','w').write(str(p.pid));time.sleep(5)"
        )
        probe = self.MembersProbe()
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            self.run_child(script, probe=probe)
        escaped = int((self.root / "escaped-pid").read_text(encoding="ascii"))
        self.assertIn(escaped, {group for _parent, group in probe.rows[escaped]})  # measured after leaving
        self.assertNotIn(escaped, safety._process_table())

    def test_row_waits_for_a_reparented_descendant_in_its_own_group(self):
        # The root exits first; its setsid'd descendant is reparented to launchd/init and
        # the row only finishes once that descendant has exited too.
        script = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import os,time;os.setsid();time.sleep(.6)']); "
            "open('escaped-pid','w').write(str(p.pid));time.sleep(.2)"
        )
        probe = self.MembersProbe()
        result = self.run_child(script, probe=probe)
        escaped = int((self.root / "escaped-pid").read_text(encoding="ascii"))
        self.assertEqual(result.returncode, 0)
        self.assertGreaterEqual(result.elapsed_seconds, .5)
        self.assertIn(escaped, probe.rows)
        self.assertNotIn(escaped, safety._process_table())

    @unittest.skipUnless(sys.platform == "darwin", "system_profiler is macOS-only")
    def test_system_profiler_collector_group_is_not_a_refusal(self):
        # SC-20684 W2 run 36808330752: the product's toolchain probe runs system_profiler,
        # whose `-nospawn` collector child starts its own process group.
        script = ("import subprocess; subprocess.run(['/usr/sbin/system_profiler', "
                  "'SPDisplaysDataType'], stdout=subprocess.DEVNULL, check=True)")
        probe = self.MembersProbe()
        result = self.run_child(script, probe=probe, policy=self.patient_policy())
        self.assertEqual(result.returncode, 0)
        own_group = [pid for pid, rows in probe.rows.items() if any(group != result.pid for _p, group in rows)]
        self.assertTrue(own_group, probe.rows)

    def test_system_probe_measures_every_owned_member(self):
        owner = safety.PosixTree(1, set())
        with patch.object(safety.platform, "system", return_value="Darwin"), \
             patch.object(safety.PosixTree, "members", return_value={21, 22, 23}), \
             patch.object(safety, "darwin_phys_footprint", side_effect=lambda pid: pid):
            self.assertEqual(safety.SystemProbe(self.policy).tree_footprint(owner), 66)

    def test_ancestry_survives_reparenting_and_forgets_exited_pids(self):
        known = {10}
        # 11 is a child of 10; 12 a grandchild that moved to its own group.
        self.assertEqual(safety._expand_owned({10: (1, 10), 11: (10, 10), 12: (11, 12), 50: (1, 50)}, known),
                         {10, 11, 12})
        # The root and 11 exit: 12 is reparented to launchd and stays owned; 13 is its child.
        self.assertEqual(safety._expand_owned({12: (1, 12), 13: (12, 12), 50: (1, 50)}, known), {12, 13})
        # Exit race: 13 vanished between listing and read (ESRCH) -- simply absent, not owned.
        self.assertEqual(safety._expand_owned({12: (1, 12), 50: (1, 50)}, known), {12})
        # PID reuse: unrelated processes now hold 10, 11 and 13; none is a descendant.
        self.assertEqual(safety._expand_owned(
            {10: (1, 10), 11: (50, 50), 12: (1, 12), 13: (1, 13), 50: (1, 50)}, known), {12})
        self.assertEqual(known, {12})

    def test_live_reserve_counts_reclaimable_file_cache(self):
        class FileCacheProbe(Probe):
            """Free plus speculative is under the 100-byte reserve on every sample; inactive
            clean file cache keeps the available measure above reserve plus cap until told."""
            def __init__(self, cache_pages):
                super().__init__()
                self.cache_pages = list(cache_pages)
            def host_free(self):
                return 0
            def host_admission(self):
                pages = self.cache_pages.pop(0) if len(self.cache_pages) > 1 else self.cache_pages[0]
                host = safety.darwin_host_memory(4096, {
                    "freePages": 0, "speculativePages": 0, "purgeablePages": 0,
                    "inactivePages": pages, "fileBackedPages": pages, "anonymousPages": 0,
                    "throttledPages": 0, "activePages": 0})
                return host["availableBytes"], host
        result = self.run_child("import time; time.sleep(.1)", probe=FileCacheProbe([1000]))
        self.assertEqual(result.returncode, 0)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaisesRegex(safety.SupervisionError,
                                    r"host-memory: host available 0 bytes \(darwin-vm-stat-available-v3\)") as caught:
            self.run_child("import time; time.sleep(5)", probe=FileCacheProbe([1000, 0]))
        self.assertIsNotNone(caught.exception.pid)
        self.assertEqual(caught.exception.admission["hostMemoryComponents"]["inactivePages"], 1000)
        # The abort record carries the tripping sample itself, next to the admission's.
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["outcome"], record["reason"]), ("aborted", "host-memory"))
        self.assertEqual(record["watchdogHostMemory"]["inactivePages"], 0)
        self.assertEqual(record["watchdogHostMemory"]["metric"], "darwin-vm-stat-available-v3")

    def test_live_reserve_and_footprint_abort(self):
        class DecliningProbe(Probe):
            def host_free(self):
                self.free -= 10**10
                return self.free
        with self.assertRaisesRegex(safety.SupervisionError, "host-memory"):
            self.run_child("import time; time.sleep(5)", probe=DecliningProbe(free=2 * 10**10 + 50))
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint"):
            self.run_child("import time; time.sleep(5)", probe=Probe(footprint=10**7))

    def test_log_cap_aborts_and_reaps(self):
        with self.assertRaisesRegex(safety.SupervisionError, "log-cap"):
            self.run_child("import time; print('x'*20000, flush=True); time.sleep(5)")
        self.assertLessEqual((self.root / "stdout").stat().st_size, self.policy.stdout_cap_bytes)

    def test_probe_failure_aborts_and_reaps(self):
        class FailedProbe(Probe):
            def tree_footprint(self, _pgid):
                raise safety.SupervisionError("probe-failure", "synthetic fault")
        with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
            self.run_child("import time; time.sleep(5)", probe=FailedProbe())

    def patient_policy(self):
        # A deadline no tick-counting test reaches: each is bounded by probe calls.
        return dataclasses.replace(self.policy, deadline_seconds=30)

    def stop_when_present_script(self):
        stop = self.root / "stop"
        return stop, (f"import os, time\nwhile not os.path.exists({str(stop)!r}):\n"
                      "    time.sleep(.01)\n")

    class HostScriptProbe(Probe):
        """host_admission answers from a script: an exception instance raises, else passes."""
        def __init__(self, script, after=None):
            super().__init__()
            self.script = list(script)
            self.after = after
            self.host_reads = 0
        def host_admission(self):
            self.host_reads += 1
            if self.script:
                step = self.script.pop(0)
            else:
                step = self.after() if self.after else None
            if isinstance(step, BaseException):
                raise step
            return super().host_admission()

    @staticmethod
    def vm_stat_fault():
        return safety.SupervisionError("probe-failure", "/usr/bin/vm_stat: exit status 1")

    def test_host_preflight_retries_then_refuses_with_the_error(self):
        probe = self.HostScriptProbe([self.vm_stat_fault()] * 3)
        self.assertEqual(self.run_child("pass", probe=probe).returncode, 0)
        self.assertGreaterEqual(probe.host_reads, 4)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        probe = self.HostScriptProbe([], after=self.vm_stat_fault)
        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_child("open('spawned','w').close()", probe=probe)
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("preflight host probe failed 4 reads; last error /usr/bin/vm_stat: exit status 1",
                      caught.exception.detail)
        self.assertEqual(probe.host_reads, 4)
        self.assertFalse((self.root / "spawned").exists())

    def test_transient_host_failures_never_stop_a_row(self):
        stop, script = self.stop_when_present_script()
        fault = self.vm_stat_fault()
        # Admission, four failed ticks, a good tick, four failed ticks: never five in a row.
        probe = self.HostScriptProbe([None] + [fault] * 16 + [None] + [fault] * 16,
                                     after=lambda: stop.touch())
        result = self.run_child(script, probe=probe, policy=self.patient_policy())
        self.assertEqual(result.returncode, 0)
        self.assertGreater(probe.host_reads, 34)

    def test_sustained_host_failure_stops_with_detail_and_count(self):
        _stop, script = self.stop_when_present_script()
        probe = self.HostScriptProbe([None], after=self.vm_stat_fault)
        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_child(script, probe=probe, policy=self.patient_policy())
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("live host probe failed on 5 consecutive ticks; last error host probe "
                      "failed 4 reads; last error /usr/bin/vm_stat: exit status 1",
                      caught.exception.detail)
        self.assertEqual(probe.host_reads, 1 + 5 * safety.HOST_READS_PER_TICK)

    def test_footprint_read_retries_within_a_tick_and_a_gone_pid_is_none(self):
        def reads(outcomes):
            calls = []
            def read(pid):
                calls.append(pid)
                outcome = outcomes[min(len(calls), len(outcomes)) - 1]
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome, outcome
            return read, calls

        eperm = OSError(1, os.strerror(1))
        read, calls = reads([eperm, eperm, eperm, 7])
        self.assertEqual(safety.darwin_phys_footprint(5, read=read), 7)
        self.assertEqual(len(calls), 4)
        read, calls = reads([eperm])
        with self.assertRaises(safety.SupervisionError) as caught:
            safety.darwin_phys_footprint(5, read=read)
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("failed 4 reads: errno 1: Operation not permitted", caught.exception.detail)
        self.assertEqual(len(calls), 4)
        read, calls = reads([OSError(3, os.strerror(3))])  # ESRCH
        self.assertIsNone(safety.darwin_phys_footprint(5, read=read))
        self.assertEqual(len(calls), 1)

    def test_transient_footprint_failures_never_stop_a_row(self):
        stop, script = self.stop_when_present_script()

        class FlakyProbe(Probe):
            # Four failed ticks, one good read, four more failed ticks: never five in a row.
            ticks = 0
            def tree_footprint(self, pgid):
                self.ticks += 1
                if self.ticks in (5, 10):
                    return 1024
                if self.ticks < 10:
                    raise safety.SupervisionError("probe-failure", "synthetic transient fault")
                stop.touch()
                return super().tree_footprint(pgid)

        probe = FlakyProbe()
        result = self.run_child(script, probe=probe, policy=self.patient_policy())
        self.assertEqual(result.returncode, 0)
        self.assertGreater(probe.ticks, 10)

    def test_gone_owned_tree_is_left_to_exit_handling(self):
        stop, script = self.stop_when_present_script()

        class GoneProbe(Probe):
            ticks = 0
            def tree_footprint(self, _pgid):
                self.ticks += 1
                if self.ticks == 3 * safety.FOOTPRINT_FAILED_TICK_LIMIT:
                    stop.touch()
                return None

        probe = GoneProbe()
        result = self.run_child(script, probe=probe, policy=self.patient_policy())
        self.assertEqual(result.returncode, 0)
        self.assertGreaterEqual(probe.ticks, 3 * safety.FOOTPRINT_FAILED_TICK_LIMIT)

    def test_sustained_footprint_failure_stops_with_errno_and_count(self):
        _stop, script = self.stop_when_present_script()
        reads = []

        def eperm(pid):
            reads.append(pid)
            raise OSError(1, os.strerror(1))

        class FailingProbe(Probe):
            def tree_footprint(self, _pgid):
                return safety.darwin_phys_footprint(4321, read=eperm)

        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_child(script, probe=FailingProbe(), policy=self.patient_policy())
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("failed on 5 consecutive ticks", caught.exception.detail)
        self.assertIn("errno 1: Operation not permitted", caught.exception.detail)
        self.assertEqual(len(reads), 5 * safety.FOOTPRINT_READS_PER_TICK)

    @staticmethod
    def ps_timeout():
        return safety.SupervisionError(
            "probe-failure", "/bin/ps: Command '['/bin/ps', '-axo', 'pid=,ppid=,pgid=,stat=']' "
            "timed out after 2.0 seconds")

    def scripted_table(self, failing):
        """_read_process_table that raises the SC-20684 ps timeout on the read indexes failing()
        selects and otherwise reads the real table."""
        real = safety._read_process_table
        reads = []
        def read():
            reads.append(None)
            if failing(len(reads)):
                raise self.ps_timeout()
            return real()
        return read, reads

    def test_process_table_retries_within_a_tick(self):
        read, reads = self.scripted_table(lambda n: n <= 3)
        with patch.object(safety, "_read_process_table", side_effect=read):
            self.assertIn(os.getpid(), safety._process_table())
        self.assertEqual(len(reads), 4)
        read, reads = self.scripted_table(lambda n: True)
        with patch.object(safety, "_read_process_table", side_effect=read):
            with self.assertRaises(safety.SupervisionError) as caught:
                safety._process_table()
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("process table probe failed 4 reads; last error /bin/ps:", caught.exception.detail)
        self.assertIn("timed out after 2.0 seconds", caught.exception.detail)
        self.assertEqual(len(reads), safety.PROCESS_TABLE_READS_PER_TICK)

    def test_transient_process_tree_failures_never_stop_a_row(self):
        stop, script = self.stop_when_present_script()
        per_tick = safety.PROCESS_TABLE_READS_PER_TICK
        # Four failed ticks, a good tick, four more failed ticks: never five in a row.
        def failing(n):
            if n > 1 + 8 * per_tick:
                stop.touch()
                return False
            return n != 1 + 4 * per_tick
        read, reads = self.scripted_table(failing)
        with patch.object(safety, "_read_process_table", side_effect=read):
            result = self.run_child(script, policy=self.patient_policy())
        self.assertEqual(result.returncode, 0)
        self.assertGreater(len(reads), 1 + 8 * per_tick)

    def test_sustained_process_tree_failure_stops_with_detail_and_count(self):
        _stop, script = self.stop_when_present_script()
        limit = safety.FOOTPRINT_FAILED_TICK_LIMIT * safety.PROCESS_TABLE_READS_PER_TICK
        # Every watchdog read fails; the reaper's later reads see the real table.
        read, reads = self.scripted_table(lambda n: n <= limit)
        with patch.object(safety, "_read_process_table", side_effect=read):
            with self.assertRaises(safety.SupervisionError) as caught:
                self.run_child(script, policy=self.patient_policy())
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("process-tree probe failed on 5 consecutive ticks; last error process table "
                      "probe failed 4 reads; last error /bin/ps:", caught.exception.detail)
        self.assertIn("timed out after 2.0 seconds", caught.exception.detail)
        self.assertGreaterEqual(len(reads), limit)

    def cuda_policy(self):
        return dataclasses.replace(
            self.patient_policy(), backend="linux-cuda",
            cuda_device_uuid="GPU-12345678-1234-1234-1234-123456789abc",
            gpu_free_reserve_bytes=100, child_gpu_cap_bytes=10**6)

    class GpuScriptProbe(Probe):
        """gpu_free_and_tree_bytes fails on the calls failing() selects."""
        def __init__(self, failing, *, preflight=()):
            super().__init__()
            self.failing = failing
            self.preflight = list(preflight)
            self.reads = 0
            self.preflight_reads = 0
        def gpu_free(self):
            self.preflight_reads += 1
            if self.preflight:
                step = self.preflight.pop(0)
                if step is not None:
                    raise step
            return super().gpu_free()
        def gpu_free_and_tree_bytes(self, owner):
            self.reads += 1
            if self.failing(self.reads):
                raise safety.SupervisionError("probe-failure", "nvidia-smi: timed out after 2.0 seconds")
            return super().gpu_free_and_tree_bytes(owner)

    def test_transient_cuda_failures_never_stop_a_row(self):
        stop, script = self.stop_when_present_script()
        per_tick = safety.GPU_READS_PER_TICK
        fault = safety.SupervisionError("probe-failure", "nvidia-smi: timed out after 2.0 seconds")
        def failing(n):
            if n > 1 + 8 * per_tick:
                stop.touch()
                return False
            return n != 1 + 4 * per_tick
        probe = self.GpuScriptProbe(failing, preflight=[fault] * 3)
        result = self.run_child(script, probe=probe, policy=self.cuda_policy())
        self.assertEqual(result.returncode, 0)
        self.assertEqual(probe.preflight_reads, 4)
        self.assertGreater(probe.reads, 1 + 8 * per_tick)

    def test_sustained_cuda_failure_stops_with_detail_and_count(self):
        _stop, script = self.stop_when_present_script()
        probe = self.GpuScriptProbe(lambda n: True)
        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_child(script, probe=probe, policy=self.cuda_policy())
        self.assertEqual(caught.exception.reason, "probe-failure")
        self.assertIn("CUDA probe failed on 5 consecutive ticks; last error CUDA probe failed 4 reads; "
                      "last error nvidia-smi: timed out after 2.0 seconds", caught.exception.detail)
        self.assertEqual(probe.reads, 5 * safety.GPU_READS_PER_TICK)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        fault = safety.SupervisionError("probe-failure", "nvidia-smi: timed out after 2.0 seconds")
        probe = self.GpuScriptProbe(lambda n: False, preflight=[fault] * 4)
        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_child("open('spawned','w').close()", probe=probe, policy=self.cuda_policy())
        self.assertIn("preflight CUDA probe failed 4 reads; last error nvidia-smi", caught.exception.detail)
        self.assertFalse((self.root / "spawned").exists())

    def test_ps_fallback_runs_only_without_libproc_and_with_the_long_timeout(self):
        calls = []
        def output(argv, *, timeout):
            calls.append((argv, timeout))
            return f"{os.getpid()} 1 {os.getpgrp()} S\n77 1 77 Z\n"
        with patch.object(safety, "_load_libproc", return_value=None), \
             patch.object(safety, "_bounded_output", side_effect=output):
            self.assertEqual(safety._process_table(), {os.getpid(): (1, os.getpgrp())})
        self.assertEqual(calls, [(["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat="], 10.0)])
        if sys.platform == "darwin":
            with patch.object(safety, "_bounded_output", side_effect=AssertionError("ps spawned")):
                self.assertIn(os.getpid(), safety._process_table())

    @unittest.skipUnless(sys.platform == "darwin", "libproc is Darwin-only")
    def test_libproc_tree_matches_ps_on_a_live_child_tree(self):
        import subprocess
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys, time\n"
             "kids = [subprocess.Popen(['sleep', '30']) for _ in range(2)]\n"
             "grand = subprocess.Popen([sys.executable, '-c', "
             "'import subprocess, time; subprocess.Popen([\"sleep\", \"30\"]); time.sleep(30)'])\n"
             "zombie = subprocess.Popen(['true']); time.sleep(.5)\n"
             "print('ready', flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, encoding="utf-8", start_new_session=True)
        try:
            self.addCleanup(child.stdout.close)
            self.assertEqual(child.stdout.readline().strip(), "ready")
            group = lambda table: {pid: row for pid, row in table.items() if row[1] == child.pid}
            libproc = group(safety.libproc_process_table(safety._load_libproc()))
            ps = group(safety.ps_process_table())
            self.assertEqual(libproc, ps)
            # Root, two sleeps, the Python grandchild and its sleep; the reaped-later zombie is excluded.
            self.assertEqual(len(libproc), 5)
            self.assertEqual(safety._expand_owned(safety._process_table(), {child.pid}), set(libproc))
        finally:
            os.killpg(child.pid, 9)
            child.wait()

    @unittest.skipUnless(sys.platform == "darwin", "proc_pid_rusage is Darwin-only")
    def test_proc_pid_rusage_reads_a_child_allocation_like_footprint(self):
        import re
        import subprocess
        block = 64 << 20
        child = subprocess.Popen(
            [sys.executable, "-c",
             f"b = bytearray({block})\nfor i in range(0, len(b), 4096): b[i] = 1\n"
             "print('ready', flush=True)\nimport time; time.sleep(30)"],
            stdout=subprocess.PIPE, encoding="utf-8")
        try:
            self.assertEqual(child.stdout.readline(), "ready\n")
            current, peak = safety.proc_pid_rusage_footprint(child.pid)
            tool = subprocess.run(["/usr/bin/footprint", "-p", str(child.pid), "-f", "bytes"],
                                  stdout=subprocess.PIPE, encoding="utf-8", check=True).stdout
            self.assertEqual(safety.darwin_phys_footprint(child.pid), current)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()
        self.assertGreaterEqual(current, block)
        self.assertLess(current, 4 * block)
        self.assertGreaterEqual(peak, current)
        field = lambda name: int(re.search(rf"^\s*{name}:\s*(\d+) B\s*$", tool, re.MULTILINE).group(1))
        self.assertEqual((field("phys_footprint"), field("phys_footprint_peak")), (current, peak))
        with self.assertRaises(ProcessLookupError):
            safety.proc_pid_rusage_footprint(child.pid)
        self.assertIsNone(safety.darwin_phys_footprint(child.pid))

    @unittest.skipUnless(sys.platform == "darwin", "proc_pid_rusage is Darwin-only")
    def test_system_probe_caps_a_real_child_through_proc_pid_rusage(self):
        # A Python interpreter's footprint is well above the 1 MB fixture cap.
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint") as caught:
            self.run_child("import time; time.sleep(5)", probe=safety.SystemProbe(self.policy))
        breach = caught.exception.watchdog_child_footprint
        # The per-process split is the same sample as the breaching total, keyed by owned PID.
        self.assertEqual([item["pid"] for item in breach["processes"]], [caught.exception.pid])
        self.assertEqual(sum(item["footprintBytes"] for item in breach["processes"]),
                         breach["footprintBytes"])
        self.assertGreater(breach["footprintBytes"], breach["capBytes"])

    def test_child_footprint_trajectory_is_evenly_thinned_and_ends_at_the_breach(self):
        trajectory = [(index * 250, 1000 + index) for index in range(1000)]
        breach = safety.child_footprint_breach(1999, 1500, trajectory, {7: 1500, 3: 499})
        points = breach["trajectory"]
        self.assertEqual(len(points), safety.CHILD_FOOTPRINT_TRAJECTORY_POINTS)
        self.assertEqual(points[0], {"elapsedMillis": 0, "footprintBytes": 1000})
        self.assertEqual(points[-1], {"elapsedMillis": 999 * 250, "footprintBytes": 1999})
        self.assertEqual([point["elapsedMillis"] for point in points],
                         sorted({point["elapsedMillis"] for point in points}))
        self.assertEqual((breach["elapsedMillis"], breach["sampleCount"]), (999 * 250, 1000))
        self.assertEqual(breach["processes"], [{"pid": 3, "footprintBytes": 499},
                                               {"pid": 7, "footprintBytes": 1500}])

    def test_footprint_exit_race_requires_terminal_root_and_empty_owned_tree(self):
        class FailedProbe(Probe):
            def __init__(self, after_snapshot=None):
                super().__init__()
                self.after_snapshot = after_snapshot
            def tree_footprint(self, _pgid):
                if self.after_snapshot:
                    self.after_snapshot()
                raise safety.SupervisionError("probe-failure", "process exited during sample")

        class Child:
            pid = 43210
            def __init__(self, second_status):
                self.polls = 0
                self.second_status = second_status
                self.waited = False
            def poll(self):
                # Live on the first poll; every later poll sees the same status.
                self.polls += 1
                return None if self.polls == 1 else self.second_status
            def wait(self, **_kwargs):
                self.waited = True
                return 0

        def invoke(child, second_table, *, after_snapshot=None, event_path=None):
            tables = iter(({child.pid: (1, child.pid)},))
            with patch.object(safety.subprocess, "Popen", return_value=child), \
                 patch.object(safety, "_process_table",
                              side_effect=lambda: next(tables, second_table)), \
                 patch.object(safety, "_stop_tree") as stop:
                try:
                    result = self.run_child("unused", probe=FailedProbe(after_snapshot), event_path=event_path)
                except safety.SupervisionError as error:
                    result = error
                return result, stop.called

        exited = Child(0)
        result, stopped = invoke(exited, {})
        self.assertIsInstance(result, safety.RunResult)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(exited.waited)
        self.assertFalse(stopped)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        live = Child(None)
        result, stopped = invoke(live, {live.pid: (1, live.pid)})
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, "probe-failure")
        self.assertIn("failed on 5 consecutive ticks", result.detail)
        self.assertTrue(stopped)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        nonzero = Child(7)
        result, stopped = invoke(nonzero, {})
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, "child-exit")
        self.assertTrue(stopped)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        capped = Child(0)
        result, stopped = invoke(capped, {}, after_snapshot=lambda: (self.root / "stdout").write_bytes(b"x" * 5000))
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, "log-cap")
        self.assertTrue(stopped)
        self.assertLessEqual((self.root / "stdout").stat().st_size, self.policy.stdout_cap_bytes)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        event = self.root / "events"
        capped_event = Child(0)
        result, stopped = invoke(capped_event, {}, after_snapshot=lambda: event.write_bytes(b"x" * (10**6 + 1)),
                                 event_path=event)
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, "event-cap")
        self.assertTrue(stopped)
        self.assertLessEqual(event.stat().st_size, self.policy.event_cap_bytes)
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()

        descendant = Child(0)
        result, stopped = invoke(descendant, {44444: (descendant.pid, descendant.pid)})
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, "probe-failure")
        self.assertTrue(stopped)

    def test_cuda_probe_parses_selected_uuid_and_owned_pid_only(self):
        gpu = "GPU-12345678-1234-1234-1234-123456789abc"
        policy = safety.SafetyPolicy(
            "linux-cuda", 1, 10, 50, 100, 10**6, 4096, 4096, 4096,
            gpu, 100, 10**6, "fixture", b"{}\n",
        )
        def output(argv, **_kwargs):
            if "--query-gpu=uuid,memory.free" in argv:
                return f"{gpu}, 8192\n"
            if "--query-compute-apps=pid,used_gpu_memory,gpu_uuid" in argv:
                return f"123, 256, {gpu}\n456, 128, {gpu}\n"
            raise AssertionError(argv)
        with patch.object(safety.platform, "system", return_value="Linux"), \
             patch.object(safety, "_bounded_output", side_effect=output):
            probe = safety.SystemProbe(policy)
            self.assertEqual(probe.gpu_free(), 8192 * 1024**2)
            owner = type("Owner", (), {"members": lambda self: {123}})()
            self.assertEqual(probe.gpu_free_and_tree_bytes(owner), (8192 * 1024**2, 256 * 1024**2))

    def test_windows_wddm_unknown_owned_memory_uses_validated_global_upper_bound(self):
        # CI_IMPACT: pure mocked SMI/Job regression for the actual Windows proof
        # parser failure; no native device, model, checkpoint, or process launch.
        data = json.loads(self.policy.canonical_bytes)
        data.update({"backend": "windows-cuda",
                     "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
                     "gpuFreeReserveBytes": 1024, "childGpuCapBytes": 1024})
        path = self.root / "windows-policy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        policy = safety.load_policy(path)
        self.assertEqual(policy.backend, "windows-cuda")
        gpu = data['cudaDeviceUuid']
        queries = []
        def output(argv, **_kwargs):
            queries.append(argv)
            if "--query-gpu=uuid,memory.free" in argv:
                return f"{gpu}, 99000\n"
            if "--query-compute-apps=pid,used_gpu_memory,gpu_uuid" in argv:
                return f"123, N/A, {gpu}\n456, N/A, {gpu}\n"
            return f"{gpu}, WDDM, 100000, 1000, 99000\n"
        with patch.object(safety.platform, "system", return_value="Windows"), \
             patch.object(safety.windows, "trusted_nvidia_smi", return_value="C:\\Windows\\System32\\nvidia-smi.exe"), \
             patch.object(safety, "_bounded_output", side_effect=output):
            probe = safety.SystemProbe(policy)
            owner = type("Owner", (), {"members": lambda self: {123}})()
            free, bound = probe.gpu_free_and_tree_bytes(owner)
            self.assertEqual((free, bound), (99000 * 1024**2, 1001 * 1024**2))
            self.assertEqual(probe.gpu_usage_basis, 'global-used-upper-bound')
            self.assertEqual(sum('--query-gpu=uuid,driver_model.current,memory.total,memory.used,memory.free' in q
                                 for q in queries), 1)
            self.assertTrue(any(f'--id={gpu}' in q for q in queries))

    def test_windows_wddm_unowned_na_does_not_replace_exact_owned_usage(self):
        data = json.loads(self.policy.canonical_bytes)
        gpu = 'GPU-12345678-1234-1234-1234-123456789abc'
        data.update(backend='windows-cuda', cudaDeviceUuid=gpu,
                    gpuFreeReserveBytes=1024, childGpuCapBytes=1024)
        path = self.root / 'windows-unowned-policy.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        def output(argv, **_kwargs):
            if '--query-gpu=uuid,memory.free' in argv:
                return f'{gpu}, 8192\n'
            if '--query-compute-apps=pid,used_gpu_memory,gpu_uuid' in argv:
                return f'456, N/A, {gpu}\n123, 256, {gpu}\n'
            raise AssertionError('global fallback must not run for an unowned N/A row')
        with patch.object(safety.platform, 'system', return_value='Windows'), \
             patch.object(safety.windows, 'trusted_nvidia_smi', return_value='C:\\Windows\\System32\\nvidia-smi.exe'), \
             patch.object(safety, '_bounded_output', side_effect=output):
            probe = safety.SystemProbe(safety.load_policy(path))
            owner = type('Owner', (), {'members': lambda self: {123}})()
            self.assertEqual(probe.gpu_free_and_tree_bytes(owner), (8192 * 1024**2, 256 * 1024**2))
            self.assertEqual(probe.gpu_usage_basis, 'owned-process-attributed')

    def test_windows_wddm_unsafe_process_or_card_rows_refuse_with_bounded_detail(self):
        data = json.loads(self.policy.canonical_bytes)
        gpu = 'GPU-12345678-1234-1234-1234-123456789abc'
        data.update(backend='windows-cuda', cudaDeviceUuid=gpu,
                    gpuFreeReserveBytes=1024, childGpuCapBytes=1024)
        path = self.root / 'windows-invalid-policy.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        cases = (
            (f'123, N/A, broken-uuid\n', '', 'malformed PID/UUID'),
            (f'123, 1, {gpu}\n123, 1, {gpu}\n', '', 'duplicates PID/UUID'),
            (f'123, ???, {gpu}\n', '', 'unknown usage'),
            (f'123, N/A, {gpu}\n', 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, WDDM, 100000, 1000, 99000\n', 'invalid identity/counters'),
            (f'123, N/A, {gpu}\n', f'{gpu}, TCC, 100000, 1000, 99000\n', 'invalid identity/counters'),
            (f'123, N/A, {gpu}\n', f'{gpu}, WDDM, 100000, N/A, 99000\n', 'invalid identity/counters'),
            (f'123, N/A, {gpu}\n', f'{gpu}, WDDM, 100000, 5000, 99000\n', 'inconsistent'),
            (f'123, N/A, {gpu}\n', f'{gpu}, WDDM, 100000, 1000, 99000\n{gpu}, WDDM, 100000, 1000, 99000\n', 'missing or duplicate'),
        )
        with patch.object(safety.platform, 'system', return_value='Windows'), \
             patch.object(safety.windows, 'trusted_nvidia_smi', return_value='C:\\Windows\\System32\\nvidia-smi.exe'):
            for usage, card, reason in cases:
                with self.subTest(reason=reason, usage=usage[:20]):
                    def output(argv, **_kwargs):
                        if '--query-gpu=uuid,memory.free' in argv:
                            return f'{gpu}, 99000\n'
                        if '--query-compute-apps=pid,used_gpu_memory,gpu_uuid' in argv:
                            return usage
                        return card
                    with patch.object(safety, '_bounded_output', side_effect=output):
                        probe = safety.SystemProbe(safety.load_policy(path))
                        owner = type('Owner', (), {'members': lambda self: {123}})()
                        with self.assertRaisesRegex(safety.SupervisionError, reason) as caught:
                            probe.gpu_free_and_tree_bytes(owner)
                        self.assertLess(len(str(caught.exception)), 300)


class OperatorStopTests(unittest.TestCase):
    def test_absent_stop_file_is_a_no_op_and_present_one_records_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stops = safety.operator_stop_files(None, root)
            self.assertEqual(stops, (root / "STOP",))
            self.assertEqual(safety.operator_stop_files(root / "halt", root), (root / "STOP", root / "halt"))
            safety.check_operator_stop(stops, root / "logs", kind="k", before="row-a", index=0, total=2)
            self.assertFalse((root / "logs").exists())
            (root / "STOP").write_bytes(b"")
            records = []
            for _ in range(2):
                with self.assertRaises(safety.OperatorStop) as caught:
                    safety.check_operator_stop(stops, root / "logs", kind="k", before="row-b", index=1, total=2)
                records.append(caught.exception.record)
            self.assertEqual([path.name for path in records],
                             ["operator-stop.attempt-0.json", "operator-stop.attempt-1.json"])
            raw = records[0].read_bytes()
            record = json.loads(raw)
            self.assertEqual((record["status"], record["beforeRow"], record["beforeRowSlug"], record["rowsAccepted"]),
                             ("stopped-by-operator", 1, "row-b", 1))
            self.assertEqual(records[0].with_name(f"{records[0].name}.sha256").read_text(encoding="utf-8"),
                             f"{safety.digest(raw)}  {records[0].name}\n")
            self.assertNotIn(safety.OPERATOR_STOP_EXIT_CODE, (0, 1, 2))
            custom = safety.operator_stop_files(root / "halt", root)
            self.assertTrue(safety.is_operator_stop_entry(root, custom, "STOP"))
            self.assertTrue(safety.is_operator_stop_entry(root, custom, "halt"))
            self.assertFalse(safety.is_operator_stop_entry(root, custom, "stray"))

    def test_both_stop_paths_stop_and_stat_errors_are_not_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "elsewhere").mkdir()
            stops = safety.operator_stop_files(root / "elsewhere" / "halt", root)
            logs = root / "logs"
            safety.check_operator_stop(stops, logs, kind="k", before="row-a", index=0, total=2)
            (root / "STOP").write_bytes(b"")  # the default path still stops a --stop-file parent
            with self.assertRaises(safety.OperatorStop) as caught:
                safety.check_operator_stop(stops, logs, kind="k", before="row-a", index=0, total=2)
            self.assertEqual(json.loads(caught.exception.record.read_bytes())["stopFiles"], [str(root / "STOP")])
            (root / "STOP").unlink()
            (root / "elsewhere" / "halt").write_bytes(b"")  # and so does the custom path alone
            with self.assertRaises(safety.OperatorStop) as caught:
                safety.check_operator_stop(stops, logs, kind="k", before="row-a", index=0, total=2)
            self.assertEqual(json.loads(caught.exception.record.read_bytes())["stopFiles"],
                             [str(root / "elsewhere" / "halt")])
            blocked = (root / "elsewhere" / "halt" / "STOP",)  # ENOTDIR, not "absent"
            with self.assertRaises(OSError):
                safety.check_operator_stop(blocked, logs, kind="k", before="row-a", index=0, total=2)


class WindowsJobCleanupTests(unittest.TestCase):
    """Injected WinAPI lifecycle: no Windows runtime, child process, model or GPU required."""

    def exercise(self, mode="success", implementation=None):
        from scripts import media_campaign_windows as windows
        events = []
        state = {"total": 4, "active": 2, "live": {22, 33}, "terminated": False}
        class Kernel:
            def OpenProcess(self, access, inherit, pid):
                events.append(("open", pid, access, inherit))
                return 0 if mode.startswith("open-failure") and pid == 22 else pid + 1000
            def IsProcessInJob(self, process, job, owned):
                events.append(("owned", process, job))
                owned._obj.value = mode != "foreign"
                return mode != "membership-failure"
            def TerminateJobObject(self, job, code):
                events.append(("terminate", job, code))
                if not state["terminated"]:
                    state["terminated"] = True
                    state.update(active=0, live=set())
                    if mode in ("late-listed", "late-vanished"):
                        state["total"] += 1
                        if mode == "late-listed":
                            state.update(active=1, live={44})
                    if mode == "remaining-member":
                        state.update(active=1, live={22})
                if mode == "success":
                    # Job accounting/list is EMPTY before either retained handle signals.
                    events.append(("empty-before-signals", members()))
                return mode != "terminate-failure"
            def WaitForSingleObject(self, process, milliseconds):
                events.append(("wait", process, milliseconds))
                if mode == "wait-timeout":
                    return 258
                if mode == "wait-failure":
                    return 0xffffffff
                if process == 1044:
                    state.update(active=0, live=set())
                return 0
            def CloseHandle(self, process):
                events.append(("close", process))
                if mode == "birth-after-handle-release" and process == 1022:
                    # Exact late window: the child's retained handle has been released,
                    # then it creates a grandchild. Teardown must NOT silently succeed.
                    state.update(total=5, active=1, live={44})
                return not (mode == "close-failure" and process == 1022)
        class Child:
            def wait(self, timeout):
                events.append(("root-wait", timeout))
                if mode == "root-timeout":
                    raise windows.subprocess.TimeoutExpired("fixture", timeout)
                return 0  # Root already exited; descendants still need their own signals.
        job = windows.WindowsJob.__new__(windows.WindowsJob)
        job.kernel = Kernel()
        job.handle = 900
        snapshots = 0
        def members():
            nonlocal snapshots
            snapshots += 1
            events.append(("members", snapshots))
            if snapshots == 1:
                return {22, 33}
            if mode == "open-failure-live":
                return {22}
            if mode == "open-failure-exited":
                return set()
            return set(state["live"])
        job.members = members
        accounting_calls = 0
        def accounting():
            nonlocal accounting_calls
            accounting_calls += 1
            if (mode == "initial-accounting-failure" and accounting_calls == 1) or (mode == "final-accounting-failure" and accounting_calls > 1):
                raise windows.WindowsJobError("fixture accounting failure")
            result = {key: state[key] for key in ("total", "active")}
            events.append(("accounting", result.copy()))
            return result
        job._accounting = accounting
        ticks = iter([0.0, 0.03125, 0.0625, 0.09375, 0.1015625, 0.109375])
        invoke = implementation or windows.WindowsJob.terminate_and_reap
        errors = {
            "foreign": windows.WindowsJobError,
            "membership-failure": windows.WindowsJobError,
            "open-failure-live": windows.WindowsJobError,
            "open-failure-exited": windows.WindowsJobError,
            "terminate-failure": windows.WindowsJobError,
            "wait-timeout": windows.WindowsJobError,
            "wait-failure": windows.WindowsJobError,
            "remaining-member": windows.WindowsJobError,
            "late-vanished": windows.WindowsJobError,
            "birth-after-handle-release": windows.WindowsJobError,
            "root-timeout": windows.WindowsJobError,
            "initial-accounting-failure": windows.WindowsJobError,
            "final-accounting-failure": windows.WindowsJobError,
            "close-failure": windows.WindowsJobError,
        }
        with patch.object(windows.time, "monotonic", side_effect=lambda: next(ticks, 0.125)), \
             patch.object(windows, "_failed", side_effect=lambda operation: windows.WindowsJobError(operation)):
            if mode in errors:
                with self.assertRaises(errors[mode]) as caught:
                    invoke(job, Child(), 0.125)
                if mode in ("late-vanished", "birth-after-handle-release"):
                    message = str(caught.exception)
                    self.assertIn("accounted=2 expected=3", message)
                    self.assertIn("assigned_total=5 initial_total=4", message)
                    self.assertIn("job_live=1" if mode == "birth-after-handle-release" else "job_live=0", message)
                    self.assertIn("check=after-handle-release" if mode == "birth-after-handle-release" else "check=after-retained-waits", message)
            else:
                invoke(job, Child(), 0.125)
        opened = [event[1] + 1000 for event in events if event[0] == "open" and not (mode.startswith("open-failure") and event[1] == 22)]
        self.assertEqual([event[1] for event in events if event[0] == "close"], opened)
        for event in events:
            if event[0] == "open":
                self.assertEqual(event[2:], (0x00100000 | 0x1000, False))
        if mode in ("foreign", "membership-failure"):
            self.assertFalse(any(event[0] == "wait" for event in events))
        if mode in ("open-failure-live", "open-failure-exited"):
            self.assertEqual([event[1] for event in events if event[0] == "wait"], [1033])
        if mode in ("wait-timeout", "wait-failure", "root-timeout", "late-vanished", "initial-accounting-failure", "final-accounting-failure"):
            self.assertEqual([event[1] for event in events if event[0] == "wait"], [1022, 1033])
        if mode == "success":
            self.assertIn(("empty-before-signals", set()), events)
            self.assertEqual(events[0][0], "accounting", "the assignment fence must precede the first PID snapshot")
            waits = [event for event in events if event[0] == "wait"]
            self.assertEqual([event[1] for event in waits], [1022, 1033])
            self.assertEqual([event[2] for event in waits], [63, 32])
            self.assertLess(next(i for i, event in enumerate(events) if event[0] == "owned"), next(i for i, event in enumerate(events) if event[0] == "terminate"))
            self.assertLess(next(i for i, event in enumerate(events) if event[0] == "root-wait"), next(i for i, event in enumerate(events) if event[0] == "wait"))
        if mode == "late-listed":
            self.assertEqual([event[1] for event in events if event[0] == "wait"], [1022, 1033, 1044])
            self.assertEqual(len([event for event in events if event[0] == "terminate"]), 2)
        return events

    def test_empty_job_list_and_exited_root_do_not_replace_descendant_exit_waits(self):
        self.exercise()

    def test_wait_membership_and_acquisition_failures_are_closed_and_handles_are_released(self):
        for mode in ("foreign", "membership-failure", "open-failure-live", "open-failure-exited", "terminate-failure", "wait-timeout", "wait-failure", "remaining-member", "root-timeout", "initial-accounting-failure", "final-accounting-failure", "close-failure"):
            with self.subTest(mode=mode):
                self.exercise(mode)

    def test_late_births_are_waited_or_reported_unproven_even_after_handle_release(self):
        for mode in ("late-listed", "late-vanished", "birth-after-handle-release"):
            with self.subTest(mode=mode):
                self.exercise(mode)

    def test_accounting_reads_job_assignment_and_live_counts_fail_closed(self):
        from scripts import media_campaign_windows as windows
        job = windows.WindowsJob.__new__(windows.WindowsJob)
        class Kernel:
            def QueryInformationJobObject(self, handle, info, output, size, returned):
                self.asserted = (handle, info)
                output._obj.TotalProcesses = 7
                output._obj.ActiveProcesses = 3
                return True
        job.kernel = Kernel()
        job.handle = 900
        self.assertEqual(job._accounting(), {"total": 7, "active": 3})
        self.assertEqual(job.kernel.asserted, (900, 1))
        with patch.object(job.kernel, "QueryInformationJobObject", return_value=False), \
             patch.object(windows, "_failed", return_value=windows.WindowsJobError("accounting failed")):
            with self.assertRaisesRegex(windows.WindowsJobError, "accounting failed"):
                job._accounting()

    def test_cleanup_regressions_are_discriminated_by_injected_lifecycle(self):
        import inspect
        import textwrap
        from scripts import media_campaign_windows as windows
        source = textwrap.dedent(inspect.getsource(windows.WindowsJob.terminate_and_reap))
        mutants = (
            ("status = self.kernel.WaitForSingleObject(process, remaining_ms)", "status = 0", "success"),
            ("if not owned.value:", "if False:", "foreign"),
            ("if not self.kernel.CloseHandle(process):", "if False:", "success"),
            ("math.ceil(max(0.0, until - time.monotonic()) * 1000)", "math.ceil(grace * 1000)", "success"),
            ('fence("after-handle-release")', "pass", "birth-after-handle-release"),
            ("expected != len(handles)", "False", "late-vanished"),
            ("child.wait(timeout=max(0.0, until - time.monotonic()))", "pass", "root-timeout"),
        )
        for original, replacement, mode in mutants:
            with self.subTest(mutation=original):
                self.assertIn(original, source)
                namespace = dict(vars(windows))
                exec(compile(source.replace(original, replacement), "<injected cleanup mutant>", "exec"), namespace)
                with self.assertRaises(AssertionError):
                    self.exercise(mode, namespace["terminate_and_reap"])


    def test_negative_repeat_control_propagates_assertions_without_subtest_suppression(self):
        import inspect
        # Run an actual unittest outcome context. Merely calling a fresh TestCase's subTest
        # directly would not reproduce unittest's suppression of an assertion during .run().
        class Control(unittest.TestCase):
            _repeat_deadline_reaping = WindowsSupervisorTests._repeat_deadline_reaping
            calls = 0
            def _deadline_reaps_root_and_grandchild_after_root_exit(self):
                self.calls += 1
                raise AssertionError("fixture descendant is still alive")
            def runTest(self):
                with tempfile.TemporaryDirectory() as directory:
                    self.root = Path(directory)
                    original = self.root
                    with self.assertRaisesRegex(AssertionError, "fixture descendant is still alive"):
                        self._repeat_deadline_reaping(subtests=False)
                    self.assertEqual(self.calls, 1)
                    self.assertEqual(self.root, original)
        result = unittest.TestResult()
        Control().run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful(), (result.failures, result.errors))
        def suppressing_mutant(case, *, subtests):
            return WindowsSupervisorTests._repeat_deadline_reaping(case, subtests=True)
        mutant_result = unittest.TestResult()
        with patch.object(Control, "_repeat_deadline_reaping", suppressing_mutant):
            Control().run(mutant_result)
        self.assertFalse(mutant_result.wasSuccessful(), "subTest assertion suppression mutant escaped")
        source = inspect.getsource(WindowsSupervisorTests.test_repeated_reaping_case_rejects_skip_descendant_reap_mutant)
        self.assertIn("self._repeat_deadline_reaping(subtests=False)", source)


@unittest.skipUnless(os.name == "nt", "real Job Object smoke requires Windows")
class WindowsSupervisorTests(unittest.TestCase):
    """Real owned child/grandchild, real Windows memory APIs; no GPU/model use."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = {
            "schemaVersion": 1, "backend": "windows-cuda", "deadlineSeconds": 3,
            "pollMillis": 20, "termGraceMillis": 2000,
            "hostFreeReserveBytes": 1024, "childFootprintCapBytes": 512 * 1024**2,
            "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 10**6,
            "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "gpuFreeReserveBytes": 1024, "childGpuCapBytes": 1024**2,
        }
        self.sampled = []
        self.membership_samples = []
        self.supervision_start = None

    def test_native_host_probe_and_trusted_cuda_probe_path(self):
        from scripts import media_campaign_windows as windows
        self.assertGreater(windows.host_free_bytes(), 0)
        self.assertTrue(Path(windows.trusted_nvidia_smi()).is_absolute())

    def _run(self, script, *, cap=None, on_spawn=None, clock=time.monotonic):
        policy_data = dict(self.data)
        if cap is not None:
            policy_data["childFootprintCapBytes"] = cap
        path = self.root / "policy.json"
        path.write_text(json.dumps(policy_data), encoding="utf-8")
        policy = safety.load_policy(path)
        parent = self
        def observed_real_clock():
            now = time.monotonic()  # Observe, never pause/advance/substitute the real deadline clock.
            if parent.supervision_start is None:
                parent.supervision_start = now
            return now
        class Probe:
            def host_free(self):
                return 10**12
            def host_admission(self):
                return self.host_free(), None
            def gpu_free(self):
                return 10**12
            def tree_footprint(self, owner):
                amount = owner.footprint_bytes()
                parent.sampled.append(amount)
                members = owner.members()
                parent.membership_samples.append((observed_real_clock(), members))
                return amount
            def gpu_free_and_tree_bytes(self, owner):
                # GPU collection is separately fixture-tested; these children never touch CUDA.
                return 10**12, 1024
        return safety.run_guarded(
            [sys.executable, "-c", script], cwd=self.root, env=os.environ.copy(),
            policy=policy, stdout_path=self.root / "stdout", stderr_path=self.root / "stderr",
            probe=Probe(),
            on_spawn=on_spawn, clock=clock,
        )

    def test_on_spawn_reports_job_owned_root_without_posix_group(self):
        calls = []
        result = self._run("import time; time.sleep(.1)",
                           on_spawn=lambda pid, pgid: calls.append((pid, pgid)))
        self.assertEqual(calls, [(result.pid, None)])

    def _has_exited(self, pid):
        import ctypes
        from scripts import media_campaign_windows as windows
        kernel, _ = windows._apis()
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            self.assertEqual(ctypes.get_last_error(), 87)  # PID no longer exists.
            return True
        try:
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.WaitForSingleObject.restype = ctypes.c_uint32
            return kernel.WaitForSingleObject(handle, 0) == 0
        finally:
            kernel.CloseHandle(handle)

    def _assert_exited(self, pid):
        self.assertTrue(self._has_exited(pid), f"owned process {pid} is still alive")

    def test_missing_readiness_reaps_owned_root(self):
        clock, owned = FixtureDeadlineClock(), []
        def never_ready(pid, _pgid):
            owned.append(pid)
            clock.admit(lambda: False, timeout=.05)
        with self.assertRaisesRegex(AssertionError, "fixture startup readiness deadline"):
            self._run("import threading; threading.Event().wait()", on_spawn=never_ready, clock=clock)
        self.assertEqual(len(owned), 1)
        self._assert_exited(owned[0])

    def test_deadline_reaps_root_and_grandchild_after_root_exit(self):
        self._deadline_reaps_root_and_grandchild_after_root_exit()

    def test_repeated_deadline_reaps_root_and_grandchild_after_root_exit(self):
        self._repeat_deadline_reaping(subtests=True)

    def _repeat_deadline_reaping(self, *, subtests):
        from contextlib import nullcontext
        original_root = self.root
        try:
            # Twenty independent post-readiness 3s deadlines exercise root-exit ordering;
            # bootstrap has its own bound, never a Python-startup speed assertion.
            for attempt in range(20):
                context = self.subTest(attempt=attempt) if subtests else nullcontext()
                with context, tempfile.TemporaryDirectory(dir=original_root) as directory:
                    self.root = Path(directory)
                    self._deadline_reaps_root_and_grandchild_after_root_exit()
        finally:
            self.root = original_root

    def _deadline_reaps_root_and_grandchild_after_root_exit(self):
        # Atomic PID publication avoids observing a newly-created but unflushed file.
        grandchild = (
            "import os,pathlib,threading; b=bytearray(4*1024*1024); "
            "pathlib.Path('grandchild-pid.tmp').write_text(str(os.getpid()),encoding='ascii'); "
            "os.replace('grandchild-pid.tmp','grandchild-pid'); threading.Event().wait()"
        )
        script = (
            "import os,pathlib,subprocess,sys; "
            "pathlib.Path('root-pid.tmp').write_text(str(os.getpid()),encoding='ascii'); "
            "os.replace('root-pid.tmp','root-pid'); "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}])"
        )
        clock = FixtureDeadlineClock()
        ready_pids = None
        # Membership observation is per attempt: each repeated run gets its own real-clock start.
        self.membership_samples = []
        self.supervision_start = None
        def fixture_ready(pid, _pgid):
            def ready():
                nonlocal ready_pids
                root_pid = published_fixture_pid(self.root / "root-pid")
                grandchild_pid = published_fixture_pid(self.root / "grandchild-pid")
                if root_pid is None or grandchild_pid is None:
                    return False
                self.assertEqual(root_pid, pid)
                self.assertGreater(grandchild_pid, 0)
                self.assertNotEqual(grandchild_pid, root_pid)
                if not self._has_exited(root_pid):
                    return False
                ready_pids = (root_pid, grandchild_pid)
                return True
            clock.admit(ready)  # Independent 30s bootstrap bound, not an extended teardown limit.
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            self._run(script, on_spawn=fixture_ready, clock=clock)
        self.assertIsNotNone(clock.ready_at)
        self.assertTrue(any(amount > 0 for amount in self.sampled))
        self.assertIsNotNone(ready_pids)
        root_pid, grandchild_pid = ready_pids
        # The epic's ownership proof: the supervisor must have observed the live grandchild as
        # owned while the root was already gone, and before the deadline fired.
        self.assertIsNotNone(self.supervision_start)
        self.assertTrue(
            any(grandchild_pid in members and root_pid not in members
                and at < self.supervision_start + self.data["deadlineSeconds"]
                for at, members in self.membership_samples),
            "deadline test never observed an owned live grandchild after root exit and before the deadline",
        )
        for pid in ready_pids:
            self._assert_exited(pid)

    def test_repeated_reaping_case_rejects_skip_descendant_reap_mutant(self):
        from scripts import media_campaign_windows as windows
        original_factory = windows.WindowsJob
        original_assert = self._assert_exited
        jobs = []
        def factory():
            job = original_factory()
            kernel = job.kernel
            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = windows.wintypes.HANDLE
            kernel.DuplicateHandle.argtypes = [windows.wintypes.HANDLE, windows.wintypes.HANDLE,
                windows.wintypes.HANDLE, windows.ctypes.POINTER(windows.wintypes.HANDLE),
                windows.wintypes.DWORD, windows.wintypes.BOOL, windows.wintypes.DWORD]
            kernel.DuplicateHandle.restype = windows.wintypes.BOOL
            duplicate = windows.wintypes.HANDLE()
            current = kernel.GetCurrentProcess()
            try:
                self.assertTrue(kernel.DuplicateHandle(current, job.handle, current,
                    windows.ctypes.byref(duplicate), 0, False, 2))  # DUPLICATE_SAME_ACCESS
            except BaseException:
                job.close()
                raise
            # A TEST-owned Job reference keeps ordinary production close fully intact, while
            # preventing last-handle kill from hiding the omitted explicit descendant reap.
            drain_job = original_factory.__new__(original_factory)
            drain_job.kernel, drain_job.psapi, drain_job.handle = kernel, job.psapi, duplicate.value
            entry = {"job": drain_job, "root_job": job, "child": None, "drained": False}
            jobs.append(entry)
            assign = job.assign_and_resume
            def assign_child(child):
                entry["child"] = child
                assign(child)
            job.assign_and_resume = assign_child
            job.terminate_and_reap = lambda child, grace: child.wait(timeout=grace)
            return job
        def drain():
            for entry in jobs:
                if not entry["drained"]:
                    try:
                        if entry["child"] is not None:
                            entry["job"].terminate_and_reap(entry["child"], self.data["termGraceMillis"] / 1000)
                    finally:
                        entry["job"].close()
                        entry["root_job"].close()
                        entry["drained"] = True
        def assert_then_drain(pid):
            try:
                original_assert(pid)  # Unchanged instantaneous Wait(handle, 0)==0.
            except AssertionError:
                drain()  # Reap BEFORE the repeated case's TemporaryDirectory is removed.
                raise
        try:
            with patch.object(windows, "WindowsJob", side_effect=factory), \
                 patch.object(self, "_assert_exited", side_effect=assert_then_drain):
                with self.assertRaises(AssertionError):
                    self._repeat_deadline_reaping(subtests=False)
        finally:
            drain()
        self.assertTrue(jobs)
        self.assertTrue(all(entry["drained"] for entry in jobs))

    def test_late_grandchild_after_snapshot_handle_release_is_reaped_or_fails_closed(self):
        import queue
        import threading
        from scripts import media_campaign_windows as windows
        kernel, _ = windows._apis()
        job = windows.WindowsJob()
        child = None
        grandchild_handle = None
        reader = None
        grandchild = "import time; time.sleep(10)"
        script = (
            "import subprocess,sys,time; sys.stdin.buffer.read(1); "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
            "print(p.pid,flush=True); time.sleep(10)"
        )
        try:
            child = safety.subprocess.Popen(
                [sys.executable, "-c", script], stdin=safety.subprocess.PIPE,
                stdout=safety.subprocess.PIPE, stderr=safety.subprocess.DEVNULL,
                creationflags=0x00000004,  # CREATE_SUSPENDED: Job admission precedes execution.
            )
            job.assign_and_resume(child)
            original_members = job.members
            first_snapshot = True
            released = False
            def members():
                nonlocal first_snapshot, released, grandchild_handle, reader
                snapshot = original_members()
                if first_snapshot:
                    first_snapshot = False
                    self.assertEqual(snapshot, {child.pid})
                    # Open and RELEASE the child's snapshot handle before permitting its birth.
                    # The production cleanup receives the now-stale snapshot, opening exactly
                    # the capture->kill window; lifetime accounting must detect this new PID.
                    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, child.pid)
                    self.assertTrue(handle)
                    owned = windows.wintypes.BOOL()
                    self.assertTrue(kernel.IsProcessInJob(handle, job.handle, windows.ctypes.byref(owned)))
                    self.assertTrue(owned.value)
                    self.assertTrue(kernel.CloseHandle(handle))
                    released = True
                    acknowledgements = queue.Queue()
                    reader = threading.Thread(target=lambda: acknowledgements.put(child.stdout.readline()), daemon=True)
                    reader.start()
                    child.stdin.write(b"x")
                    child.stdin.flush()
                    pid = int(acknowledgements.get(timeout=3).strip())
                    grandchild_handle = kernel.OpenProcess(0x00100000, False, pid)
                    self.assertTrue(grandchild_handle)
                return snapshot
            job.members = members
            try:
                job.terminate_and_reap(child, self.data["termGraceMillis"] / 1000)
            except windows.WindowsJobError as error:
                self.assertIn("owned Job teardown unproven", str(error))
                self.assertIn("accounted=", str(error))
                self.assertIn("job_live=", str(error))
                self.assertIn("check=", str(error))
            else:
                self.assertEqual(kernel.WaitForSingleObject(grandchild_handle, 0), 0)
            self.assertTrue(released)
        finally:
            job.close()  # Kill-on-close still includes EVERY assignment, including late births.
            if child is not None:
                child.wait(timeout=3)
                if child.stdin is not None:
                    child.stdin.close()
                if reader is not None:
                    reader.join(timeout=3)
                    self.assertFalse(reader.is_alive())
                if child.stdout is not None:
                    child.stdout.close()
            if grandchild_handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(grandchild_handle, 3000), 0)
                finally:
                    kernel.CloseHandle(grandchild_handle)

    def test_child_cap_aborts_and_reaps(self):
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint"):
            self._run("import os,time; open('root-pid','w').write(str(os.getpid())); b=bytearray(128*1024*1024); time.sleep(10)",
                      cap=64 * 1024**2)
        self.assertTrue(any(amount > 64 * 1024**2 for amount in self.sampled))
        self._assert_exited(int((self.root / "root-pid").read_text(encoding="ascii")))


if __name__ == "__main__":
    unittest.main()
