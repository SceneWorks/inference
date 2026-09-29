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


class Probe:
    def __init__(self, *, free=10**12, footprint=1024, gpu_free=10**12):
        self.free = free
        self.footprint = footprint
        self.gpu = gpu_free

    def host_free(self):
        return self.free

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
                  event_path=None, on_spawn=None):
        return safety.run_guarded(
            [sys.executable, "-c", script], cwd=self.root, env=os.environ.copy(),
            policy=policy or self.policy, stdout_path=self.root / "stdout",
            stderr_path=self.root / "stderr", static_floor_host_bytes=floor,
            static_floor_gpu_bytes=gpu_floor, probe=probe or Probe(),
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
        # The unknown transient peak is a recorded field (null plus reason), not a refusal.
        self.assertIsNone(admission["wholeProcessPeakBoundBytes"])
        self.assertEqual(admission["wholeProcessPeakUnknownReason"], safety.UNKNOWN_PEAK_REASON)
        for mutation in ({"mode": "static-peak-proof"}, {"wholeProcessPeakUnknownReason": ""},
                         {"wholeProcessPeakBoundBytes": 10**6}, {"policySha256": "0" * 64},
                         {"backend": "unknown"}, {"staticFloorHostBytes": 10**6 + 1}):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_admission({**admission, **mutation}, policy_sha256=self.policy.sha256)
        cuda = safety.runtime_guarded_admission(dataclasses.replace(
            self.policy, backend="linux-cuda", gpu_free_reserve_bytes=1024, child_gpu_cap_bytes=10**6,
            cuda_device_uuid="GPU-12345678-1234-1234-1234-123456789abc"), static_floor_gpu_bytes=1024)
        safety.validate_admission(cuda, policy_sha256=self.policy.sha256)
        for mutation in ({"cudaDeviceUuid": None}, {"gpuFreeReserveBytes": None},
                         {"childGpuCapBytes": 0}, {"staticFloorGpuBytes": 10**6 + 1}):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_admission({**cuda, **mutation}, policy_sha256=self.policy.sha256)

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
            safety.runtime_guarded_admission(cuda, static_floor_gpu_bytes=10**6 + 1)
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
        # Free RAM exactly covering cap plus reserve, and a floor under the cap, are admitted.
        result = self.run_child("open('spawned','w').close()", probe=Probe(free=needed), floor=1024)
        self.assertEqual(result.admission["staticFloorHostBytes"], 1024)
        self.assertTrue((self.root / "spawned").exists())

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
        (self.root / "stdout").unlink()
        (self.root / "stderr").unlink()
        with self.assertRaisesRegex(safety.SupervisionError, "child-exit") as caught:
            self.run_child("import sys; sys.exit(7)")
        record = self.read_unaccepted(caught.exception)
        self.assertEqual((record["accepted"], record["outcome"]), (False, "failed"))
        self.assertIsInstance(record["pid"], int)

    def test_policy_parsing_and_preflight_refuse_before_spawn(self):
        self.assertEqual(self.policy.sha256, safety.digest(self.policy.canonical_bytes))
        self.assertEqual(safety.darwin_free_bytes("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2.\nPages speculative: 3.\n"), 5 * 16384)
        self.assertEqual(safety.darwin_footprint_bytes("    phys_footprint: 32768 B\n"), 32768)
        self.assertEqual(safety.linux_free_bytes("MemAvailable: 2048 kB\n"), 2048 * 1024)
        with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
            safety.darwin_footprint_bytes("phys_footprint: unavailable")
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
                case.assertEqual(self_status["pgid"], pgid)
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

    def test_observed_child_that_escapes_process_group_is_reaped(self):
        script = (
            "import os,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import os,time;time.sleep(.1);os.setsid();time.sleep(5)']); "
            "open('escaped-pid','w').write(str(p.pid));time.sleep(5)"
        )
        with self.assertRaisesRegex(safety.SupervisionError, "process-escape"):
            self.run_child(script)
        self.assertNotIn(int((self.root / "escaped-pid").read_text(encoding="ascii")), safety._process_table())

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
                self.statuses = iter((None, second_status))
                self.waited = False
            def poll(self):
                return next(self.statuses)
            def wait(self, **_kwargs):
                self.waited = True
                return 0

        def invoke(child, second_table, *, after_snapshot=None, event_path=None):
            tables = ({child.pid: (1, child.pid)}, second_table)
            with patch.object(safety.subprocess, "Popen", return_value=child), \
                 patch.object(safety, "_process_table", side_effect=tables), \
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
             patch.object(safety, "_bounded_output", side_effect=output), \
             patch.object(safety, "_group_pids", return_value={123}):
            probe = safety.SystemProbe(policy)
            self.assertEqual(probe.gpu_free(), 8192 * 1024**2)
            self.assertEqual(probe.gpu_free_and_tree_bytes(999), (8192 * 1024**2, 256 * 1024**2))

    def test_windows_policy_and_unknown_gpu_memory_refuse(self):
        data = json.loads(self.policy.canonical_bytes)
        data.update({"backend": "windows-cuda",
                     "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
                     "gpuFreeReserveBytes": 1024, "childGpuCapBytes": 1024})
        path = self.root / "windows-policy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        policy = safety.load_policy(path)
        self.assertEqual(policy.backend, "windows-cuda")
        def output(argv, **_kwargs):
            if "--query-gpu=uuid,memory.free" in argv:
                return "GPU-12345678-1234-1234-1234-123456789abc, 8192\n"
            return "123, N/A, GPU-12345678-1234-1234-1234-123456789abc\n"
        with patch.object(safety.platform, "system", return_value="Windows"), \
             patch.object(safety.windows, "trusted_nvidia_smi", return_value="C:\\Windows\\System32\\nvidia-smi.exe"), \
             patch.object(safety, "_bounded_output", side_effect=output):
            probe = safety.SystemProbe(policy)
            owner = type("Owner", (), {"members": lambda self: {123}})()
            with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
                probe.gpu_free_and_tree_bytes(owner)


@unittest.skipUnless(os.name == "nt", "real Job Object smoke requires Windows")
class WindowsSupervisorTests(unittest.TestCase):
    """Real owned child/grandchild, real Windows memory APIs; no GPU/model use."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = {
            "schemaVersion": 1, "backend": "windows-cuda", "deadlineSeconds": 3,
            "pollMillis": 20, "termGraceMillis": 300,
            "hostFreeReserveBytes": 1024, "childFootprintCapBytes": 512 * 1024**2,
            "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 10**6,
            "cudaDeviceUuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "gpuFreeReserveBytes": 1024, "childGpuCapBytes": 1024**2,
        }
        self.sampled = []

    def test_native_host_probe_and_trusted_cuda_probe_path(self):
        from scripts import media_campaign_windows as windows
        self.assertGreater(windows.host_free_bytes(), 0)
        self.assertTrue(Path(windows.trusted_nvidia_smi()).is_absolute())

    def _run(self, script, *, cap=None, on_spawn=None):
        policy_data = dict(self.data)
        if cap is not None:
            policy_data["childFootprintCapBytes"] = cap
        path = self.root / "policy.json"
        path.write_text(json.dumps(policy_data), encoding="utf-8")
        policy = safety.load_policy(path)
        parent = self
        class Probe:
            def host_free(self):
                return 10**12
            def gpu_free(self):
                return 10**12
            def tree_footprint(self, owner):
                amount = owner.footprint_bytes()
                parent.sampled.append(amount)
                return amount
            def gpu_free_and_tree_bytes(self, owner):
                # GPU collection is separately fixture-tested; these children never touch CUDA.
                return 10**12, 1024
        return safety.run_guarded(
            [sys.executable, "-c", script], cwd=self.root, env=os.environ.copy(),
            policy=policy, stdout_path=self.root / "stdout", stderr_path=self.root / "stderr",
            probe=Probe(),
            on_spawn=on_spawn,
        )

    def test_on_spawn_reports_job_owned_root_without_posix_group(self):
        calls = []
        result = self._run("import time; time.sleep(.1)",
                           on_spawn=lambda pid, pgid: calls.append((pid, pgid)))
        self.assertEqual(calls, [(result.pid, None)])

    def _assert_exited(self, pid):
        import ctypes
        from scripts import media_campaign_windows as windows
        kernel, _ = windows._apis()
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            self.assertEqual(ctypes.get_last_error(), 87)  # PID no longer exists.
            return
        try:
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.WaitForSingleObject.restype = ctypes.c_uint32
            self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0)
        finally:
            kernel.CloseHandle(handle)

    def test_deadline_reaps_root_and_grandchild_after_root_exit(self):
        grandchild = "import os,time; b=bytearray(4*1024*1024); open('grandchild-pid','w').write(str(os.getpid())); time.sleep(10)"
        script = (
            "import os,subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
            "open('root-pid','w').write(str(os.getpid())); time.sleep(.5)"
        )
        with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
            self._run(script)
        self.assertTrue(any(amount > 0 for amount in self.sampled))
        self._assert_exited(int((self.root / "root-pid").read_text(encoding="ascii")))
        self._assert_exited(int((self.root / "grandchild-pid").read_text(encoding="ascii")))

    def test_child_cap_aborts_and_reaps(self):
        with self.assertRaisesRegex(safety.SupervisionError, "child-footprint"):
            self._run("import os,time; open('root-pid','w').write(str(os.getpid())); b=bytearray(128*1024*1024); time.sleep(10)",
                      cap=64 * 1024**2)
        self.assertTrue(any(amount > 64 * 1024**2 for amount in self.sampled))
        self._assert_exited(int((self.root / "root-pid").read_text(encoding="ascii")))


if __name__ == "__main__":
    unittest.main()
