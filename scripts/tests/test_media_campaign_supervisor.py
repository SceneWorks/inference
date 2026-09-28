"""CPU-only fail-closed process supervision tests; no model or GPU is started."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import media_campaign_supervisor as safety


class Probe:
    def __init__(self, *, free=10**12, footprint=1024):
        self.free = free
        self.footprint = footprint

    def host_free(self):
        return self.free

    def tree_footprint(self, _pgid):
        return self.footprint


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

    def run_child(self, script, *, probe=None, bound=1024, policy=None, event_path=None):
        return safety.run_guarded(
            [sys.executable, "-c", script], cwd=self.root, env=os.environ.copy(),
            policy=policy or self.policy, stdout_path=self.root / "stdout",
            stderr_path=self.root / "stderr", source_peak_host_bytes=bound,
            probe=probe or Probe(), event_path=event_path,
        )

    def test_policy_parsing_and_preflight_refuse_before_spawn(self):
        self.assertEqual(self.policy.sha256, safety.digest(self.policy.canonical_bytes))
        self.assertEqual(safety.darwin_free_bytes("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2.\nPages speculative: 3.\n"), 5 * 16384)
        self.assertEqual(safety.darwin_footprint_bytes("    phys_footprint: 32768 B\n"), 32768)
        self.assertEqual(safety.linux_free_bytes("MemAvailable: 2048 kB\n"), 2048 * 1024)
        with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
            safety.darwin_footprint_bytes("phys_footprint: unavailable")
        with self.assertRaisesRegex(safety.SupervisionError, "unbounded-source"):
            self.run_child("open('spawned','w').close()", bound=None)
        self.assertFalse((self.root / "spawned").exists())
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

    def _run(self, script, *, cap=None):
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
            source_peak_host_bytes=1024, source_peak_gpu_bytes=1024, probe=Probe(),
        )

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
