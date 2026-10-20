"""CPU-only tests for the MOSS CUDA unit supervisor; no Cargo, GPU or model is started."""

import json
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock
from pathlib import Path

from scripts.ci import moss_cuda_unit as unit

OK_OUTPUT = (
    "running 1 test\n"
    f"test {unit.TEST_NAME} ... ok\n\n"
    "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 41 filtered out; finished in 1.20s\n"
)


def artifact(exe, *, package="path+file:///r/crates/candle-gen-moss-image#candle-gen-moss-image@0.1.0",
             name="candle_gen_moss_image", kind=("lib",), test=True):
    return json.dumps({
        "reason": "compiler-artifact", "package_id": package,
        "target": {"name": name, "kind": list(kind)}, "profile": {"test": test}, "executable": str(exe),
    })


class SelectArtifactTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.target = self.dir / "target"
        (self.target / "debug").mkdir(parents=True)
        self.exe = self.target / "debug" / "candle_gen_moss_image-abc"
        self.exe.write_text("x", encoding="utf-8")

    def select(self, *lines):
        return unit.select_artifact("\n".join(lines), self.target)

    def test_single_matching_lib_test_executable_is_selected(self):
        noise = json.dumps({"reason": "build-finished", "success": True})
        self.assertEqual(self.select(noise, artifact(self.exe), noise), self.exe)

    def test_legacy_package_id_format_matches(self):
        legacy = "candle-gen-moss-image 0.1.0 (path+file:///r/crates/candle-gen-moss-image)"
        self.assertEqual(self.select(artifact(self.exe, package=legacy)), self.exe)

    def test_every_non_matching_or_ambiguous_artifact_fails_closed(self):
        other = self.target / "debug" / "other"
        other.write_text("x", encoding="utf-8")
        outside = self.dir / "elsewhere"
        outside.write_text("x", encoding="utf-8")
        cases = {
            "no artifact": [],
            "sibling package": [artifact(self.exe, package="path+file:///r/x#candle-gen-moss-image-extra@0.1.0")],
            "non-test profile": [artifact(self.exe, test=False)],
            "integration target kind": [artifact(self.exe, kind=("test",))],
            "other target name": [artifact(self.exe, name="integration")],
            "two executables": [artifact(self.exe), artifact(other)],
            "outside target dir": [artifact(outside)],
            "missing file": [artifact(self.target / "debug" / "gone")],
            "non-JSON line": ["not json", artifact(self.exe)],
        }
        for label, lines in cases.items():
            with self.subTest(label), self.assertRaises(unit.UnitFailure):
                self.select(*lines)


class VerifyOutputTests(unittest.TestCase):
    def test_exact_positive_result_passes_whatever_the_filtered_count(self):
        unit.verify_output(OK_OUTPUT)
        unit.verify_output(OK_OUTPUT.replace("41 filtered", "0 filtered"))

    def test_zero_skipped_mismatched_or_failed_results_are_rejected(self):
        cases = {
            "zero tests": OK_OUTPUT.replace("running 1 test", "running 0 tests").replace("1 passed", "0 passed"),
            "ignored": OK_OUTPUT.replace("0 ignored", "1 ignored"),
            "failed": OK_OUTPUT.replace("0 failed", "1 failed").replace("ok. 1", "FAILED. 1"),
            "different name": OK_OUTPUT.replace("cuda_production", "cpu_production"),
            "not ok line": OK_OUTPUT.replace(" ... ok", " ... ignored"),
            "two tests": OK_OUTPUT.replace("running 1 test", "running 2 tests").replace("1 passed", "2 passed"),
            "running count alone differs": OK_OUTPUT.replace("running 1 test", "running 2 tests"),
            "empty": "",
        }
        for label, output in cases.items():
            with self.subTest(label), self.assertRaises(unit.UnitFailure):
                unit.verify_output(output)


def child_script(body):
    return [sys.executable, "-c", textwrap.dedent(body)]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds
        time.sleep(0.002)


@unittest.skipUnless(os.name == "posix", "owned process-group tests need POSIX")
class SuperviseTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = self.dir / "out.log"

    def run_child(self, body, *, clock=None, **kwargs):
        clock = clock or FakeClock()
        owner = unit.PosixOwner()
        self.owner = owner
        return unit.supervise(
            child_script(body), owner, self.log, clock=clock, sleep=clock.sleep,
            poll=1.0, grace=5.0, env=kwargs.pop("env", dict(os.environ)), **kwargs,
        )

    def test_success_returns_the_child_output(self):
        output = self.run_child("print('hello')")
        self.assertEqual(output.strip(), "hello")

    def test_nonzero_exit_fails(self):
        with self.assertRaisesRegex(unit.UnitFailure, "exited with 3"):
            self.run_child("import sys; sys.exit(3)")

    def test_deadline_kills_the_owned_tree_including_a_grandchild(self):
        pid_file = self.dir / "grandchild.pid"
        body = f"""
            import subprocess, sys, time
            g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
            open({str(pid_file)!r}, 'w').write(str(g.pid))
            time.sleep(120)
        """
        with self.assertRaisesRegex(unit.UnitFailure, "deadline"):
            self.run_child(body, deadline=3.0)
        self.assertTrue(pid_file.exists(), "grandchild never started; the test proved nothing")
        self.assert_dead(int(pid_file.read_text(encoding="utf-8")))
        self.assertFalse(self.owner.alive())

    def test_rss_over_the_cap_kills_the_tree(self):
        pid_file = self.dir / "child.pid"
        body = f"""
            import os, time
            open({str(pid_file)!r}, 'w').write(str(os.getpid()))
            time.sleep(120)
        """
        owner = unit.PosixOwner()
        owner.footprint = lambda: 10**9  # injected sample: above the 1-byte cap below
        clock = FakeClock()
        with self.assertRaisesRegex(unit.UnitFailure, "above the 1 byte cap"):
            unit.supervise(child_script(body), owner, self.log, rss_cap=1, clock=clock,
                           sleep=clock.sleep, poll=1.0, grace=5.0)
        self.assertFalse(owner.alive())

    def test_rss_at_or_under_the_cap_does_not_trip(self):
        owner = unit.PosixOwner()
        owner.footprint = lambda: 5
        clock = FakeClock()
        output = unit.supervise(child_script("import time; time.sleep(0.05); print('ok')"), owner,
                                self.log, rss_cap=5, clock=clock, sleep=clock.sleep, poll=0.01)
        self.assertEqual(output.strip(), "ok")

    def test_real_footprint_probe_counts_the_owned_group(self):
        owner = unit.PosixOwner()
        child = owner.spawn(child_script("import time; time.sleep(30)"), dict(os.environ), None, None)
        try:
            time.sleep(0.3)
            self.assertGreater(owner.footprint(), 0)
        finally:
            owner.terminate(5.0)
            owner.close()
        self.assertIsNotNone(child.returncode)

    def test_probe_error_still_cleans_up_the_tree(self):
        owner = unit.PosixOwner()

        def broken():
            raise RuntimeError("probe broke")

        owner.footprint = broken
        clock = FakeClock()
        with self.assertRaisesRegex(RuntimeError, "probe broke"):
            unit.supervise(child_script("import time; time.sleep(120)"), owner, self.log,
                           clock=clock, sleep=clock.sleep, poll=1.0, grace=5.0)
        self.assertFalse(owner.alive())

    def test_scrubbed_and_forced_environment(self):
        env = dict(os.environ, MOSS_PORTABLE_EXCHANGE_OUTPUT="x", MOSS_PORTABLE_EXCHANGE_INPUT="y",
                   RUST_TEST_THREADS="9")
        output = self.run_child(
            "import os; print(os.environ.get('MOSS_PORTABLE_EXCHANGE_OUTPUT'), "
            "os.environ.get('MOSS_PORTABLE_EXCHANGE_INPUT'), os.environ['RUST_TEST_THREADS'])",
            env=env,
        )
        self.assertEqual(output.strip(), "None None 1")

    def assert_dead(self, pid):
        for _ in range(200):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.01)
        self.fail(f"owned grandchild {pid} survived supervision")


@unittest.skipUnless(os.name == "nt", "real Job Object ownership needs Windows")
class WindowsOwnerTests(unittest.TestCase):
    """The production WindowsOwner with fake Python children: no CUDA, Cargo or weights."""

    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def supervise(self, script, **kwargs):
        return unit.supervise([sys.executable, "-c", script], unit.WindowsOwner(), self.dir / "out.log",
                              env=dict(os.environ), poll=0.05, grace=5.0, **kwargs)

    def assert_exited(self, name):
        from scripts.tests.test_media_campaign_supervisor import WindowsSupervisorTests

        WindowsSupervisorTests._assert_exited(self, int((self.dir / name).read_text(encoding="ascii")))

    def test_success_and_nonzero_exit(self):
        self.assertEqual(self.supervise("print('hello')").strip(), "hello")
        with self.assertRaisesRegex(unit.UnitFailure, "exited with 3"):
            self.supervise("import sys; sys.exit(3)")

    def test_deadline_reaps_the_root_and_a_grandchild(self):
        grandchild = f"import os,time; open({str(self.dir / 'g')!r},'w').write(str(os.getpid())); time.sleep(60)"
        script = (
            "import os,subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
            f"open({str(self.dir / 'r')!r},'w').write(str(os.getpid())); time.sleep(60)"
        )
        with self.assertRaisesRegex(unit.UnitFailure, "deadline"):
            self.supervise(script, deadline=2.0)
        self.assert_exited("r")
        self.assert_exited("g")

    def test_rss_cap_is_sampled_from_the_job_and_reaps(self):
        script = (
            f"import os,time; open({str(self.dir / 'r')!r},'w').write(str(os.getpid())); "
            "b=bytearray(128*1024*1024); time.sleep(60)"
        )
        with self.assertRaisesRegex(unit.UnitFailure, "byte cap"):
            self.supervise(script, deadline=30.0, rss_cap=64 * 1024**2)
        self.assert_exited("r")


class Gpu0OccupancyTests(unittest.TestCase):
    GPUS = "0, GPU-aaa\n1, GPU-bbb\n"

    def check(self, gpus=GPUS, apps="", fail=None):
        def query(*args):
            if fail:
                raise fail
            return gpus if "index,uuid" in args[0] else apps
        unit.require_gpu0_idle(query)

    def test_idle_or_other_gpu_only_is_admitted(self):
        self.check()
        self.check(apps="GPU-bbb, 4242\n")

    def test_gpu0_process_is_refused_without_signalling_it(self):
        with self.assertRaisesRegex(unit.UnitFailure, "compute process 7"):
            self.check(apps="GPU-bbb, 4242\nGPU-aaa, 7\n")

    def test_unreadable_malformed_or_unknown_occupancy_is_refused(self):
        cases = {
            "tool error": dict(fail=unit.UnitFailure("nvidia-smi exited with 9")),
            "os error": dict(fail=FileNotFoundError("nvidia-smi")),
            "timeout": dict(fail=unit.subprocess.TimeoutExpired("nvidia-smi", 15)),
            "no gpus": dict(gpus=""),
            "gpu0 absent": dict(gpus="1, GPU-bbb\n"),
            "duplicate index": dict(gpus="0, GPU-aaa\n0, GPU-bbb\n"),
            "bad uuid": dict(gpus="0, nope\n"),
            "malformed gpu row": dict(gpus="0\n"),
            "unknown uuid": dict(apps="GPU-zzz, 5\n"),
            "N/A pid": dict(apps="GPU-bbb, [N/A]\n"),
            "short app row": dict(apps="GPU-bbb\n"),
        }
        for label, kwargs in cases.items():
            with self.subTest(label), self.assertRaises(unit.UnitFailure):
                self.check(**kwargs)

    def test_nonzero_exit_is_refused_even_with_a_valid_idle_inventory_on_stdout(self):
        from unittest import mock

        def completed(returncode):
            def run(argv, **_):
                out = "0, GPU-aaa\n" if "index,uuid" in argv[1] else ""
                return unit.subprocess.CompletedProcess(argv, returncode, stdout=out, stderr="boom")
            return run

        with mock.patch.object(unit, "smi_path", return_value="nvidia-smi"):
            with mock.patch.object(unit.subprocess, "run", completed(0)):
                unit.require_gpu0_idle()  # the same stdout with exit 0 is admitted
            with mock.patch.object(unit.subprocess, "run", completed(9)):
                with self.assertRaisesRegex(unit.UnitFailure, "exited with 9"):
                    unit.require_gpu0_idle()

    def test_run_checks_occupancy_before_spawning(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as scratch, \
             mock.patch.object(unit, "os", types.SimpleNamespace(name="nt")), \
             mock.patch.object(unit, "provenance", return_value="x"), \
             mock.patch.object(unit, "require_gpu0_idle", side_effect=unit.UnitFailure("busy")), \
             mock.patch.object(unit, "WindowsOwner") as owner, \
             mock.patch.object(unit, "supervise") as supervise:
            record = Path(scratch) / "exe"
            exe = Path(scratch) / "t"
            exe.write_text("x", encoding="utf-8")
            record.write_text(str(exe), encoding="utf-8")
            with self.assertRaisesRegex(unit.UnitFailure, "busy"):
                unit.run_command(record)
            supervise.assert_not_called()
            owner.assert_not_called()


class SmokeGateTests(unittest.TestCase):
    def result(self, run, skipped=(), failed=False):
        outcome = unittest.TestResult()
        outcome.testsRun = run
        outcome.skipped = [(None, "s")] * len(skipped)
        if failed:
            outcome.failures.append((None, "f"))
        return outcome

    def test_only_a_clean_run_with_executed_tests_is_accepted(self):
        self.assertIsNone(unit.smoke_failure(self.result(5)))
        self.assertIn("executed 0", unit.smoke_failure(self.result(0)))
        self.assertIn("2 skipped", unit.smoke_failure(self.result(4, skipped=(1, 2))))
        self.assertIn("2 skipped", unit.smoke_failure(self.result(2, skipped=(1, 2))))
        self.assertIn("failed", unit.smoke_failure(self.result(5, failed=True)))

    @unittest.skipIf(os.name == "nt", "the fail-closed path is the non-Windows one")
    def test_without_windows_ownership_smoke_and_run_fail_closed_not_unguarded(self):
        with self.assertRaisesRegex(unit.UnitFailure, "unavailable"):
            unit.smoke_command()
        self.assertEqual(unit.main(["smoke"]), 1)
        with tempfile.TemporaryDirectory() as scratch:
            record = Path(scratch) / "exe"
            record.write_text(sys.executable, encoding="utf-8")
            self.assertEqual(unit.main(["run", "--exe-file", str(record)]), 1)

    def test_smoke_names_the_existing_windows_job_tests_and_the_owner_tests(self):
        self.assertEqual(
            unit.SMOKE_TESTS,
            ("scripts.tests.test_media_campaign_supervisor.WindowsSupervisorTests",
             "scripts.tests.test_moss_cuda_unit.WindowsOwnerTests"),
        )


class InventoryTests(unittest.TestCase):
    GPU = '0, GPU-a, "RTX, test", 0, 24000\n1, GPU-b, other, N/A, 24000\n'
    APPS = 'GPU-a, 7256, C:\\secret-dir\\foreign.exe, N/A\nGPU-b, 7256, N/A, 128\n'

    def capture(self, gpu=None, apps=None, cim=None, machine="MICHAEL-TRX50"):
        query = mock.Mock(side_effect=[self.GPU if gpu is None else gpu, self.APPS if apps is None else apps])
        process = mock.Mock(return_value={} if cim is None else cim)
        with mock.patch.object(unit.platform, "node", return_value=machine), mock.patch.object(unit.platform, "system", return_value="Windows"):
            return unit.inventory_document(query, process), query, process

    def test_current_metadata_preserves_identity_unknown_memory_and_pid_scope(self):
        with mock.patch.dict(os.environ, {"RUNNER_NAME": "cuda-windows-2", "GITHUB_SHA": "a" * 40, "SECRET_CREDENTIAL": "MUST_NOT_PUBLISH"}):
            doc, query, process = self.capture(cim={7256: {"pid": 7256, "name": "foreign.exe"}})
        self.assertEqual(doc["runner"], "cuda-windows-2")
        self.assertEqual(doc["source_sha"], "a" * 40)
        self.assertNotIn("MUST_NOT_PUBLISH", json.dumps(doc))
        self.assertEqual(doc["status"], "CAPTURED")
        self.assertEqual(doc["machine"], "MICHAEL-TRX50")
        self.assertEqual(doc["gpus"]["rows"][0]["free_memory_mib"], 0)
        self.assertEqual(doc["gpus"]["rows"][1]["free_memory_mib"], "UNKNOWN")
        first, second = doc["compute_processes"]["rows"]
        self.assertEqual((first["pid"], first["gpu_index"], first["process_basename"], first["used_memory_mib"]), (7256, 0, "foreign.exe", "UNKNOWN"))
        self.assertEqual((second["gpu_index"], second["process_basename"], second["used_memory_mib"]), (1, "UNKNOWN", 128))
        self.assertNotIn("secret-dir", json.dumps(doc))
        process.assert_called_once_with([7256])
        self.assertEqual(query.call_args_list, [mock.call("--query-gpu=index,uuid,name,memory.free,memory.total"), mock.call("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")])

    def test_wrong_machine_never_queries_and_unknown_is_not_idle(self):
        doc, query, process = self.capture(machine="SOME-OTHER-HOST")
        self.assertEqual(doc["status"], "WRONG_MACHINE")
        self.assertEqual(doc["compute_processes"]["status"], "UNKNOWN")
        query.assert_not_called(); process.assert_not_called()
        doc, _, _ = self.capture(gpu="", apps="bad inventory")
        self.assertEqual(doc["status"], "PARTIAL")
        self.assertEqual(doc["gpus"]["status"], "UNKNOWN")
        self.assertEqual(doc["compute_processes"]["status"], "UNKNOWN")

    def test_denied_nvml_and_cim_are_recorded_not_fabricated_idle(self):
        with mock.patch.object(unit.platform, "node", return_value="MICHAEL-TRX50"), mock.patch.object(unit.platform, "system", return_value="Windows"):
            doc = unit.inventory_document(mock.Mock(side_effect=OSError("denied")), mock.Mock())
            self.assertEqual(doc["status"], "PARTIAL")
            self.assertEqual(doc["compute_processes"]["status"], "UNKNOWN")
            doc = unit.inventory_document(mock.Mock(side_effect=[self.GPU, self.APPS]), mock.Mock(side_effect=PermissionError("denied")))
            self.assertEqual(doc["compute_processes"]["status"], "KNOWN")
            self.assertEqual(doc["compute_processes"]["rows"][0]["windows_process"], "UNKNOWN")
            self.assertEqual(doc["windows_metadata"], {"status": "UNKNOWN", "reason": "PermissionError"})

    def test_invalid_nvml_pid_never_reaches_cim(self):
        for pid in ("0", "1 OR 1=1", "4294967296", "N/A"):
            doc, _, cim = self.capture(apps=f"GPU-a, {pid}, test, 0\n")
            self.assertEqual(doc["compute_processes"]["status"], "UNKNOWN", pid)
            cim.assert_not_called()

    def test_cim_reads_only_nvml_pids_and_returns_only_allowed_metadata(self):
        raw = [{"ProcessId": 7256, "ParentProcessId": 100, "Name": r"C:\secret-dir\foreign.exe", "CreationDate": "2026-10-20T22:00:00Z", "SessionId": 0, "CommandLine": "SECRET", "Environment": "SECRET"}]
        with mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout=json.dumps(raw))) as run:
            rows = unit.inventory_processes([7256])
        command = run.call_args.args[0][-1]
        self.assertIn("-Filter 'ProcessId = 7256'", command)
        self.assertIn("-Property ProcessId,ParentProcessId,Name,CreationDate,SessionId", command)
        self.assertNotIn("CommandLine", command)
        self.assertNotIn("Environment", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertIn("-NoProfile", run.call_args.args[0])
        self.assertIn("-NonInteractive", run.call_args.args[0])
        self.assertEqual(rows, {7256: {"pid": 7256, "parent_pid": 100, "name": "foreign.exe", "creation_time": "2026-10-20T22:00:00Z", "session_id": 0}})
        self.assertNotIn("SECRET", json.dumps(rows))
        with mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=9, stdout=json.dumps(raw))):
            with self.assertRaisesRegex(unit.UnitFailure, "restricted CIM query exited with 9"):
                unit.inventory_processes([7256])
        with mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="[]")) as run:
            self.assertEqual(unit.inventory_processes([]), {})
            for ids in (["7256"], [0], [4294967296]):
                with self.assertRaises(unit.UnitFailure): unit.inventory_processes(ids)
            run.assert_not_called()

    def test_unknown_numbers_missing_cim_and_non_windows_are_not_invented(self):
        for value in ("N/A", "[N/A]", "nan", "inf", "-1"):
            self.assertEqual(unit.inventory_memory(value), "UNKNOWN")
        with mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout='[{"ProcessId":7256}]')):
            self.assertEqual(unit.inventory_processes([7256]), {7256: {"pid": 7256, "parent_pid": "UNKNOWN", "name": "UNKNOWN", "creation_time": "UNKNOWN", "session_id": "UNKNOWN"}})
        with mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout='[{"ProcessId":123}]')):
            with self.assertRaises(unit.UnitFailure): unit.inventory_processes([7256])
        query = mock.Mock(side_effect=[self.GPU, self.APPS])
        with mock.patch.object(unit.platform, "node", return_value="MICHAEL-TRX50"), mock.patch.object(unit.platform, "system", return_value="Darwin"):
            self.assertEqual(unit.inventory_document(query, mock.Mock(return_value={}))["status"], "WRONG_MACHINE")
            query.assert_not_called()
        doc, _, cim = self.capture(apps="")
        self.assertEqual(doc["compute_processes"], {"status": "KNOWN", "rows": []})
        self.assertEqual(doc["status"], "CAPTURED")
        cim.assert_called_once_with([])

    def test_inventory_nvml_query_has_utf8_timeout_and_nonzero_refusal(self):
        with mock.patch.object(unit, "smi_path", return_value=r"C:\trusted\nvidia-smi.exe"), mock.patch.object(unit.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="valid")) as run:
            self.assertEqual(unit.inventory_query("--query-gpu=index,uuid"), "valid")
            self.assertEqual(run.call_args.kwargs["timeout"], 10)
            self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
            self.assertIn("--format=csv,noheader,nounits", run.call_args.args[0])
            self.assertEqual(run.call_args.args[0][0], r"C:\trusted\nvidia-smi.exe")
            run.return_value = types.SimpleNamespace(returncode=9, stdout="valid idle output")
            with self.assertRaisesRegex(unit.UnitFailure, "exited with 9"):
                unit.inventory_query("--query-gpu=index,uuid")

    def test_fixed_inventory_artifact_and_cli_reject_arbitrary_commands(self):
        with tempfile.TemporaryDirectory() as scratch, mock.patch.dict(os.environ, {"RUNNER_TEMP": scratch}), mock.patch.object(unit, "inventory_document", return_value={"status": "CAPTURED", "marker": "actual"}):
            self.assertEqual(unit.main(["inventory"]), 0)
            path = Path(scratch) / "moss-cuda-inventory.json"
            self.assertTrue(path.is_file(), "the diagnostic artifact has a fixed basename")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["marker"], "actual")
        with tempfile.TemporaryDirectory() as scratch, mock.patch.dict(os.environ, {"RUNNER_TEMP": scratch}), mock.patch.object(unit, "inventory_document", return_value={"status": "WRONG_MACHINE", "machine": "other"}):
            with self.assertRaisesRegex(unit.UnitFailure, "unknown is not idle"):
                unit.inventory_command()
            self.assertEqual(json.loads((Path(scratch) / unit.INVENTORY_FILE).read_text(encoding="utf-8"))["status"], "WRONG_MACHINE")
        with self.assertRaises(SystemExit): unit.main(["inventory", "--pid", "7256"])
        with self.assertRaises(SystemExit): unit.main(["inventory", "--command", "whoami"])


class FixedCommandTests(unittest.TestCase):
    def test_the_commands_and_bounds_are_the_reviewed_constants(self):
        self.assertEqual(
            unit.COMPILE_ARGV,
            ["cargo", "test", "--locked", "--lib", "-p", "candle-gen-moss-image", "--features",
             "cuda", "--no-run", "--message-format=json"],
        )
        self.assertEqual(
            unit.RUN_ARGS,
            [unit.TEST_NAME, "--exact", "--nocapture", "--color", "never", "--test-threads=1"],
        )
        self.assertEqual((unit.DEADLINE_SECONDS, unit.RSS_CAP_BYTES), (300.0, 6 * 1024**3))

    def test_the_fixed_test_exists_unignored_in_the_package_source(self):
        source = (unit.REPO_ROOT / "crates/media/candle-gen/candle-gen-moss-image/src/tests/training_contract_tests.rs").read_text(
            encoding="utf-8"
        )
        head = source.index("fn cuda_production_graph_four_cases_checkpoint_gradients_updates_and_strict_reload")
        attributes = source[source.rindex("\n}\n", 0, head):head]
        self.assertIn('#[cfg(feature = "cuda")]', attributes)
        self.assertIn("#[test]", attributes)
        self.assertNotIn("ignore", attributes)
        body = source[head:source.index("\n}\n", head)]
        self.assertIn("Device::new_cuda(0).expect(", body)
        self.assertIn("is_cuda()", body)
        self.assertEqual(unit.TEST_NAME.rpartition("::")[0], "tests::training_contract_tests")

    def test_a_missing_executable_record_fails_the_run_command(self):
        with tempfile.TemporaryDirectory() as scratch:
            record = Path(scratch) / "exe"
            record.write_text(str(Path(scratch) / "absent"), encoding="utf-8")
            self.assertEqual(unit.main(["run", "--exe-file", str(record)]), 1)


if __name__ == "__main__":
    unittest.main()
