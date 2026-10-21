"""CPU-only tests for the MOSS CUDA unit supervisor; no Cargo, GPU or model is started."""

import json
import copy
import math
from unittest import mock
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest
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
    def test_actual_uuid_accepts_prefix_interleaved_or_standalone_but_not_missing_duplicate_wrong(self):
        uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        marker = f"MOSS_CUDA_CONTEXT_UUID={uuid}"
        interleaved = OK_OUTPUT.replace(" ... ok", f" ... {marker}\nok")
        standalone = OK_OUTPUT.replace(" ... ok", f" ... \n{marker}\nok")
        for valid in (interleaved, standalone, interleaved.replace("\n", "\r\n"), standalone.replace("\n", "\r\n")):
            try:
                unit.verify_output(valid, uuid)
            except unit.UnitFailure as error:
                self.fail(f"valid UUID output form was rejected: {error}")
        for output in (OK_OUTPUT, interleaved + marker + "\n", interleaved.replace(uuid, "GPU-ffeeddcc-bbaa-9988-7766-554433221100"), interleaved.replace(uuid, "GPU-bad"), interleaved.replace(" ... ", "wrong ... ")):
            with self.assertRaises(unit.UnitFailure): unit.verify_output(output, uuid)

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


class Gpu0ResourceTests(unittest.TestCase):
    UUID = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    OTHER = "GPU-ffeeddcc-bbaa-9988-7766-554433221100"

    def check(self, gpus=None, failure=None):
        query = mock.Mock(return_value=gpus or f"0, {self.UUID}, 8192, 16384\n1, {self.OTHER}, N/A, N/A\n", side_effect=failure)
        result = unit.require_gpu0_budget(query)
        query.assert_called_once_with("--query-gpu=index,uuid,memory.free,memory.total")
        return result

    def test_desktop_processes_do_not_define_admission_and_exact_free_floor_passes(self):
        # WDDM reports DWM/desktop apps in compute-apps. That table is not consulted at all.
        try:
            result = self.check()
        except unit.UnitFailure as error:
            self.fail(f"valid desktop-compatible resources were refused: {error}")
        self.assertEqual(result["uuid"], self.UUID)
        floor = result["budget"]["required_bytes"] // unit.MIB
        try:
            self.check(f"0, {self.UUID}, {floor}, 16384\n")
        except unit.UnitFailure as error:
            self.fail(f"exact free-VRAM floor was refused: {error}")
        with self.assertRaisesRegex(unit.UnitFailure, "below unit requirement"):
            self.check(f"0, {self.UUID}, {floor - 1}, 16384\n")

    def test_unknown_malformed_ambiguous_and_insufficient_resource_fail_closed(self):
        rows = [" ", f"1, {self.OTHER}, 5000, 6000", f"0, {self.UUID}, 5000, 6000\n0, {self.OTHER}, 5000, 6000",
                f"0, {self.UUID}, 5000, 6000\n1, {self.UUID}, 5000, 6000", "0, GPU-bad, 5000, 6000",
                f"0, {self.UUID}, N/A, 6000", f"0, {self.UUID}, nan, 6000", f"0, {self.UUID}, inf, 6000",
                f"0, {self.UUID}, -1, 6000", f"0, {self.UUID}, 7000, 6000", f"0, {self.UUID}, 5000, inf", f"0, {self.UUID}, 5000, N/A", f"0, {self.UUID}, 5000"]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(unit.UnitFailure): self.check(row)
        for failure in (OSError("denied"), RuntimeError("trusted tool unavailable"), unit.subprocess.TimeoutExpired("nvml", 15)):
            with self.assertRaises(unit.UnitFailure): self.check(failure=failure)

    def test_nonzero_exit_refused_despite_valid_resource_stdout(self):
        data = f"0, {self.UUID}, 8192, 16384\n"
        with mock.patch.object(unit, "smi_path", return_value="trusted-nvml"), mock.patch.object(unit.subprocess, "run", return_value=unit.subprocess.CompletedProcess([], 0, stdout=data)) as run:
            unit.require_gpu0_budget()
            run.return_value = unit.subprocess.CompletedProcess([], 9, stdout=data)
            with self.assertRaisesRegex(unit.UnitFailure, "exited with 9"): unit.require_gpu0_budget()

    def test_actual_shared_fixture_defines_monotonic_exact_ledger_and_growth(self):
        resource, fixture, dmd = unit.read_unit_resources()
        before = unit.derive_unit_budget(resource, fixture, dmd)
        a = resource["accounting"]
        expected_transient = a["intermediates_per_block"] * a["retained_passes"] * a["value_adjoint_slots"] * before["blocks"] * a["case_cache_allowance"] * before["largest_tensor_bytes"]
        self.assertEqual(before["transient_bytes"], expected_transient)
        self.assertEqual(before["unrounded_bytes"], expected_transient + before["persistent_bytes"])
        self.assertGreater(before["unrounded_bytes"], 300 * unit.MIB)
        self.assertGreaterEqual(before["tensor_bytes"], before["unrounded_bytes"])
        self.assertEqual(before["tensor_bytes"], 512 * unit.MIB)
        self.assertEqual(before["required_bytes"], before["tensor_bytes"] + 1024 * unit.MIB)
        # Change the ACTUAL author-fixture input, preserving every shape/value population and
        # unchanged model weights/config. Spatial growth is valid for this same transformer.
        grown = copy.deepcopy(fixture)
        tensor = grown["inputs"]["hidden_states"]
        tensor["shape"][2] *= 2; tensor["shape"][3] *= 2
        tensor["values"] *= 4
        self.assertEqual(len(tensor["values"]), math.prod(tensor["shape"]))
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "actual-author-input.json"
            path.write_text(json.dumps(grown), encoding="utf-8")
            after = unit.derive_unit_budget(resource, json.loads(path.read_text(encoding="utf-8")), dmd)
        self.assertGreater(after["unrounded_bytes"], before["unrounded_bytes"])
        self.assertGreater(after["required_bytes"], before["required_bytes"])
        malformed = copy.deepcopy(fixture); malformed["inputs"]["hidden_states"]["shape"][2] *= 2
        with self.assertRaises(unit.UnitFailure): unit.derive_unit_budget(resource, malformed, dmd)
        heavier = copy.deepcopy(fixture)
        heavier["weights"]["accounted_extra"] = {"shape": [1024], "values": [0.] * 1024}
        self.assertGreater(unit.derive_unit_budget(resource, heavier, dmd)["unrounded_bytes"], before["unrounded_bytes"])
        higher_rank = dict(resource, rank=resource["rank"] * 2)
        self.assertGreater(unit.derive_unit_budget(higher_rank, fixture, dmd)["unrounded_bytes"], before["unrounded_bytes"])

    def test_budget_refusal_precedes_owner_and_spawn(self):
        with tempfile.TemporaryDirectory() as scratch, mock.patch.object(unit, "os", types.SimpleNamespace(name="nt")), mock.patch.object(unit, "provenance", return_value="x"), mock.patch.object(unit, "require_gpu0_budget", side_effect=unit.UnitFailure("insufficient")), mock.patch.object(unit, "WindowsOwner") as owner, mock.patch.object(unit, "supervise") as supervise:
            record = Path(scratch) / "exe"; exe = Path(scratch) / "t"
            exe.write_text("x", encoding="utf-8"); record.write_text(str(exe), encoding="utf-8")
            with self.assertRaisesRegex(unit.UnitFailure, "insufficient"): unit.run_command(record)
            supervise.assert_not_called(); owner.assert_not_called()

    def test_fresh_uuid_binding_is_child_only_and_overrides_inherited_values(self):
        with tempfile.TemporaryDirectory() as scratch, mock.patch.object(unit, "os", types.SimpleNamespace(name="nt", environ=dict(os.environ, CUDA_VISIBLE_DEVICES="stale", CUDA_DEVICE_ORDER="FASTEST_FIRST", MOSS_CUDA_UNIT_EXPECTED_GPU_UUID=self.OTHER))), mock.patch.object(unit, "provenance", return_value="x"), mock.patch.object(unit, "require_gpu0_budget", return_value={"uuid": self.UUID, "free_bytes": 9999999999, "budget": {"required_bytes": 1610612736}}), mock.patch.object(unit, "WindowsOwner"), mock.patch.object(unit, "supervise", return_value=OK_OUTPUT.replace(" ... ok", f" ... MOSS_CUDA_CONTEXT_UUID={self.UUID}\nok")) as supervise:
            record = Path(scratch) / "exe"; exe = Path(scratch) / "t"
            exe.write_text("x", encoding="utf-8"); record.write_text(str(exe), encoding="utf-8")
            before = dict(unit.os.environ)
            unit.run_command(record)
            self.assertIn("env", supervise.call_args.kwargs)
            child = supervise.call_args.kwargs["env"]
            self.assertEqual(child["CUDA_VISIBLE_DEVICES"], self.UUID)
            self.assertEqual(child["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(child[unit.EXPECTED_UUID_ENV], self.UUID)
            self.assertEqual(unit.os.environ, before)




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
