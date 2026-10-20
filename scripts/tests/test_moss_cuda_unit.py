"""CPU-only tests for the MOSS CUDA unit supervisor; no Cargo, GPU or model is started."""

import json
import os
import sys
import tempfile
import textwrap
import time
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
        self.exe.write_text("x")

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
        other.write_text("x")
        outside = self.dir / "elsewhere"
        outside.write_text("x")
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
        self.assert_dead(int(pid_file.read_text()))
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
            record.write_text(sys.executable)
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
            record.write_text(str(Path(scratch) / "absent"))
            self.assertEqual(unit.main(["run", "--exe-file", str(record)]), 1)


if __name__ == "__main__":
    unittest.main()
