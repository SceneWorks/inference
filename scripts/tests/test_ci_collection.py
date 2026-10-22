"""Failure collection uses fake subprocesses; these tests never invoke Cargo."""

import contextlib
import errno
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.ci import run_ci_collection as collection


ROOT = Path(__file__).resolve().parents[2]
SETUP = ("mlx_checkout", "mlx_versions", "mlx_metal", "mlx_rust", "mlx_prebuilt", "mlx_setup")
GUARD = "${{ !cancelled() && " + " && ".join(
    f"steps.{name}.outcome == 'success'" for name in SETUP
) + " && env.CI_COLLECTION_RESOURCE_FAILURE != 'true' && env.CI_COLLECTION_NATIVE_FATAL != 'true' }}"


class CollectionTests(unittest.TestCase):
    def fake_command(self, code, output):
        process = type("Process", (), {"stdout": io.StringIO(output), "wait": lambda self: code})()
        stream = io.StringIO()
        with patch.object(collection.subprocess, "Popen", return_value=process) as execute, \
                contextlib.redirect_stdout(stream):
            actual = collection.run_command(["cargo", "test", "--unchanged"])
        self.assertEqual(execute.call_args.args[0], ["cargo", "test", "--unchanged"])
        return actual, stream.getvalue()

    def test_success_and_ordinary_failure_keep_actual_exit_codes(self):
        for code in (0, 7, 101):
            with self.subTest(code=code):
                actual, _ = self.fake_command(code, "ordinary result\n")
                self.assertEqual(actual, code)

    def test_combined_command_is_forwarded_without_package_or_feature_rewriting(self):
        command = ["cargo", "test", "--locked", "--keep-going", "--no-fail-fast",
                   "--lib", "--tests", "-p", "mlx-gen-iris", "-p", "mlx-gen-sdxl",
                   "-p", "mlx-gen-z-image"]
        process = type("Process", (), {"stdout": io.StringIO(""), "wait": lambda self: 0})()
        with patch.object(collection.subprocess, "Popen", return_value=process) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collection.run_command(command), 0)
        self.assertEqual(execute.call_args.args[0], command)

    def test_compile_failure_is_reported_as_blocked_not_executed(self):
        code, output = self.fake_command(101, "error: could not compile `mlx-gen-iris`\n")
        self.assertEqual(code, 101)
        self.assertIn("dependent test binaries are compile-blocked, not executed", output)
        self.assertIn("compile_blocked=True", output)

    def test_resource_or_native_poison_blocks_downstream_and_stays_red(self):
        cases = ((0, "No space left on device\n", "CI_COLLECTION_RESOURCE_FAILURE"),
                 (1, "Disk quota exceeded\n", "CI_COLLECTION_RESOURCE_FAILURE"),
                 (-9, "", "CI_COLLECTION_RESOURCE_FAILURE"),
                 (0, "SubmissionsIgnored\n", "CI_COLLECTION_NATIVE_FATAL"))
        for code, output, flag in cases:
            with self.subTest(code=code, output=output), tempfile.TemporaryDirectory() as tmp:
                env, summary = Path(tmp) / "env", Path(tmp) / "summary"
                with patch.dict(os.environ, {"GITHUB_ENV": str(env), "GITHUB_STEP_SUMMARY": str(summary)}):
                    actual, _ = self.fake_command(code, output)
                self.assertNotEqual(actual, 0)
                self.assertEqual(env.read_text(), f"{flag}=true\n")
                self.assertIn("**blocked**", summary.read_text())
                self.assertIn("not an independent test failure", summary.read_text())

    def test_recoverable_metal_error_does_not_claim_device_poison(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / "env"
            with patch.dict(os.environ, {"GITHUB_ENV": str(env)}):
                code, _ = self.fake_command(7, "[METAL] Command buffer execution failed: synthetic\n")
            self.assertEqual(code, 7)
            self.assertFalse(env.exists())

    def test_missing_command_and_quota_errors_remain_failed(self):
        for error, blocked in ((FileNotFoundError(2, "missing command"), False),
                               (OSError(errno.EDQUOT, "quota"), True)):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                env = Path(tmp) / "env"
                with patch.dict(os.environ, {"GITHUB_ENV": str(env)}), \
                        patch.object(collection.subprocess, "Popen", side_effect=error), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = collection.run_command(["fake-command"])
                self.assertNotEqual(code, 0)
                self.assertEqual(env.exists(), blocked)

    def test_poisoned_collection_never_starts_another_command(self):
        for flag in ("CI_COLLECTION_RESOURCE_FAILURE", "CI_COLLECTION_NATIVE_FATAL"):
            with patch.object(collection.sys, "argv", ["collector", "--", "fake-command"]), \
                    patch.dict(os.environ, {flag: "true"}), \
                    patch.object(collection, "run_command") as execute, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(collection.main(), 1)
                execute.assert_not_called()

    def test_empty_command_cannot_report_success(self):
        with patch.object(collection.sys, "argv", ["collector"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            collection.main()
        self.assertEqual(error.exception.code, 2)

    def test_existing_lane_requires_all_setup_and_keeps_failing_steps_red(self):
        job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]["macos-metal"]
        self.assertEqual(job["runs-on"], "macos-15")
        steps = job["steps"]
        for name in SETUP:
            self.assertEqual(sum(step.get("id") == name for step in steps), 1)
        collectors = [step for step in steps if "run_ci_collection.py" in step.get("run", "")]
        self.assertGreater(len(collectors), 10)
        for step in collectors:
            self.assertEqual(" ".join(step.get("if", "").split()), GUARD, step["name"])
            self.assertFalse(step.get("continue-on-error", False), step["name"])
            run = step["run"]
            for forbidden in ("--per-package", "--ignored", "--list", "--all-features", "|| true"):
                self.assertNotIn(forbidden, run)
            if "cargo clippy " in run or " clippy --locked " in run:
                self.assertIn("--keep-going", run)
                self.assertTrue(run.rstrip().endswith("-- -D warnings"))
        broad = next(step["run"] for step in steps if step.get("name") == "Test MLX packages")
        for name in ("sceneworks-gen-core", "mlx-gen", "mlx-llm", "mlx-llm-server",
                     "runtime-ios", "runtime-macos"):
            self.assertIn("-p " + name, broad)
        self.assertIn("-p 'mlx-gen-*'", broad)
        self.assertIn("--no-fail-fast", broad)
        self.assertEqual(broad.count("cargo test"), 1)
        self.assertFalse(any(step.get("name") in (
            "Compile affected iOS pricing test targets", "Test affected iOS pricing packages",
            "Collect affected iOS pricing package tests",
        ) for step in steps))


if __name__ == "__main__":
    unittest.main()
