"""Source/routing and proof-protocol tests only; never compiles or runs a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.ci import llm_decoder_fixture as decoder
from scripts.ci.moss_cuda_unit import UnitFailure
from scripts.tests import test_moss_fixture_cpu_profile as moss_tests

ROOT = Path(__file__).resolve().parents[2]


class DecoderRoutingTests(unittest.TestCase):
    def workflow(self):
        return yaml.load((ROOT / ".github/workflows/moss-model-fixtures.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    def ordinary_workflow(self):
        return yaml.load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    def test_each_real_decoder_test_is_ignored_and_registered_without_a_population_gate(self):
        for backend, names in decoder.CASES.items():
            text = (ROOT / decoder.SOURCES[backend]).read_text(encoding="utf-8")
            ignored = set(re.findall(r'#\[ignore = "' + re.escape(decoder.IGNORE_REASON) + r'"\]\s*fn (\w+)\s*\(', text))
            self.assertTrue({name.rsplit("::", 1)[-1] for name in names} <= ignored)
            self.assertTrue(names)
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(name.startswith("models::qwen35::tests::") for name in names))
        declared = self.workflow()["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(declared), set(decoder.DISPATCH_INPUT_CHOICES))
        for name, choices in decoder.DISPATCH_INPUT_CHOICES.items():
            self.assertEqual(tuple(declared[name]["options"]), choices)
            self.assertTrue(declared[name]["required"])

    def test_mixed_or_unknown_dispatch_selectors_are_rejected_before_compilation(self):
        for choice in decoder.DISPATCH_CHOICES:
            with mock.patch.dict(os.environ, {"MOSS_FIXTURE_SELECTED_LANE": choice}):
                self.assertEqual(decoder.validate_dispatch("workflow_dispatch", {"inputs": {"lanes": choice}}), choice)
                self.assertEqual(decoder.validate_dispatch("workflow_dispatch", {"inputs": {"profile": choice}}), choice)
                for malformed in ({"lanes": choice, "profile": choice}, {"lanes": "unknown"},
                                  {"profile": "unknown"}, {"lanes": [choice]}, {"lanes": "all"}):
                    with self.subTest(malformed=malformed), self.assertRaises(UnitFailure):
                        decoder.validate_dispatch("workflow_dispatch", {"inputs": malformed})
                with mock.patch.dict(os.environ, {"MOSS_FIXTURE_SELECTED_LANE": "moss-cuda-unit"}):
                    if choice != "moss-cuda-unit":
                        with self.assertRaises(UnitFailure):
                            decoder.validate_dispatch("workflow_dispatch", {"inputs": {"lanes": choice}})
        self.assertIsNone(decoder.validate_dispatch("pull_request", {"inputs": {"lanes": "invalid"}}))
    def test_only_narrow_hosted_rows_activate_and_guards_precede_toolchain(self):
        workflow = self.workflow()
        job = workflow["jobs"][decoder.LANE]
        self.assertEqual(job["if"], "github.event_name == 'workflow_dispatch' && inputs.lanes == 'llm-decoder-fixture'")
        self.assertEqual(job["strategy"]["matrix"]["include"], [
            {"backend": "candle", "runner": "ubuntu-latest"}, {"backend": "mlx", "runner": "macos-15"}])
        steps = job["steps"]
        guard = next(i for i,s in enumerate(steps) if s.get("name") == "Reject conflicting decoder selectors before toolchain setup")
        tool = next(i for i,s in enumerate(steps) if s.get("name") == "Install pinned Rust toolchain")
        self.assertLess(guard, tool)
        self.assertEqual(steps[guard]["run"], "python3 scripts/ci/llm_decoder_fixture.py preflight")
        self.assertEqual(next(s["run"] for s in steps if s.get("name") == "Execute only registered shared decoder model cases"),
                         "python3 scripts/ci/llm_decoder_fixture.py run ${{ matrix.backend }}")
        upload = next(s for s in steps if s.get("name") == "Upload exact-head shared decoder evidence")
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertNotIn(decoder.LANE, self.ordinary_workflow()["jobs"])
        bridge = yaml.load((ROOT / ".github/workflows/real-weights.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["jobs"]["moss-fixture-bridge"]
        self.assertIn("inputs.profile == 'llm-decoder-fixture'", bridge["if"])
        self.assertEqual(bridge["uses"], "./.github/workflows/moss-model-fixtures.yml")
    def test_ordinary_package_tests_keep_compilation_but_never_include_ignored(self):
        jobs = self.ordinary_workflow()["jobs"]
        for job, package in [("candle-cpu-test", "candle-llm"), ("macos-metal", "mlx-llm")]:
            commands = [step.get("run", "") for step in jobs[job]["steps"]]
            self.assertTrue(any("cargo test" in command and "-p " + package in command for command in commands))
            for command in commands:
                if "cargo test" in command:
                    self.assertNotRegex(command, r"--(?:include-ignored|ignored)\b")
                self.assertNotIn("llm_decoder_fixture.py run", command)
        for job, package in [("candle-cpu-lint", "candle-llm"), ("macos-metal", "mlx-llm")]:
            self.assertTrue(any("cargo clippy" in step.get("run", "") and "-p " + package in step["run"]
                                and "--all-targets" in step["run"] for step in jobs[job]["steps"]))

    def test_requested_matrix_skip_or_failure_never_makes_gate_green(self):
        workflow = self.workflow()
        job = workflow["jobs"][decoder.LANE]
        self.assertEqual(job["strategy"]["fail-fast"], "false")
        self.assertNotIn("continue-on-error", job)
        self.assertEqual(job["timeout-minutes"], "120")
        self.assertEqual(set(workflow["on"]), {"workflow_call", "workflow_dispatch"})
        self.assertNotIn(decoder.LANE, self.ordinary_workflow()["jobs"]["gate"]["needs"])
    def test_resource_metadata_missing_zero_or_wrong_cannot_pass(self):
        good = {"observed_peak_child_rss_bytes": 1234, "observed_elapsed_seconds": 0.5,
                "configured_rss_cap_bytes": decoder.shared.RSS_CAP,
                "configured_deadline_seconds": decoder.shared.DEADLINE, "owned_released": True}
        self.assertTrue(decoder.resources_verified(good))
        for key, bad in [("observed_peak_child_rss_bytes", None), ("observed_peak_child_rss_bytes", 0),
                         ("observed_peak_child_rss_bytes", decoder.shared.RSS_CAP + 1),
                         ("observed_elapsed_seconds", float("nan")), ("configured_deadline_seconds", 181),
                         ("owned_released", False)]:
            self.assertFalse(decoder.resources_verified({**good, key: bad}))

    def test_named_execution_is_nonignored_and_assertions_aggregate_without_false_green(self):
        for mode in ["pass", "assertion", "wrong-source", "missing-rss"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); binary = root / "binary"; binary.write_bytes(b"protocol-only artifact")
                digest = hashlib.sha256(binary.read_bytes()).hexdigest()
                stamp = {"head": "a"*40, "tracked_delta_sha256": "b"*64}
                calls = []
                def execute(path, name, log):
                    calls.append(name)
                    row = {"test": name, "status": "passed", "source": stamp, "binary_sha256": digest,
                           "observed_peak_child_rss_bytes": 123, "observed_elapsed_seconds": .5,
                           "configured_rss_cap_bytes": decoder.shared.RSS_CAP,
                           "configured_deadline_seconds": decoder.shared.DEADLINE, "owned_released": True}
                    if mode == "assertion" and len(calls) == 1: row["status"] = "assertion_failed"
                    if mode == "wrong-source": row["source"] = {}
                    if mode == "missing-rss": row["observed_peak_child_rss_bytes"] = None
                    log.with_suffix(".result.json").write_text(json.dumps(row), encoding="utf-8")
                    if row["status"] == "assertion_failed": raise decoder.shared.CaseAssertionFailure(row)
                names = decoder.CASES["candle"]
                with mock.patch.object(decoder, "venue", return_value={"head":"a"*40}), mock.patch.object(
                        decoder.shared,"source_stamp",return_value=stamp), mock.patch.object(decoder,"compile_binary",return_value=(binary,{"binary_sha256":digest})), mock.patch.object(
                        decoder,"supervise",return_value="\n".join(name+": test" for name in names)), mock.patch.object(decoder.shared,"CpuOwner") as owner, mock.patch.object(
                        decoder.shared,"execute",side_effect=execute):
                    owner.return_value.collect.return_value = {}
                    if mode in ("missing-rss", "wrong-source"):
                        message = "RSS/deadline/release" if mode == "missing-rss" else "protocol identity"
                        with self.assertRaisesRegex(UnitFailure,message):
                            decoder.run("candle", root/"evidence")
                        self.assertEqual(calls, list(names[:1]))
                    else:
                        proof=decoder.run("candle",root/"evidence")
                        self.assertEqual(proof["passed"], mode == "pass")
                        self.assertEqual(calls,list(names))
                saved=json.loads((root/"evidence/proof.json").read_text(encoding="utf-8"))
                self.assertTrue(saved["source_unchanged"])
                self.assertEqual(saved["passed"], mode == "pass")

    def test_compile_selects_exact_package_lib_and_never_treats_compile_failure_as_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); stamp={"head":"a"*40}
            result=mock.Mock(returncode=101,stdout='{"reason":"build-finished","success":false}\n')
            with mock.patch.object(decoder.subprocess,"run",return_value=result) as run, mock.patch.object(decoder.shared,"source_stamp",return_value=stamp), mock.patch.object(decoder.shared,"select_binary") as select:
                with self.assertRaisesRegex(UnitFailure,"compile failure is not model execution"):
                    decoder.compile_binary("mlx",root,stamp)
                command=run.call_args.args[0]
                self.assertIn("mlx-llm",command)
                self.assertIn("--lib",command)
                self.assertIn("--no-run",command)
                self.assertIn("--locked",command)
                select.assert_not_called()


if __name__ == "__main__":
    unittest.main()
