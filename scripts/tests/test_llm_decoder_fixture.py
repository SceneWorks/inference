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
        return yaml.load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    def test_each_real_decoder_test_is_ignored_and_registered_without_a_population_gate(self):
        for backend, names in decoder.CASES.items():
            text = (ROOT / decoder.SOURCES[backend]).read_text(encoding="utf-8")
            ignored = set(re.findall(r'#\[ignore = "' + re.escape(decoder.IGNORE_REASON) + r'"\]\s*fn (\w+)\s*\(', text))
            self.assertEqual(ignored, {name.rsplit("::", 1)[-1] for name in names})
            self.assertTrue(names)
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(name.startswith("models::qwen35::tests::") for name in names))
        declared = self.workflow()["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(declared), set(decoder.DISPATCH_INPUT_CHOICES))
        for name, choices in decoder.DISPATCH_INPUT_CHOICES.items():
            self.assertEqual(tuple(declared[name]["options"]), choices)
            self.assertEqual(declared[name]["default"], choices[0])

    def test_mixed_or_unknown_dispatch_selectors_are_rejected_before_compilation(self):
        for choice in decoder.DISPATCH_CHOICES:
            self.assertEqual(decoder.validate_dispatch("workflow_dispatch", {"inputs": {"lanes": choice}}), choice)
        for event in [
            {"inputs": {"lanes": "llm-decoder-fixture,moss-metal-fixture"}},
            {"inputs": {"lanes": "all llm-decoder-fixture"}},
            {"inputs": {"lanes": ["llm-decoder-fixture", "all"]}},
            {"inputs": {"lanes": "llm-decoder-fixture", "backend": "cuda"}},
            {"inputs": {"lanes": "unknown"}},
        ]:
            with self.subTest(event=event), self.assertRaisesRegex(UnitFailure, "selector|select"):
                decoder.validate_dispatch("workflow_dispatch", event)
        for lane in decoder.DISPATCH_CHOICES:
            self.assertEqual(decoder.validate_dispatch("workflow_dispatch", {"inputs": {
                "lanes": lane, "test_suite": "routine", "synthetic_w4a8": "none"}}), lane)
        for suite in decoder.DISPATCH_INPUT_CHOICES["test_suite"]:
            for synthetic in decoder.DISPATCH_INPUT_CHOICES["synthetic_w4a8"]:
                for lane in ("all", "windows-cuda"):
                    self.assertEqual(decoder.validate_dispatch("workflow_dispatch", {"inputs": {
                        "lanes": lane, "test_suite": suite, "synthetic_w4a8": synthetic}}), lane)
        for inputs in ({"test_suite": "unknown"}, {"synthetic_w4a8": ["mlx", "cuda"]},
                       {"lanes": decoder.LANE, "synthetic_w4a8": "mlx"},
                       {"lanes": "moss-cpu-fixture", "test_suite": "exhaustive"}):
            with self.subTest(inputs=inputs), self.assertRaisesRegex(UnitFailure, "selector|select"):
                decoder.validate_dispatch("workflow_dispatch", {"inputs": inputs})
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
        helper = moss_tests.HostedCpuRoutingTests("test_exact_manual_choice_and_only_two_fixed_hosted_rows")
        selected = helper.select(helper.selector(), "workflow_dispatch", decoder.LANE)
        self.assertEqual({key for key,value in selected.items() if value == "true"}, {"workspace"})
        addition = '\nelif [[ "$EVENT_NAME" == "workflow_dispatch" && "$DISPATCH_LANES" == "llm-decoder-fixture" ]]; then\n  python3 scripts/ci/select_lanes.py --only workspace --github-output "$GITHUB_OUTPUT"'
        selector = helper.selector()
        self.assertEqual(selector.count(addition), 1)
        before = selector.replace(addition, "")
        for event, ref in [("pull_request", "branch"), ("push", "branch"), ("push", "tag"), ("merge_group", "branch")]:
            self.assertEqual(helper.select(selector, event, decoder.LANE, ref), helper.select(before, event, decoder.LANE, ref))
        changes = workflow["jobs"]["changes"]["steps"]
        root_guard = next(i for i,s in enumerate(changes) if s.get("name") == "Validate explicit dispatch selectors before affected runtime lanes")
        select = next(i for i,s in enumerate(changes) if s.get("id") == "select")
        self.assertLess(root_guard, select)

    def test_ordinary_package_tests_keep_compilation_but_never_include_ignored(self):
        jobs = self.workflow()["jobs"]
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
        gate = self.workflow()["jobs"]["gate"]
        self.assertIn(decoder.LANE, gate["needs"])
        for result in ["success", "skipped", "failure", "cancelled"]:
            env = dict(os.environ, RESULTS="success skipped", REQUIRE_MOSS_CUDA_UNIT="false",
                       REQUIRE_MOSS_CPU_FIXTURE="false", REQUIRE_MOSS_METAL_FIXTURE="false",
                       REQUIRE_LLM_DECODER_FIXTURE="true", LLM_DECODER_FIXTURE_RESULT=result)
            proc = subprocess.run(["bash", "-c", gate["steps"][0]["run"]], env=env, capture_output=True)
            self.assertEqual(proc.returncode == 0, result == "success")


class DecoderEvidenceTests(unittest.TestCase):
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
