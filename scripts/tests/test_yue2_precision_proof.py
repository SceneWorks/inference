"""CPU-only controls for the dispatch-only precision proof."""
import importlib.util
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
import copy
import base64
import os
import subprocess
import time
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_cuda_idle_context as IDLE
import yue2_precision_reference_transfer as TRANSFER
SPEC = importlib.util.spec_from_file_location("yue2_precision_proof", ROOT / "scripts/ci/yue2_precision_proof.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)
WORKFLOW = ROOT / ".github/workflows/yue2-precision-proof.yml"


class PrecisionControlTests(unittest.TestCase):
    @staticmethod
    def concurrency_settings(source):
        block = re.search(r"(?m)^concurrency:\n((?: +[^\n]*\n)+)", source)
        if block is None:
            raise AssertionError("workflow concurrency block missing")
        return dict(line.strip().split(": ", 1) for line in block[1].splitlines())

    @staticmethod
    def concurrency_group(group, stage, run_id, scheduling="shared-host", receipt="", engine=""):
        # A restricted GitHub &&/|| interpreter reads both workflow expressions.
        # It has no scheduling policy of its own; the tests below specify routing.
        if not (group.startswith("${{ ") and group.endswith(" }}")):
            raise AssertionError("unexpected precision concurrency expression")
        values = {"stage": stage, "backend": stage, "cuda_scheduling_mode": scheduling,
                  "idle_cuda_context_run_id": receipt, "expected_engine_sha": engine}
        for branch in group[4:-3].split(" || "):
            clauses = branch.split(" && ")
            selected = True
            for clause in clauses[:-1]:
                match = re.fullmatch(r"inputs\.([a-z_]+) == '([^']*)'", clause)
                if match is None or match[1] not in values:
                    raise AssertionError("unsupported selector")
                selected &= values[match[1]] == match[2]
            if selected:
                result = clauses[-1]
                match = re.fullmatch(r"format\('([^']+)', github\.run_id\)", result)
                if match:
                    return match[1].format(run_id)
                match = re.fullmatch(r"'([^']+)'", result)
                if match:
                    return match[1]
                raise AssertionError("unsupported group")
        raise AssertionError("group expression has no default")

    def test_fixture_transfers_use_distinct_run_owned_cpu_groups(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        settings = self.concurrency_settings(source)
        group = settings["group"]
        first = self.concurrency_group(group, "fixture", "101")
        second = self.concurrency_group(group, "fixture", "102")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, "inference-real-weights-physical-host")
        fixture = source.split("  reference:\n", 1)[1].split("\n  cuda:", 1)[0]
        self.assertIn("    if: inputs.stage == 'fixture'\n", fixture)
        self.assertIn("    runs-on: ubuntu-latest\n", fixture)

    def test_accelerators_retain_their_exact_physical_host_groups(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        settings = self.concurrency_settings(source)
        group = settings["group"]
        for stage in ("cuda", "metal", "cuda-diagnostic", "", "unknown"):
            for run_id in ("101", "102"):
                with self.subTest(stage=stage, run_id=run_id):
                    self.assertEqual(self.concurrency_group(group, stage, run_id),
                                     "yue2-app-precision-nax-macos-2" if stage == "metal" else
                                     "inference-real-weights-physical-host")
        # Keep all other workflow users of the accelerator lock byte-consistent.
        for name in ("real-weights.yml", "real-weights-yue.yml", "yue2-bf16-tile-diagnostic.yml",
                     "ltx25-quant-campaign.yml", "ltx25-quant-promotion.yml"):
            other = self.concurrency_settings(WORKFLOW.with_name(name).read_text(encoding="utf-8"))
            with self.subTest(workflow=name):
                self.assertEqual(other["group"],
                                 self.concurrency_group(group, "cuda", "101"))
                self.assertEqual(other["cancel-in-progress"], "false")
        app = self.concurrency_settings(
            WORKFLOW.with_name("yue2-app-precision-profile.yml").read_text(encoding="utf-8"))
        for backend, expected in (("cuda", "inference-real-weights-physical-host"),
                                  ("metal", "yue2-app-precision-nax-macos-2")):
            self.assertEqual(self.concurrency_group(app["group"], backend, "101"), expected)
        self.assertEqual(app["cancel-in-progress"], "false")
        self.assertEqual(self.concurrency_group(group, "metal", "101"),
                         self.concurrency_group(app["group"], "metal", "102"))

    def test_precision_queue_preserves_existing_pending_and_running_work(self):
        settings = self.concurrency_settings(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(settings["queue"], "max")
        self.assertEqual(settings["cancel-in-progress"], "false")

    def test_optional_app_sha_is_absent_when_empty_and_one_argument_when_set(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        cuda = workflow.split("      - name: Run two exact CUDA smokes then the unchanged precision test with external sampling\n", 1)[1].split("      - name: Upload raw CUDA proof", 1)[0]
        metal = workflow.split("      - name: Run exactly one Metal precision test with external sampling\n", 1)[1].split("      - name: Upload raw Metal proof", 1)[0]

        def conditional_flags(cuda_source, metal_source):
            self.assertRegex(cuda_source, r"\$proofArgs = @\('run', '--backend', 'cuda'[^\n]*\)")
            self.assertIn("if ($env:EXPECTED_APP_SHA) { $proofArgs += @('--app-sha', $env:EXPECTED_APP_SHA) }", cuda_source)
            self.assertIn("yue2_precision_proof.py @proofArgs", cuda_source)
            self.assertRegex(metal_source, r"proof_args=\(run --backend metal[^\n]*\)")
            self.assertIn('if [[ -n "$EXPECTED_APP_SHA" ]]; then proof_args+=(--app-sha "$EXPECTED_APP_SHA"); fi', metal_source)
            self.assertIn('yue2_precision_proof.py "${proof_args[@]}"', metal_source)

        conditional_flags(cuda, metal)
        with self.assertRaises(AssertionError):
            conditional_flags(cuda.replace("if ($env:EXPECTED_APP_SHA) { ", "", 1), metal)
        with self.assertRaises(AssertionError):
            conditional_flags(cuda, metal.replace('if [[ -n "$EXPECTED_APP_SHA" ]]; then ', "", 1))

        # Execute the Bash workflow body with a fake Python entrypoint to inspect
        # its real argument vector without loading a model or running the controller.
        lines = metal.split("        run: |\n", 1)[1].splitlines()
        script = "\n".join(line[10:] for line in lines if line.startswith("          ")) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yue2-precision-proof").mkdir()
            (root / "yue2-precision-proof/binary.txt").write_text("test-binary\n", encoding="utf-8")
            stub = root / "python3.12"
            stub.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
            stub.chmod(0o755)
            base = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}", "RUNNER_TEMP": directory,
                    "YUE2_PRECISION_WORK_DIR": str(root / "listening"), "EXPECTED_ENGINE_SHA": "a" * 40,
                    "EXPECTED_CONTROL_SHA": "b" * 40}
            for app_sha in ("", "c" * 40):
                result = subprocess.run(["bash", "-e"], input=script, text=True, encoding="utf-8", capture_output=True,
                                        env={**base, "EXPECTED_APP_SHA": app_sha}, check=True)
                argv = json.loads(result.stdout)
                self.assertEqual(argv.pop(0), "../control/scripts/ci/yue2_precision_proof.py")
                self.assertEqual(argv.count("--app-sha"), bool(app_sha))
                if app_sha:
                    self.assertEqual(argv[-2:], ["--app-sha", app_sha])
                else:
                    self.assertEqual(argv[-2:], ["--control-sha", "b" * 40])
                with patch.object(CONTROL, "execute") as execute, \
                     patch.object(sys, "argv", ["yue2_precision_proof.py", *argv]):
                    CONTROL.main()
                self.assertEqual(execute.call_args.args[0].app_sha, app_sha)

    def test_workflow_control_and_engine_revisions_are_independent_and_exact(self):
        engine, control = "a" * 40, "b" * 40
        result = type("Result", (), {"stdout": engine + "\n"})()
        with patch.dict("os.environ", {"GITHUB_SHA": control}), \
             patch.object(CONTROL.subprocess, "run", return_value=result):
            CONTROL.verify_revisions(engine, control)
            with self.assertRaisesRegex(RuntimeError, "control SHA differs"):
                CONTROL.verify_revisions(engine, "c" * 40)
            with self.assertRaisesRegex(RuntimeError, "engine checkout differs"):
                CONTROL.verify_revisions("c" * 40, control)
            with self.assertRaisesRegex(RuntimeError, "control SHA must be"):
                CONTROL.verify_revisions(engine, "short")

    def test_pmon_refuses_compute_and_mixed_compute_graphics(self):
        output = "# gpu pid type fb sm\n0 111 G 12 0\n0 222 C+G 0 0\n0 333 C 256 75\n"
        self.assertEqual(CONTROL.compute_capable_rows(output), ["0 222 C+G 0 0", "0 333 C 256 75"])
        self.assertEqual(CONTROL.compute_capable_rows("# gpu pid type fb sm\n0 111 G 12 0\n"), [])
        with self.assertRaisesRegex(RuntimeError, "typed process columns"):
            CONTROL.compute_capable_rows("0 333 C 256 75\n")
        for ambiguous in ("0 444 ? 0 0", "0 xyz C+G 0 0", "0 555", "0 - C 0 0", "1 444 C 0 0"):
            with self.subTest(ambiguous=ambiguous), self.assertRaises(RuntimeError):
                CONTROL.compute_capable_rows("# gpu pid type fb sm\n" + ambiguous + "\n")
        self.assertEqual(CONTROL.compute_capable_rows("# gpu pid type fb sm\n0 - - - -\n"), [])

    def test_cuda_pmon_fallback_refuses_any_query_process_and_fails_closed(self):
        failed = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "pmon unsupported"})()
        apps = type("Result", (), {"returncode": 0, "stdout": "222, C:\\cuda-test.exe\n", "stderr": ""})()
        with patch.object(CONTROL.subprocess, "run", side_effect=[failed, apps]) as run:
            raw, busy = CONTROL.cuda_census()
        self.assertIn("pmon unsupported", raw)
        self.assertEqual(busy, ["222, C:\\cuda-test.exe"])
        self.assertIn("--query-compute-apps=pid,process_name", run.call_args_list[1].args[0])
        for bad in ("not a csv row", "222, ", "abc, C:\\cuda-test.exe"):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "ambiguous"):
                CONTROL.query_compute_apps_rows(bad)
        query_failed = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "unsupported"})()
        with patch.object(CONTROL.subprocess, "run", side_effect=[failed, query_failed]):
            with self.assertRaisesRegex(RuntimeError, "CUDA census unavailable"):
                CONTROL.cuda_census()

    def test_reviewed_idle_context_is_exact_and_expiring(self):
        fixture = json.loads((ROOT / "scripts/tests/fixtures/yue2-idle-context-redacted.json").read_text(encoding="utf-8"))
        baseline, fresh = fixture["baseline"], fixture["fresh"]
        self.assertEqual(len(baseline["counters"]["engine"]), 22)
        self.assertEqual(baseline["counters"]["processDedicated"], 19_341_312)
        self.assertEqual(baseline["gpu"]["usedMiB"], 19)
        IDLE.validate_current(fresh, baseline)
        mutations = (
            ("identity", lambda row: row["identity"].__setitem__(3, "different start")),
            ("pci", lambda row: row.__setitem__("pci", ":c1:00.0")),
            ("engine set", lambda row: row["counters"]["engine"].pop(next(iter(row["counters"]["engine"])))),
            ("activity", lambda row: row["counters"]["engine"].__setitem__(next(iter(row["counters"]["engine"])), 0.1)),
            ("process residency", lambda row: row["counters"].__setitem__("processDedicated", 19_341_313)),
            ("adapter residency", lambda row: row["counters"].__setitem__("adapterDedicated", 23_834_625)),
            ("NVML residency", lambda row: row["gpu"].__setitem__("usedMiB", 20)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(fresh)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    IDLE.validate_current(changed, baseline)
        completed = IDLE.parse_completed_utc(baseline["completedUtc"])
        IDLE.check_window(baseline["completedUtc"], completed + timedelta(hours=11))
        for bad in (completed - timedelta(seconds=1), completed + timedelta(hours=12, seconds=1)):
            with self.assertRaisesRegex(RuntimeError, "owner window"):
                IDLE.check_window(baseline["completedUtc"], bad)
        IDLE.check_device_selection("nt", "0", "PCI_BUS_ID")
        for platform, visible, order in (("posix", "0", "PCI_BUS_ID"), ("nt", "1", "PCI_BUS_ID"),
                                         ("nt", "0", None), ("nt", "0", "FASTEST_FIRST")):
            with self.subTest(platform=platform, visible=visible, order=order), \
                 self.assertRaisesRegex(RuntimeError, "PCI-ordered CUDA GPU0"):
                IDLE.check_device_selection(platform, visible, order)
        self.assertEqual(IDLE.RUN_ID, "37145909196")
        self.assertEqual(IDLE.BASELINE_DIGEST,
                         "d67f8d2c7cd79040bbe48ada71e6e27722237a3316bf622e69808ca341f77a2c")
        IDLE.check_dispatch(IDLE.RUN_ID, "b" * 40, "a" * 40, "a" * 40)
        for run_id, engine, control, github in (
            ("36956986577", IDLE.BASELINE_ENGINE_SHA, "a" * 40, "a" * 40),
            ("37122359802", IDLE.BASELINE_ENGINE_SHA, "a" * 40, "a" * 40),
            ("other", IDLE.BASELINE_ENGINE_SHA, "a" * 40, "a" * 40),
            (IDLE.RUN_ID, "bad", "a" * 40, "a" * 40),
            (IDLE.RUN_ID, IDLE.BASELINE_ENGINE_SHA, "a" * 40, "b" * 40),
        ):
            with self.assertRaises(RuntimeError):
                IDLE.check_dispatch(run_id, engine, control, github)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"reviewed":true}', encoding="utf-8")
            with patch.object(IDLE, "BASELINE_DIGEST", IDLE.artifact_digest(Path(directory))):
                IDLE.verify_artifact(Path(directory))
                path.write_text('{"reviewed":false}', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                    IDLE.verify_artifact(Path(directory))

    def test_all_consumers_bind_one_exact_new_baseline(self):
        import yue2_gpu0_owner_guard as owner
        self.assertEqual(IDLE.BASELINE_ENGINE_SHA, "4127a675fc8575555e029e01b7f6867488880a8f")
        self.assertEqual(IDLE.BASELINE_CONTROL_SHA, "e538120368ac279cc42176c44f9d88fa5af9c9b4")
        self.assertEqual(owner.RECEIPT, IDLE.RUN_ID)
        self.assertEqual(owner.RECEIPT_DIGEST, IDLE.BASELINE_DIGEST)
        expected = (f"yue2-cuda-diagnostic-engine-{IDLE.BASELINE_ENGINE_SHA}"
                    f"-control-{IDLE.BASELINE_CONTROL_SHA}-{IDLE.RUN_ID}-1")
        for name in ("yue2-precision-proof.yml", "yue2-app-precision-profile.yml",
                     "yue2-bf16-tile-diagnostic.yml"):
            source = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertEqual(source.count(f"name: {expected}\n"), 1, name)
            self.assertNotIn("-37122359802-1", source)
            if name != "yue2-bf16-tile-diagnostic.yml":
                self.assertIn(f"inputs.idle_cuda_context_run_id == '{IDLE.RUN_ID}'", source)
                self.assertNotIn("inputs.idle_cuda_context_run_id == '37122359802'", source)

    def test_saved_runner_is_pinned_and_fresh_runner_matches_an_eligible_listener(self):
        self.assertEqual(IDLE.BASELINE_RUNNER, "cuda-windows")
        source = {"completed": True, "targetPid": 38212, "engineSha": IDLE.BASELINE_ENGINE_SHA,
                  "controlSha": IDLE.BASELINE_CONTROL_SHA}
        with patch.object(IDLE, "read_json", return_value={**source, "runner": "cuda-windows-2"}), \
             self.assertRaisesRegex(RuntimeError, "manifest/source/runner mismatch"):
            IDLE.summarize(Path("unused"), baseline=True, pid=38212,
                           engine_sha=IDLE.BASELINE_ENGINE_SHA, control_sha=IDLE.BASELINE_CONTROL_SHA)
        # Reaching the catalog proves the fresh manifest passed runner provenance;
        # full fresh evidence still requires the separate 29-file device checks.
        for runner in ("cuda-windows", "cuda-windows-2"):
            with self.subTest(runner=runner), patch.dict("os.environ", {"RUNNER_NAME": runner}), \
                 patch.object(IDLE, "read_json", side_effect=[{**source, "runner": runner},
                                                            RuntimeError("catalog reached")]), \
                 self.assertRaisesRegex(RuntimeError, "catalog reached"):
                IDLE.summarize(Path("unused"), baseline=False, pid=38212,
                               engine_sha=IDLE.BASELINE_ENGINE_SHA, control_sha=IDLE.BASELINE_CONTROL_SHA)
        with patch.dict("os.environ", {"RUNNER_NAME": "cuda-windows"}), \
             patch.object(IDLE, "read_json", return_value={**source, "runner": "cuda-windows-2"}), \
             self.assertRaisesRegex(RuntimeError, "manifest/source/runner mismatch"):
            IDLE.summarize(Path("unused"), baseline=False, pid=38212,
                           engine_sha=IDLE.BASELINE_ENGINE_SHA, control_sha=IDLE.BASELINE_CONTROL_SHA)
        with patch.dict("os.environ", {"RUNNER_NAME": "unknown-listener"}), \
             patch.object(IDLE, "read_json", return_value={**source, "runner": "unknown-listener"}), \
             self.assertRaisesRegex(RuntimeError, "manifest/source/runner mismatch"):
            IDLE.summarize(Path("unused"), baseline=False, pid=38212,
                           engine_sha=IDLE.BASELINE_ENGINE_SHA, control_sha=IDLE.BASELINE_CONTROL_SHA)

    def test_shared_census_requires_live_receipt_only_for_mixed_context(self):
        def result(output):
            return type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()
        header = "# gpu pid type fb sm\n"
        with patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": IDLE.RUN_ID}):
            with patch.object(CONTROL.subprocess, "run", return_value=result(header + "0 - - - -\n")), \
                 patch.object(IDLE, "census_mixed_context") as attestation, \
                 patch.object(IDLE, "census_empty_device", return_value=("fresh empty proof", True)) as empty:
                self.assertEqual(CONTROL.cuda_census(), ("fresh empty proof", []))
                attestation.assert_not_called()
                empty.assert_called_once()
            with patch.object(CONTROL.subprocess, "run", return_value=result(header + "0 38212 C+G 0 -\n")), \
                 patch.object(IDLE, "census_mixed_context", return_value=("reviewed raw receipt", True)):
                self.assertEqual(CONTROL.cuda_census(), ("reviewed raw receipt", []))
            reordered = "# pid gpu type fb sm\n38212 0 C+G 0 -\n"
            with patch.object(CONTROL.subprocess, "run", return_value=result(reordered)), \
                 patch.object(IDLE, "census_mixed_context", return_value=("reviewed reordered", True)) as attestation:
                self.assertEqual(CONTROL.cuda_census(), ("reviewed reordered", []))
                self.assertEqual(attestation.call_args.args[0], 38212)
            with patch.object(CONTROL.subprocess, "run", return_value=result(header + "0 38212 C+G 0 -\n")), \
                 patch.object(IDLE, "census_mixed_context", side_effect=RuntimeError("expired")):
                raw, busy = CONTROL.cuda_census()
                self.assertIn("expired", raw)
                self.assertEqual(len(busy), 1)
            with patch.object(CONTROL.subprocess, "run", return_value=result(header + "0 38212 C+G 0 -\n")), \
                 patch.object(IDLE, "census_mixed_context", return_value=("raw active counters", False)):
                self.assertEqual(CONTROL.cuda_census(), ("raw active counters", ["0 38212 C+G 0 -"]))
            with patch.object(CONTROL.subprocess, "run", return_value=result(header + "0 38212 C 0 0\n")), \
                 patch.object(IDLE, "census_mixed_context") as attestation:
                self.assertEqual(len(CONTROL.cuda_census()[1]), 1)
                attestation.assert_not_called()

    def test_saved_physical_refusal_is_visible_without_exempting_pid(self):
        busy = ["0 38212 C+G - - ChatGPT.exe"]
        for reason in ("adapterDedicated rose over reviewed baseline",
                       "target GPU Engine activity", "process identity changed"):
            raw = json.dumps({"commandExit": 0, "refusal": reason})
            with patch.object(CONTROL, "cuda_census", return_value=(raw, busy)):
                saved, refused = CONTROL.cuda_physical_census()
            self.assertEqual(refused, busy)
            message = CONTROL.physical_busy_message(saved, refused, "before test")
            self.assertIn(reason, message)
            self.assertIn("38212", message)
        for raw in ("typed pmon rows", "[]", '{"refusal": true}', '{"refusal": ""}'):
            self.assertEqual(CONTROL.physical_busy_message(raw, busy, "before test"),
                             f"before test: {busy}")
        for pid in (38213, 123):
            foreign = [f"0 {pid} C+G - - ChatGPT.exe"]
            with patch.object(CONTROL, "cuda_census", return_value=("typed rows", foreign)):
                self.assertEqual(CONTROL.cuda_physical_census()[1], foreign)

    def test_full_cuda_proof_refuses_bare_pmon_and_requires_all_fresh_files(self):
        for raw, busy in (("# gpu pid type\n0 - -\n", []),
                          ("typed busy", ["0 123 C 0 0"])):
            with patch.object(CONTROL, "cuda_census", return_value=(raw, busy)):
                _, refused = CONTROL.cuda_physical_census()
                self.assertTrue(refused)
        good = {"commandExit": 0,
                "diagnosticFiles": {f"sample-{i}.json": "{}" for i in range(29)},
                "diagnosticFileBytesB64": {f"sample-{i}.json": base64.b64encode(b"{}").decode()
                                           for i in range(29)}}
        with patch.object(CONTROL, "cuda_census", return_value=(json.dumps(good), [])):
            self.assertEqual(CONTROL.cuda_physical_census()[1], [])
        for mutate in (lambda item: item["diagnosticFiles"].pop("sample-0.json"),
                       lambda item: item.__setitem__("refusal", "active")):
            changed = copy.deepcopy(good)
            mutate(changed)
            with patch.object(CONTROL, "cuda_census", return_value=(json.dumps(changed), [])):
                self.assertTrue(CONTROL.cuda_physical_census()[1])
        with tempfile.TemporaryDirectory() as directory:
            inventory = CONTROL.retain_cuda_physical_evidence(Path(directory), "before", json.dumps(good))
            self.assertEqual(len(inventory), 29)
            self.assertEqual(len(list((Path(directory) / "physical-before").iterdir())), 29)
        changed = copy.deepcopy(good)
        changed["diagnosticFileBytesB64"]["sample-0.json"] = base64.b64encode(b"wrong").decode()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "raw bytes disagree"):
            CONTROL.retain_cuda_physical_evidence(Path(directory), "before", json.dumps(changed))

    def test_fresh_physical_files_preserve_windows_bom_and_crlf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            original = {}
            for index in range(29):
                name = f"sample-{index:02d}.json"
                data = (b"\xef\xbb\xbf{\r\n  \"epoch\": 1\r\n}\r\n" if index % 2 == 0
                        else b"{\n  \"epoch\": 1\n}\n")
                (source / name).write_bytes(data)
                original[name] = data
            files, encoded = IDLE.diagnostic_file_pairs(source)
            self.assertEqual(files["sample-00.json"], '{\r\n  "epoch": 1\r\n}\r\n')
            self.assertEqual(files["sample-01.json"], '{\n  "epoch": 1\n}\n')
            self.assertEqual({name: base64.b64decode(value) for name, value in encoded.items()}, original)
            probe = {"diagnosticFiles": files, "diagnosticFileBytesB64": encoded}
            inventory = CONTROL.retain_cuda_physical_evidence(root, "before", json.dumps(probe))
            self.assertEqual(len(inventory), 29)
            for name, data in original.items():
                self.assertEqual((root / "physical-before" / name).read_bytes(), data)

            normalized = copy.deepcopy(probe)
            normalized["diagnosticFiles"]["sample-00.json"] = files["sample-00.json"].replace("\r\n", "\n")
            with self.assertRaisesRegex(RuntimeError, "raw bytes disagree"):
                CONTROL.retain_cuda_physical_evidence(root, "normalized", json.dumps(normalized))
            changed = copy.deepcopy(probe)
            changed["diagnosticFileBytesB64"]["sample-00.json"] = base64.b64encode(
                original["sample-00.json"].replace(b"1", b"2", 1)).decode("ascii")
            with self.assertRaisesRegex(RuntimeError, "raw bytes disagree"):
                CONTROL.retain_cuda_physical_evidence(root, "mutated", json.dumps(changed))

    def test_reviewed_baseline_copy_remains_byte_bound_to_original_28_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for index in range(28):
                (source / f"sample-{index:02d}.json").write_bytes(b"{\"reviewed\":true}\n")
            pinned = IDLE.artifact_digest(source)
            (root / "evidence").mkdir()
            with patch.object(IDLE, "BASELINE_DIGEST", pinned):
                copied = CONTROL.retain_reviewed_baseline(root / "evidence", source)
                self.assertEqual(len(copied), 28)
                self.assertEqual(IDLE.artifact_digest(root / "evidence" / "reviewed-idle-context"), pinned)
            (source / "sample-00.json").write_bytes(b"{\"reviewed\":false}\n")
            (root / "other").mkdir()
            with patch.object(IDLE, "BASELINE_DIGEST", pinned), \
                 self.assertRaisesRegex(RuntimeError, "differs from pinned"):
                CONTROL.retain_reviewed_baseline(root / "other", source)

    def test_runtime_engine_sha_is_not_relabelled_as_baseline_source(self):
        completion = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="microseconds").replace("+00:00", "0Z")
        baseline = {"identity": [38212, "owner", "C:/owner.exe", "birth"],
                    "completedUtc": completion}
        result = type("Result", (), {"returncode": 0, "stderr": ""})()
        with patch.dict("os.environ", {"EXPECTED_ENGINE_SHA": "b" * 40,
                                    "GITHUB_SHA": "c" * 40}), \
             patch.object(IDLE, "reviewed_baseline", return_value=(baseline, Path("receipt"))), \
             patch.object(IDLE, "_pmon_output"), \
             patch.object(IDLE.subprocess, "run", return_value=result) as command, \
             patch.object(IDLE, "summarize", return_value=baseline) as summarize, \
             patch.object(IDLE, "validate_current"):
            _, verified = IDLE.census_mixed_context(38212, "typed pmon")
            self.assertTrue(verified)
            self.assertIn("b" * 40, command.call_args.args[0])
            self.assertEqual(summarize.call_args.kwargs["engine_sha"], "b" * 40)
            self.assertEqual(IDLE.BASELINE_ENGINE_SHA,
                             "4127a675fc8575555e029e01b7f6867488880a8f")
        with patch.object(IDLE, "reviewed_baseline", return_value=(baseline, Path("receipt"))):
            IDLE.require_remaining_window(60)
            with self.assertRaisesRegex(RuntimeError, "cannot cover"):
                IDLE.require_remaining_window(12 * 3600)

    def test_only_owned_precision_child_is_killed_on_timeout(self):
        class Owned:
            pid = 8123
            def __init__(self):
                self.kills = 0
                self.calls = []
            def wait(self, timeout=None):
                self.calls.append(timeout)
                if len(self.calls) == 1:
                    raise subprocess.TimeoutExpired("precision", timeout)
                return -9
            def kill(self):
                self.kills += 1
            def poll(self):
                return -9 if self.kills else None
        child = Owned()
        self.assertEqual(CONTROL.wait_owned_child(child, "cuda"), (-9, True, None))
        self.assertEqual(child.kills, 1)
        self.assertEqual(child.calls, [CONTROL.CUDA_CHILD_TIMEOUT_SECONDS, 30])
        self.assertEqual(CONTROL.CUDA_CHILD_TIMEOUT_SECONDS, 180 * 60)
        self.assertEqual(CONTROL.CUDA_POSTFLIGHT_SECONDS, 600)
        other = Owned()
        def broken_wait(timeout=None):
            other.calls.append(timeout)
            if len(other.calls) == 1:
                raise OSError("owned wait failed")
            return -9
        other.wait = broken_wait
        code, timed_out, wait_error = CONTROL.wait_owned_child(other, "cuda")
        self.assertEqual((code, timed_out), (-9, False))
        self.assertIn("owned wait failed", wait_error)
        self.assertEqual(other.kills, 1)

    def test_timed_out_cuda_child_still_writes_release_and_postflight_proof(self):
        class Owned:
            pid = 8123
            killed = False
            def wait(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired("precision", timeout)
                return -9
            def kill(self):
                self.killed = True
            def poll(self):
                return -9 if self.killed else None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, reference, binary = root / "evidence", root / "reference", root / "binary"
            reference.mkdir()
            (reference / "vae_real_reference.safetensors").write_bytes(b"fixture")
            binary.write_bytes(b"binary")
            args = type("Args", (), {"evidence": evidence, "reference": reference,
                                     "binary": binary, "work_dir": root / "listening",
                                     "quant_smoke_binary": binary, "vae_smoke_binary": binary,
                                     "engine_sha": "a" * 40, "control_sha": "b" * 40,
                                     "app_sha": "", "backend": "cuda"})()
            baseline = {"completedUtc": "2026-10-03T00:00:00.0000000Z"}
            census = '{"diagnosticFiles":{},"diagnosticFileBytesB64":{}}'
            child = Owned()
            with patch.dict("os.environ", {"RUNNER_NAME": "cuda-windows-2",
                                        "CUDA_VISIBLE_DEVICES": "0",
                                        "YUE2_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns())}), \
                 patch.object(CONTROL, "sha256", return_value=CONTROL.REFERENCE_SHA256), \
                 patch.object(CONTROL, "verify_revisions"), \
                 patch.object(CONTROL.subprocess, "run", return_value=type("Result", (), {"stdout": ""})()), \
                 patch.object(IDLE, "require_remaining_window", return_value=(baseline, root)), \
                 patch.object(IDLE, "check_empty_dispatch"), \
                 patch.object(CONTROL, "retain_reviewed_baseline", return_value=[]), \
                 patch.object(CONTROL, "cuda_physical_census", return_value=(census, [])) as physical, \
                 patch.object(CONTROL, "retain_cuda_physical_evidence", return_value=[]), \
                 patch.object(CONTROL, "verify_binary_identity", return_value={}), \
                 patch.object(CONTROL.subprocess, "Popen", return_value=child), \
                 patch("builtins.print"), \
                 patch.object(CONTROL, "sample_cuda", return_value={"raw": "0,0,19,1000", "started_utc_ns": 1, "ended_utc_ns": 2}):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    CONTROL.execute(args)
            result = json.loads((evidence / "control.json").read_text(encoding="utf-8"))
            self.assertTrue(child.killed)
            self.assertTrue(result["owned_test_timed_out"])
            self.assertTrue(result["owned_test_released"])
            self.assertIsNone(result["post_census_error"])
            self.assertEqual(physical.call_count, 4)
            self.assertTrue((evidence / "external-samples.json").is_file())
            self.assertTrue((evidence / "census-after.txt").is_file())

    def _simulate_cuda_smokes(self, zero_quant=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, reference, binary = root / "evidence", root / "reference", root / "binary"
            reference.mkdir()
            (reference / "vae_real_reference.safetensors").write_bytes(b"fixture")
            binary.write_bytes(b"binary")
            args = type("Args", (), {"evidence": evidence, "reference": reference,
                                     "binary": binary, "quant_smoke_binary": binary,
                                     "vae_smoke_binary": binary, "work_dir": root / "listening",
                                     "engine_sha": "a" * 40, "control_sha": "b" * 40,
                                     "app_sha": "", "backend": "cuda"})()
            baseline = {"completedUtc": "2026-10-03T00:00:00.0000000Z"}
            census = '{"diagnosticFiles":{},"diagnosticFileBytesB64":{}}'
            launched = []
            events = []
            def identity(path, label, out):
                events.append(f"identity-{label}")
                return {"binary_sha256": "verified"}
            def physical_census():
                events.append("physical-census")
                return census, []
            def fake_child(path, name, label, backend, env, out, total_deadline, guard, identity):
                events.append(f"launch-{label}")
                launched.append((label, name, total_deadline))
                (out / ("test.log" if label == "precision" else f"{label}-smoke.log")).write_text(
                    f"running 1 test\ntest {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n",
                    encoding="utf-8")
                if label == "precision":
                    (out / "precision-receipt.json").write_text("{}", encoding="utf-8")
                    args.work_dir.mkdir()
                    (args.work_dir / "audio.wav").write_bytes(b"wav")
                row = {"label": label, "name": name, "pid": 8000 + len(launched),
                       "exit_code": 0, "timed_out": False, "wait_error": None,
                       "released": True, "exact_one_test_passed": True,
                       "binary_unchanged_after_child": True,
                       "sample_count": 0 if zero_quant and label == "quant" else 1,
                       "sampler_faults": [], "scheduling": {"mode": "shared-host"}}
                sample = {"raw": "0,0,19,1000", "started_utc_ns": 1, "ended_utc_ns": 2}
                return row, ([] if zero_quant and label == "quant" else [sample]), []
            with patch.dict("os.environ", {"RUNNER_NAME": "cuda-windows-2", "CUDA_VISIBLE_DEVICES": "0",
                                        "YUE2_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns())}), \
                 patch.object(CONTROL, "sha256", return_value=CONTROL.REFERENCE_SHA256), \
                 patch.object(CONTROL, "verify_revisions"), \
                 patch.object(CONTROL.subprocess, "run", return_value=type("Result", (), {"stdout": ""})()), \
                 patch.object(IDLE, "require_remaining_window", return_value=(baseline, root)), \
                 patch.object(IDLE, "check_empty_dispatch"), \
                 patch.object(CONTROL, "retain_reviewed_baseline", return_value=[]), \
                 patch.object(CONTROL, "cuda_physical_census", side_effect=physical_census) as physical, \
                 patch.object(CONTROL, "retain_cuda_physical_evidence", return_value=[]), \
                 patch.object(CONTROL, "verify_binary_identity", side_effect=identity), \
                 patch.object(CONTROL, "run_test_child", side_effect=fake_child), \
                 patch.object(CONTROL, "validate_receipt"), \
                 patch.object(CONTROL, "missing_stage_markers", return_value=[]), \
                 patch.object(CONTROL, "stage_markers", return_value=[{"stage": "test", "event": "start", "unixMs": 1}]), \
                 patch("builtins.print"):
                if zero_quant:
                    with self.assertRaisesRegex(RuntimeError, "owned exact-test sequence failed"):
                        CONTROL.execute(args)
                else:
                    CONTROL.execute(args)
            self.assertEqual([row[0] for row in launched], ["quant"] if zero_quant else ["quant", "vae", "precision"])
            if not zero_quant:
                self.assertEqual(launched[0][2], launched[1][2])
                self.assertEqual(launched[1][2], launched[2][2])
            report = json.loads((evidence / "control.json").read_text(encoding="utf-8"))
            self.assertEqual([row["label"] for row in report["owned_children"]],
                             ["quant"] if zero_quant else ["quant", "vae", "precision"])
            self.assertEqual(physical.call_count, 4 if zero_quant else 8)
            for label in [row[0] for row in launched]:
                start = events.index(f"identity-{label}")
                launch = events.index(f"launch-{label}")
                self.assertEqual(events[launch - 1], "physical-census")
                self.assertLess(start, launch - 1)

    def test_cuda_smokes_complete_in_order_before_real_weight_child(self):
        self._simulate_cuda_smokes()

    def test_zero_sample_smoke_refuses_before_next_child(self):
        good = {"exit_code": 0, "timed_out": False, "wait_error": None,
                "released": True, "exact_one_test_passed": True,
                "binary_unchanged_after_child": True, "sample_count": 1,
                "sampler_faults": [], "post_census_error": None, "post_census_busy": []}
        self.assertTrue(CONTROL.child_stage_succeeded(good))
        self.assertFalse(CONTROL.child_stage_succeeded(dict(good, sample_count=0)))
        self.assertFalse(CONTROL.child_stage_succeeded(dict(good, binary_unchanged_after_child=False)))
        self._simulate_cuda_smokes(zero_quant=True)

    def test_counter_status_and_missing_fields_refuse_instead_of_becoming_zero(self):
        luid = "luid_0x00000000_0x00020d46"
        pid = 38212
        def row(path, instance, value=0, status="0"):
            return {"counter": path, "samples": [{"instance": instance, "cookedValue": value, "status": status}]}
        rows = [
            row(r"\GPU Engine(*)\Utilization Percentage", f"pid_{pid}_{luid}_phys_0_eng_0_engtype_3d"),
            row(r"\GPU Process Memory(*)\Dedicated Usage", f"pid_{pid}_{luid}_phys_0", 19_341_312),
            row(r"\GPU Process Memory(*)\Shared Usage", f"pid_{pid}_{luid}_phys_0", 10_391_552),
            row(r"\GPU Process Memory(*)\Total Committed", f"pid_{pid}_{luid}_phys_0", 29_732_864),
            row(r"\GPU Adapter Memory(*)\Dedicated Usage", f"{luid}_phys_0", 23_834_624),
            row(r"\GPU Adapter Memory(*)\Shared Usage", f"{luid}_phys_0", 10_391_552),
            row(r"\GPU Adapter Memory(*)\Total Committed", f"{luid}_phys_0", 34_226_176),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows-counters-0.json"
            def check(value):
                path.write_text(json.dumps({"targetPid": pid, "counters": value}), encoding="utf-8")
                return IDLE._counters(Path(directory), 0, pid, luid, baseline=False)
            self.assertEqual(check(rows)["processDedicated"], 19_341_312)
            for index in range(len(rows)):
                for invalid in (float("inf"), float("-inf"), float("nan"), True):
                    changed = copy.deepcopy(rows)
                    changed[index]["samples"][0]["cookedValue"] = invalid
                    with self.subTest(counter=index, invalid=invalid), self.assertRaises(RuntimeError):
                        check(changed)
            for mutation in (
                lambda x: x[0]["samples"][0].__setitem__("status", "1"),
                lambda x: x[0]["samples"][0].__setitem__("cookedValue", 0.1),
                lambda x: x[1].__setitem__("error", "unsupported"),
                lambda x: x.pop(),
            ):
                changed = copy.deepcopy(rows)
                mutation(changed)
                with self.assertRaises(RuntimeError):
                    check(changed)

    def test_fresh_probe_refuses_new_or_disappearing_mixed_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pmon = root / "pmon-0-0.json"
            apps = root / "compute-apps-0-0.json"
            def check(rows, app_rows):
                pmon.write_text(json.dumps({"exitCode": 0, "output": ["# gpu pid type fb sm mem enc dec jpg ofa", *rows]}), encoding="utf-8")
                apps.write_text(json.dumps({"exitCode": 0, "output": app_rows}), encoding="utf-8")
                IDLE._pmon(root, "pmon-0-0", 38212)
                IDLE._compute_apps(root, "compute-apps-0-0", 38212)
            check(["0 38212 C+G 0 - - - - - -"], ["38212, C:\\Redacted\\desktop.exe, [N/A]"])
            for rows, app_rows in (
                ([], ["38212, C:\\Redacted\\desktop.exe, [N/A]"]),
                (["0 38212 C+G 0 - - - - - -", "0 999 C+G 0 - - - - - -"], ["38212, C:\\Redacted\\desktop.exe, [N/A]"]),
                (["0 38212 C+G 0 - - - - - -", "0 999 C 0 2 - - - - -"], ["38212, C:\\Redacted\\desktop.exe, [N/A]"]),
                (["0 38212 C+G 0 - - - - - -"], ["38212, C:\\Redacted\\desktop.exe, [N/A]", "999, C:\\Other.exe, [N/A]"]),
            ):
                with self.subTest(rows=rows, app_rows=app_rows), self.assertRaises(RuntimeError):
                    check(rows, app_rows)

    def test_pmon_positive_utilization_refuses_even_with_idle_windows_samples(self):
        header = "# gpu pid type sm mem enc dec jpg ofa fb ccpm command"
        values = ["0", "38212", "C+G", "-", "-", "-", "-", "-", "-", "19", "0", "desktop.exe"]
        IDLE._pmon_output([header, " ".join(values)], "initial pmon", 38212)
        for missing in range(3, 9):
            columns = header.split()[1:]
            fields = values.copy()
            columns.pop(missing)
            fields.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                IDLE._pmon_output(["# " + " ".join(columns), " ".join(fields)], "initial pmon", 38212)
        for index in range(3, 9):
            for invalid in ("50", "0.1", "NaN", "Infinity", "bad"):
                changed = values.copy()
                changed[index] = invalid
                with self.subTest(metric=index, value=invalid), self.assertRaises(RuntimeError):
                    IDLE._pmon_output([header, " ".join(changed)], "initial pmon", 38212)

    def test_metal_census_refuses_foreign_workers_by_executable_only(self):
        rows = "\n".join((
            "101 /tmp/precision_real_weights-deadbeef",
            "102 /tmp/sequential_residency_real_weights-deadbeef",
            "103 /tmp/mlx-gen-qwen-image",
            "104 /tmp/memory-mlx-adapter",
            "105 /tmp/sceneworks-worker",
            "106 /opt/actions-runner/bin/Runner.Worker",
            "107 /bin/zsh",
            "108 /Applications/Safari.app/Contents/MacOS/Safari",
            "109 /tmp/mlx-gen",
            "110 /tmp/candle-gen",
        )) + "\n"
        result = type("Result", (), {"returncode": 0, "stdout": rows, "stderr": ""})()
        with patch.object(CONTROL.subprocess, "run", return_value=result) as run:
            raw, busy = CONTROL.metal_census()
        self.assertEqual(raw, rows)
        self.assertEqual([int(line.split()[0]) for line in busy], [101, 102, 103, 104, 105, 109, 110])
        self.assertEqual(run.call_args.args[0], ["/bin/ps", "-axo", "pid=,comm="])

    def test_one_exact_ignored_test_must_execute(self):
        good = "running 1 test\ntest explicit_stage_precision_real_weights ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out\n"
        self.assertTrue(CONTROL.one_test_executed(good))
        for bad in (good.replace("1 passed", "0 passed"), good.replace("explicit_stage_precision_real_weights", "wrong_test"), good.replace("0 ignored", "1 ignored")):
            self.assertFalse(CONTROL.one_test_executed(bad))
        for _, name in CONTROL.CUDA_SMOKES:
            smoke = f"running 1 test\ntest {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 9 filtered out\n"
            self.assertTrue(CONTROL.exact_one_test_executed(smoke, name))
            self.assertFalse(CONTROL.exact_one_test_executed(smoke.replace("1 passed", "0 passed"), name))
            self.assertFalse(CONTROL.exact_one_test_executed(smoke.replace(name, "wrong::test"), name))
            self.assertFalse(CONTROL.exact_one_test_executed(smoke + smoke, name))
        captured_shape = ("running 1 test\n"
                          f"test {CONTROL.TEST_NAME} ... YUE2_PRECISION_LISTENING_DIR E:\\audio\n"
                          'YUE2_PRECISION_STAGE {"stage":"Bf16:registered_load","event":"start","unixMs":1}\n'
                          "ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out\n")
        self.assertTrue(CONTROL.one_test_executed(captured_shape))
        self.assertFalse(CONTROL.one_test_executed(captured_shape.replace("\nok\n", "\nnot ok\n")))
        self.assertFalse(CONTROL.one_test_executed(captured_shape.replace("running 1 test", "running 0 tests")))
        self.assertFalse(CONTROL.one_test_executed(captured_shape.replace("\nok\n", "\ntest wrong::extra ... ok\n")))

    def test_exact_cargo_binary_identity_and_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, build, output = root / "test.exe", root / "build.jsonl", root / "quant-binary.txt"
            binary.write_bytes(b"owned test binary")
            manifest = (Path.cwd() / CONTROL.BUILD_TARGETS["quant"][3]).resolve()
            package_id = "path+file:///engine/crates/kernels/candle-quant-kernels#candle-quant-kernels@0.0.0"
            metadata = {"packages": [{"name": "candle-quant-kernels", "id": package_id,
                                      "manifest_path": str(manifest)}]}
            row = {"reason": "compiler-artifact", "package_id": package_id,
                   "target": {"name": "candle_quant_kernels", "kind": ["lib"],
                              "src_path": str((Path.cwd() / CONTROL.BUILD_TARGETS["quant"][4]).resolve())},
                   "profile": {"test": True}, "executable": str(binary)}
            def resolve(rows):
                build.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
                args = type("Args", (), {"target": "quant", "build_json": build, "output": output})()
                result = type("Result", (), {"stdout": json.dumps(metadata)})()
                with patch.object(CONTROL.subprocess, "run", return_value=result):
                    CONTROL.resolve_binary(args)
            resolve([row])
            self.assertEqual(output.read_text(encoding="utf-8").strip(), str(binary.resolve()))
            self.assertEqual(CONTROL.verify_binary_identity(binary, "quant", root)["package_id"], package_id)
            for bad in (dict(row, package_id="wrong"),
                        dict(row, target={"name": "candle_quant_kernels", "kind": ["test"]}),
                        dict(row, target={"name": "candle_quant_kernels", "kind": ["lib"],
                                          "src_path": str(Path(directory) / "wrong.rs")}),
                        dict(row, profile={"test": False})):
                with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "exactly one"):
                    resolve([bad])
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                resolve([row, row])
            resolve([row])
            binary.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                CONTROL.verify_binary_identity(binary, "quant", root)

    def test_three_children_share_one_cuda_deadline_and_tail(self):
        job_start = time.time_ns()
        with patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": IDLE.RUN_ID}), \
             patch.object(IDLE, "require_remaining_window", return_value=({}, Path("baseline"))) as check, \
             patch.object(CONTROL.time, "monotonic", side_effect=[100.0, 110.0, 179.0]):
            self.assertEqual(CONTROL.remaining_cuda_budget(200.0, job_start), 100.0)
            self.assertEqual(CONTROL.remaining_cuda_budget(200.0, job_start), 90.0)
            self.assertEqual(CONTROL.remaining_cuda_budget(200.0, job_start), 21.0)
            self.assertEqual([call.args[0] for call in check.call_args_list], [700.0, 690.0, 621.0])
        with patch.object(CONTROL.time, "monotonic", return_value=201.0), \
             self.assertRaisesRegex(RuntimeError, "combined CUDA child deadline"):
            CONTROL.remaining_cuda_budget(200.0, job_start)

    def test_slow_prelaunch_checks_cannot_renew_absolute_child_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "test.exe"
            binary.write_bytes(b"binary")
            job_start = time.time_ns()
            with patch.object(IDLE, "require_remaining_window", return_value=({}, root)), \
                 patch.object(CONTROL.time, "monotonic", return_value=100.0):
                self.assertEqual(CONTROL.remaining_cuda_budget(200.0, job_start), 100.0)
            # Identity, owner preflight, and the 29-file census can consume the
            # provisional budget. The child must recheck the same deadline at
            # the final Popen boundary, without giving those seconds back.
            with patch.object(CONTROL.time, "monotonic", return_value=201.0), \
                 patch.object(CONTROL.subprocess, "Popen") as launch, \
                 self.assertRaisesRegex(RuntimeError, "expired before Popen"):
                CONTROL.run_test_child(binary, "exact::test", "quant", "cuda", {}, root,
                                       200.0, None, {"binary_sha256": CONTROL.sha256(binary)})
            launch.assert_not_called()

    def test_stage_markers_remain_machine_readable(self):
        line = 'test explicit_stage_precision_real_weights ... YUE2_PRECISION_STAGE {"stage":"Bf16:standard:encoder","event":"start","unixMs":100}'
        self.assertEqual(CONTROL.stage_markers(line)[0]["unixMs"], 100)
        with self.assertRaises(json.JSONDecodeError):
            CONTROL.stage_markers(line.replace('"unixMs":100', '"unixMs":'))
        markers = [{'stage':'Bf16:standard:encoder','event':'start','unixMs':100},
                   {'stage':'Bf16:standard:encoder','event':'end','unixMs':200}]
        samples = [{'started_utc_ns':110_000_000,'ended_utc_ns':120_000_000},
                   {'started_utc_ns':195_000_000,'ended_utc_ns':205_000_000}]
        self.assertEqual(CONTROL.stage_sample_coverage(markers,samples)['Bf16:standard:encoder'],
                         {'fully_contained_samples':1,'overlapping_samples':2})
        self.assertIn('Bf16:cross_policy_cached_decode:end', CONTROL.missing_stage_markers(markers))

    def test_reference_mutation_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vae_real_reference.safetensors").write_bytes(b"wrong reference")
            (root / "NONCOMMERCIAL.txt").write_bytes(b"license")
            metadata = {"engine_sha": "a" * 40, "control_sha": "c" * 40,
                        "sha256": CONTROL.REFERENCE_SHA256, "runner": "hosted-cpu-transfer",
                        "transfer_run_id": "123", "transfer_run_attempt": "1",
                        "source_run_id": TRANSFER.SOURCE_RUN_ID, "source_run_attempt": 1,
                        "source_engine_sha": TRANSFER.SOURCE_ENGINE_SHA,
                        "source_artifact_id": TRANSFER.SOURCE_ARTIFACT_ID,
                        "source_artifact_zip_sha256": TRANSFER.SOURCE_ZIP_SHA256,
                        "source_provenance_sha256": TRANSFER.SOURCE_METADATA_SHA256,
                        "noncommercial_sha256": TRANSFER.LICENSE_SHA256}
            (root / "reference-provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = type("Args", (), {"directory": root, "engine_sha": "a" * 40,
                                     "control_sha": "c" * 40, "run_id": "123"})()
            actual_sha = CONTROL.sha256
            with patch.object(CONTROL, "sha256", side_effect=lambda path:
                              TRANSFER.LICENSE_SHA256 if path.name == "NONCOMMERCIAL.txt" else actual_sha(path)):
                with self.assertRaisesRegex(RuntimeError, "digest differs"):
                    CONTROL.verify_reference(args)
            args.engine_sha = "b" * 40
            with self.assertRaisesRegex(RuntimeError, "engine SHA differs"):
                CONTROL.verify_reference(args)
            args.engine_sha = "a" * 40
            args.run_id = "124"
            with self.assertRaisesRegex(RuntimeError, "transfer provenance"):
                CONTROL.verify_reference(args)
            args.run_id = "123"
            args.control_sha = "d" * 40
            with self.assertRaisesRegex(RuntimeError, "transfer provenance"):
                CONTROL.verify_reference(args)
            args.control_sha = "c" * 40
            metadata.update({"reference_source_mode": "relay",
                             "relay_run_id": TRANSFER.RELAY_RUN_ID,
                             "relay_artifact_id": TRANSFER.RELAY_ARTIFACT_ID,
                             "relay_engine_sha": TRANSFER.RELAY_ENGINE_SHA,
                             "relay_control_sha": TRANSFER.RELAY_CONTROL_SHA,
                             "relay_artifact_zip_sha256": TRANSFER.RELAY_ZIP_SHA256,
                             "relay_provenance_sha256": TRANSFER.RELAY_METADATA_SHA256})
            (root / "reference-provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch.object(CONTROL, "sha256", side_effect=lambda path:
                              TRANSFER.LICENSE_SHA256 if path.name == "NONCOMMERCIAL.txt" else actual_sha(path)):
                with self.assertRaisesRegex(RuntimeError, "digest differs"):
                    CONTROL.verify_reference(args)
            for field, wrong in (("relay_run_id", 1),
                                 ("relay_artifact_zip_sha256", "0" * 64),
                                 ("relay_provenance_sha256", None)):
                changed = dict(metadata, **{field: wrong})
                (root / "reference-provenance.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "relay identity"):
                    CONTROL.verify_reference(args)

    def test_actual_rust_receipt_shape_and_cross_policy_mutations(self):
        def decoder(variant, dtype):
            return {"variant": variant, "weightsSha256": CONTROL.DECODER_SHA256[variant],
                    "parameterDtype": dtype, "activationDtype": dtype,
                    "decodeCases": [{"name": "long"}, {"name": "production"}],
                    "encoderMean": {"snrDb": 100}, "encoderScale": {"snrDb": 100}}
        cases = []
        for index, policy in enumerate(("Bf16", "Auto", "Fp32"), 1):
            vae = "bfloat16" if policy == "Bf16" else "float32"
            model = "float32" if policy == "Fp32" else "bfloat16"
            dtype = "BF16" if policy == "Bf16" else "F32"
            cases.append({"requestedPolicy": policy,
                          "effectiveDtypes": {"ar": model, "nar": model, "vaeDecoder": vae, "vaeEncoder": vae},
                          "generation": {"config": {"compute_policy": policy.lower(), "vae_dtype": vae},
                                         "runIdentity": str(index), "legacyCacheIdentity": str(index + 3)},
                          "decoders": [decoder("standard", dtype), decoder("legacy", dtype)]})
        receipt = {"schemaVersion": 1, "backend": "cuda", "referenceSha256": CONTROL.REFERENCE_SHA256,
                   "listeningDir": "/persistent/audio/run-1",
                   "cases": cases, "legacyToBf16CachedDecode": {
                       "sourceIdentity": "legacy", "targetIdentity": "new-waveform",
                       "sourceGeneration": {"identity": "legacy", "config": {"vae_dtype": "float32"}},
                       "targetConfig": {"compute_policy": "bf16", "vae_dtype": "bfloat16",
                                        "decoder_release": "legacy", "cached_decode": {"source_identity": "legacy"}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            def check(value):
                path.write_text(json.dumps(value), encoding="utf-8")
                CONTROL.validate_receipt(path, "cuda", Path("/persistent/audio/run-1"))
            check(receipt)
            for mutation in (
                lambda x: x["cases"][0]["effectiveDtypes"].__setitem__("vaeDecoder", "float32"),
                lambda x: x["cases"][0]["decoders"][1].__setitem__("weightsSha256", "bad"),
                lambda x: x["legacyToBf16CachedDecode"]["targetConfig"].__setitem__("vae_dtype", "float32"),
                lambda x: x.__setitem__("listeningDir", "/tmp/deleted-audio"),
            ):
                changed = json.loads(json.dumps(receipt))
                mutation(changed)
                with self.assertRaises(RuntimeError):
                    check(changed)

    def test_workflow_is_dispatch_only_and_selects_one_new_test(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("schedule:", source)
        self.assertIn("options: [fixture, cuda, metal, cuda-diagnostic]", source)
        self.assertIn("if: inputs.stage == 'fixture'", source)
        self.assertIn("if: inputs.stage == 'cuda'", source)
        self.assertIn("if: inputs.stage == 'metal'", source)
        self.assertEqual(self.concurrency_group(self.concurrency_settings(source)["group"],
                                               "cuda", "101"), "inference-real-weights-physical-host")
        self.assertIn('CUDA_VISIBLE_DEVICES: "0"', source)
        self.assertEqual(source.count("path: ${{ env.YUE2_PRECISION_WORK_DIR }}/**/*.wav"), 2)
        self.assertEqual(source.count("if: ${{ always() && env.YUE2_PRECISION_WORK_DIR != '' }}"), 2)
        self.assertNotIn("path: ${{ env.YUE2_PRECISION_WORK_DIR }}\n", source)
        self.assertIn("yue2-precision-listening-cuda-cc-by-nc-internal-", source)
        self.assertIn("yue2-precision-listening-metal-cc-by-nc-internal-", source)
        self.assertIn("test \"$RUNNER_NAME\" = nax-macos-2", source)
        self.assertIn("--test precision_real_weights", source)
        cuda = source.split("  cuda:\n", 1)[1].split("  metal:\n", 1)[0]
        for package, target in (("candle-quant-kernels", "quant"), ("candle-audio-yue2", "vae")):
            self.assertIn(f"-p {package} --features cuda --lib --no-run", cuda)
            self.assertIn(f"resolve-binary --target {target}", cuda)
            self.assertIn(f"--{target}-smoke-binary", cuda)
        self.assertLess(cuda.index("--target quant"), cuda.index("--target vae"))
        self.assertLess(cuda.index("--target vae"), cuda.index("$proofArgs = @('run'"))
        self.assertIn("expected_control_sha:", source)
        self.assertIn("ref: ${{ inputs.expected_engine_sha }}", source)
        self.assertLess(source.index("Select checked Git Bash before pinned Rust"),
                        source.index("uses: dtolnay/rust-toolchain@"))
        self.assertIn('if not exist "C:\\Program Files\\Git\\bin\\bash.exe" exit /b 1', source)
        self.assertIn('echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"', source)
        self.assertNotIn("tier_quality_against_the_f32_reference", source)
        self.assertNotIn("registered_loader_generates_a_song_with_every_artifact", source)
        self.assertNotIn("SIGKILL", source)
        app_source = (ROOT / ".github/workflows/yue2-app-precision-profile.yml").read_text(encoding="utf-8")
        tile_source = (ROOT / ".github/workflows/yue2-bf16-tile-diagnostic.yml").read_text(encoding="utf-8")
        for workflow in (source, app_source, tile_source):
            self.assertIn("idle_cuda_context_run_id:", workflow)
            self.assertIn("yue2-reviewed-idle-context", workflow)
            self.assertIn("if: inputs.idle_cuda_context_run_id != ''", workflow)
            self.assertIn(f"{IDLE.BASELINE_ENGINE_SHA}-control-{IDLE.BASELINE_CONTROL_SHA}-{IDLE.RUN_ID}-1", workflow)
            self.assertNotIn("-36956986577-1", workflow)
            self.assertIn("run-id: ${{ inputs.idle_cuda_context_run_id }}", workflow)
            self.assertIn('CUDA_VISIBLE_DEVICES: "0"\n      CUDA_DEVICE_ORDER: PCI_BUS_ID', workflow)
        self.assertEqual(source.count('CUDA_VISIBLE_DEVICES: "0"\n      CUDA_DEVICE_ORDER: PCI_BUS_ID'), 2)

    def test_cuda_deadline_stamp_uses_existing_directory_before_checkout(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        cuda = workflow.split("  cuda:\n", 1)[1].split("  metal:\n", 1)[0]
        self.assertIn("working-directory: engine", cuda)
        first_step = cuda.split("    steps:\n", 1)[1].split(
            "      - uses: actions/checkout@", 1)[0]
        self.assertIn("- name: Record CUDA job start for bounded owned-test deadline", first_step)
        self.assertIn("working-directory: ${{ github.workspace }}", first_step)
        self.assertIn("YUE2_PRECISION_JOB_STARTED_UTC_NS=$stamp", first_step)
        self.assertNotIn("working-directory: engine", first_step)

    def test_cuda_diagnostic_is_provenance_guarded_and_cannot_launch_proof(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        job = workflow.split("  cuda_diagnostic:\n", 1)[1].split("  reference:\n", 1)[0]
        probe = (ROOT / "scripts/ci/yue2_cuda_context_diagnostic.ps1").read_text(encoding="utf-8")
        self.assertIn("if: inputs.stage == 'cuda-diagnostic'", job)
        self.assertEqual(self.concurrency_group(self.concurrency_settings(workflow)["group"],
                                               "cuda-diagnostic", "101"), "inference-real-weights-physical-host")
        self.assertIn("$env:GITHUB_SHA -cne $env:EXPECTED_CONTROL_SHA", job)
        self.assertIn("(git -C ../engine rev-parse HEAD).Trim() -cne $env:EXPECTED_ENGINE_SHA", job)
        self.assertIn("diagnostic_pid must be 0 or a positive decimal PID", job)
        self.assertIn("if: always()", job)
        self.assertNotIn("cargo ", job)
        self.assertNotIn("download-artifact", job)
        self.assertNotIn("yue2_precision_proof.py run", job)
        for required in ("cuDeviceGetLuid", "cuDeviceGetPCIBusId", "Get-Counter",
                         "Get-AuthenticodeSignature", "process-before", "process-after",
                         "compute-apps-$gpu-$i", "pmon-$gpu-$i", "driverInitializationOnly",
                         "-ListSet $name", "windows-counter-catalog.json",
                         "GPU Process Memory(*)\\Dedicated Usage", "GPU Process Memory(*)\\Shared Usage"):
            self.assertIn(required, probe)
        for forbidden in ("extern int cuCtxCreate", "extern int cuDevicePrimaryCtxRetain",
                          "extern int cudaMalloc", "Start-Process", "Stop-Process"):
            self.assertNotIn(forbidden, probe)

    def test_optional_hyperv_process_mapping_is_read_only_metadata(self):
        probe = (ROOT / "scripts/ci/yue2_cuda_context_diagnostic.ps1").read_text(encoding="utf-8")
        lookup = probe.split("function Get-HyperVVmMapping", 1)[1].split(
            "function Save-ProcessIdentity", 1)[0]
        identity = probe.split("function Save-ProcessIdentity", 1)[1].split(
            "function Save-Counters", 1)[0]
        self.assertIn("Get-CimInstance -Namespace 'root\\virtualization\\v2'", lookup)
        self.assertIn("-ClassName 'Msvm_ComputerSystem'", lookup)
        self.assertIn('-Filter "ProcessID = $ProcessId" -OperationTimeoutSec 10 -ErrorAction Stop', lookup)
        for field in ("elementName", "name", "processId"):
            self.assertIn(f"{field} =", lookup)
        self.assertIn("status = 'error'", lookup)
        self.assertIn("errorCategory =", lookup)
        self.assertLess(identity.index("if ($TargetPid -eq 0)"),
                        identity.index("Get-HyperVVmMapping -ProcessId $TargetPid"))
        self.assertEqual(identity.count("hyperVVm = $vm"), 3)
        for forbidden in ("Start-VM", "Stop-VM", "Restart-VM", "Set-VM",
                          "Invoke-CimMethod", "Set-CimInstance", "Remove-CimInstance",
                          "-Credential"):
            self.assertNotIn(forbidden, lookup)


if __name__ == "__main__":
    unittest.main()
