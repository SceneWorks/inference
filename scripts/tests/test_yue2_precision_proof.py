"""CPU-only controls for the dispatch-only precision proof."""
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import copy
import base64
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
        self.assertEqual(IDLE.RUN_ID, "37073714206")
        self.assertEqual(IDLE.BASELINE_DIGEST,
                         "a05afe09223020d39f698f9ec5ed9bc4f2258a1fa8950e69ee4fa9e2769339a1")
        IDLE.check_dispatch(IDLE.RUN_ID, "b" * 40, "a" * 40, "a" * 40)
        for run_id, engine, control, github in (
            ("36956986577", IDLE.BASELINE_ENGINE_SHA, "a" * 40, "a" * 40),
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

    def test_saved_runner_is_pinned_and_fresh_runner_matches_an_eligible_listener(self):
        self.assertEqual(IDLE.BASELINE_RUNNER, "cuda-windows-2")
        source = {"completed": True, "targetPid": 38212, "engineSha": IDLE.BASELINE_ENGINE_SHA,
                  "controlSha": IDLE.BASELINE_CONTROL_SHA}
        with patch.object(IDLE, "read_json", return_value={**source, "runner": "cuda-windows"}), \
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
                 patch.object(IDLE, "census_mixed_context") as attestation:
                self.assertEqual(CONTROL.cuda_census()[1], [])
                attestation.assert_not_called()
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
                 patch.object(CONTROL, "retain_reviewed_baseline", return_value=[]), \
                 patch.object(CONTROL, "cuda_physical_census", return_value=(census, [])) as physical, \
                 patch.object(CONTROL, "retain_cuda_physical_evidence", return_value=[]), \
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
            self.assertEqual(physical.call_count, 2)
            self.assertTrue((evidence / "external-samples.json").is_file())
            self.assertTrue((evidence / "census-after.txt").is_file())

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
        good = "test explicit_stage_precision_real_weights ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out\n"
        self.assertTrue(CONTROL.one_test_executed(good))
        for bad in (good.replace("1 passed", "0 passed"), good.replace("explicit_stage_precision_real_weights", "wrong_test"), good.replace("0 ignored", "1 ignored")):
            self.assertFalse(CONTROL.one_test_executed(bad))

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
        self.assertIn("group: inference-real-weights-physical-host", source)
        self.assertIn('CUDA_VISIBLE_DEVICES: "0"', source)
        self.assertEqual(source.count("path: ${{ env.YUE2_PRECISION_WORK_DIR }}/**/*.wav"), 2)
        self.assertEqual(source.count("if: ${{ always() && env.YUE2_PRECISION_WORK_DIR != '' }}"), 2)
        self.assertNotIn("path: ${{ env.YUE2_PRECISION_WORK_DIR }}\n", source)
        self.assertIn("yue2-precision-listening-cuda-cc-by-nc-internal-", source)
        self.assertIn("yue2-precision-listening-metal-cc-by-nc-internal-", source)
        self.assertIn("test \"$RUNNER_NAME\" = nax-macos-2", source)
        self.assertIn("--test precision_real_weights", source)
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

    def test_cuda_diagnostic_is_provenance_guarded_and_cannot_launch_proof(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        job = workflow.split("  cuda_diagnostic:\n", 1)[1].split("  reference:\n", 1)[0]
        probe = (ROOT / "scripts/ci/yue2_cuda_context_diagnostic.ps1").read_text(encoding="utf-8")
        self.assertIn("if: inputs.stage == 'cuda-diagnostic'", job)
        self.assertIn("group: inference-real-weights-physical-host", workflow)
        self.assertIn("$env:GITHUB_SHA -cne $env:EXPECTED_CONTROL_SHA", job)
        self.assertIn("(git -C ../engine rev-parse HEAD).Trim() -cne $env:EXPECTED_ENGINE_SHA", job)
        self.assertIn("diagnostic_pid must be a positive decimal PID", job)
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


if __name__ == "__main__":
    unittest.main()
