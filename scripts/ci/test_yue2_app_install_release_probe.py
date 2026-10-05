"""CPU-only release-observation contract tests; no Windows process or GPU calls."""

import importlib.util
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile


SOURCE = Path(__file__).with_name("yue2_app_install_release_probe.py")
TEST_WORKER_ID = "yue2-acceptance-test-worker"
PRECISE_BIRTHS = (
    "2026-10-05T13:30:44.4259790Z",
    "2026-10-05T13:33:24.8993850Z",
    "2026-10-05T13:36:04.5215930Z",
)
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("yue2_app_install_release_probe", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        current = cls.fromisoformat("2026-10-05T11:01:00+00:00")
        return current.astimezone(tz) if tz else current


class MutableDateTime(datetime):
    current = "2026-10-05T11:01:00+00:00"

    @classmethod
    def now(cls, tz=None):
        current = cls.fromisoformat(cls.current)
        return current.astimezone(tz) if tz else current


def row(pid=100, name="node.exe", created="2026-10-05T10:50:00+00:00",
        executable=r"C:\Program Files\nodejs\node.exe", command=r'"C:\Program Files\nodejs\node.exe" C:\foreign\run.js'):
    return {"pid": pid, "parentPid": 50, "name": name, "createdUtc": created,
            "executablePath": executable, "commandLineAvailable": command is not None,
            "commandLineLength": len(command) if command is not None else None,
            "commandLineSha256": sha256(command.encode()).hexdigest() if command is not None else None,
            "oldRootInCommandLine": MODULE.has_old_root(command, MODULE.TARGET_RUN_ROOT) if command else False,
            "oldRootInExecutable": MODULE.has_old_root(executable, MODULE.TARGET_RUN_ROOT) if executable else False,
            "workerIdInCommandLine": TEST_WORKER_ID.casefold() in command.casefold() if command else False}


def snapshot(rows, start="2026-10-05T11:00:00+00:00", end="2026-10-05T11:00:01+00:00",
             witness=True):
    collector = row(pid=999, name="powershell.exe", created="2026-10-05T10:59:00+00:00",
                    executable=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                    command=r"powershell.exe -File C:\new-release-check\collector.ps1")
    retained = rows + ([collector] if witness else [])
    return {"complete": True, "queriedUtc": start, "completedUtc": end,
            "collectorPid": 999, "totalCimCount": len(retained) + 20, "rows": retained}


def snapshot_pair(pair_number, inaccessible=False, persistent=False, old_after_unknown=False,
                  malformed_after_unknown=False, changed_collector=False):
    base = datetime.fromisoformat("2026-10-05T11:00:00+00:00") + timedelta(seconds=7 * (pair_number - 1))
    first_start = base
    second_start = base + timedelta(seconds=3, milliseconds=500)

    def make(start, suffix):
        created = start + timedelta(milliseconds=100)
        rows = []
        if inaccessible and suffix == "before" or persistent and suffix in ("before", "after"):
            rows.append(row(pid=24516, created=created.isoformat(), command=None))
        if old_after_unknown and suffix == "before":
            rows.append(row(pid=24517, created=created.isoformat(),
                            command='node.exe "' + MODULE.TARGET_RUN_ROOT + r'\child.exe"'))
        if malformed_after_unknown and suffix == "before":
            malformed = row(pid=24518, created=created.isoformat())
            malformed["parentPid"] = None
            rows.append(malformed)
        result = snapshot(rows, start.isoformat(), (start + timedelta(milliseconds=200)).isoformat())
        if changed_collector and suffix == "after":
            result["rows"][-1]["createdUtc"] = (start - timedelta(seconds=10)).isoformat()
        return result

    return make(first_start, "before"), make(second_start, "after")


def write_source_bound_case(metrics: Path, name: str, index: int, outcome="completed", faults=b"",
                            process_witness=True, layout="profile"):
    case_dir = metrics / layout / name
    case_dir.mkdir(parents=True, exist_ok=True)
    pid, parent = 5232 + index, 4000 + index
    exe_name = f"sceneworks_worker-{index:016x}.exe"
    exe = str(MODULE.PureWindowsPath(MODULE.TARGET_RUN_ROOT) / "target" / "release" / "deps" / exe_name)
    case_bytes = (SOURCE.parent / "yue2-app-precision-cases" / f"{name}.json").read_bytes()
    (case_dir / "case.json").write_bytes(case_bytes)
    precise_birth = PRECISE_BIRTHS[index] if index < len(PRECISE_BIRTHS) else \
        f"2026-10-05T13:{38 + index:02d}:00.0000000Z"
    record_birth = precise_birth[:23] + "Z"
    journal = {"pid": pid, "luid": "luid_0x00000000_0x0001f78f", "bytes": 1024,
               "counter": {"parentPid": parent, "createdUtc": precise_birth}}
    journal_bytes = (json.dumps(journal) + "\n").encode()
    (case_dir / "cuda-owned-samples.jsonl").write_bytes(journal_bytes)
    (case_dir / "cuda-owned-faults.jsonl").write_bytes(faults)
    (case_dir / "profile-process.json").write_text(json.dumps({"processId": pid}), encoding="utf-8")
    _, case_name, _, policy, _, _ = MODULE.CASES[name]
    owned = {"journalSha256": sha256(journal_bytes).hexdigest(),
             "selectedLuid": "luid_0x00000000_0x0001f78f"}
    if process_witness:
        owned["process"] = {"pid": pid, "parentPid": parent,
            "createdUtc": record_birth, "executablePath": exe,
            "executableSha256": f"{index + 1:064x}"}
    record = {"caseId": MODULE.case_id("cuda", name), "backend": "cuda",
              "request": {"name": case_name, "computePolicy": policy},
              "outcome": {"status": outcome},
              "measured": {"owned": owned}}
    (case_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
    return sha256(case_bytes).hexdigest()


def binding_metadata(metrics_zip: Path, run_conclusion="failure", job_conclusion="failure"):
    run = {"id": MODULE.TARGET_RUN_ID, "run_attempt": MODULE.TARGET_ATTEMPT,
           "head_sha": MODULE.TARGET_CONTROL_SHA, "event": "workflow_dispatch",
           "path": MODULE.TARGET_WORKFLOW, "status": "completed", "conclusion": run_conclusion,
           "repository": {"full_name": MODULE.TARGET_REPOSITORY}}
    job = {"id": MODULE.TARGET_JOB_ID, "name": "cuda", "run_id": MODULE.TARGET_RUN_ID,
           "status": "completed", "conclusion": job_conclusion,
           "runner_name": MODULE.TARGET_RUNNER, "runner_id": MODULE.TARGET_RUNNER_ID,
           "started_at": "2026-10-05T13:25:52Z", "completed_at": "2026-10-05T14:00:00Z"}
    artifact = {"id": MODULE.TARGET_ARTIFACT_ID, "name": MODULE.TARGET_ARTIFACT_NAME, "expired": False,
           "workflow_run": {"id": MODULE.TARGET_RUN_ID, "run_attempt": MODULE.TARGET_ATTEMPT,
                                 "head_sha": MODULE.TARGET_CONTROL_SHA},
                "digest": "sha256:" + MODULE.file_sha256(metrics_zip)}
    return run, job, artifact


class FakePairCollector:
    def __init__(self, evidence, pairs, finish_callback=None):
        self.evidence = evidence
        self.pairs = pairs
        self.read_count = 0
        self.decisions = []
        self.finished = False
        self.finish_callback = finish_callback

    def snapshot_paths(self, pair_number):
        stem = "process-snapshot" if pair_number == 1 else f"process-snapshot-{pair_number}"
        return (self.evidence / f"{stem}-before.json", self.evidence / f"{stem}-after.json")

    def read_pair(self, pair_number):
        self.read_count += 1
        paths = self.snapshot_paths(pair_number)
        for path, payload in zip(paths, self.pairs[pair_number - 1]):
            path.write_text(json.dumps(payload), encoding="utf-8")
        return paths

    def decide_continue(self, pair_number, should_continue):
        self.decisions.append((pair_number, should_continue))

    def finish(self):
        self.finished = True
        if self.finish_callback:
            self.finish_callback()

    def stop(self):
        self.decisions.append((self.read_count, False))


class ReleaseProbeTests(unittest.TestCase):
    def _binding_fixture(self, root: Path, count: int, *, outcome="completed", no_process_at=None):
        metrics = root / "metrics"
        metrics.mkdir()
        (metrics / "sources.json").write_text(json.dumps({
            "control_sha": MODULE.TARGET_CONTROL_SHA, "app_sha": MODULE.TARGET_APP_SHA,
            "engine_sha": MODULE.TARGET_ENGINE_SHA, "app_pins": [MODULE.TARGET_ENGINE_SHA]}),
            encoding="utf-8")
        target_device = {"physicalMode": "shared-gpu1", "physicalIndex": 1, "cudaOrdinal": 0,
            "uuid": "GPU-e4b79931-7be6-f216-460a-f5405cfafffe", "pci": "00000000:C1:00.0",
            "luid": "luid_0x00000000_0x0001f78f"}
        (metrics / "preflight-initial.json").write_text(json.dumps({"backend": "cuda", "label": "initial",
            "admitted": True, "runner": MODULE.TARGET_RUNNER, "hostname": "MICHAEL-TRX50",
            "census": json.dumps({"validatedDevice": target_device})}),
            encoding="utf-8")
        manifest_rows = []
        for index, name in enumerate(MODULE.NAMES[:count]):
            case_hash = write_source_bound_case(metrics, name, index,
                outcome=outcome if index == count - 1 else "completed",
                faults=b"{}\n" if index == count - 1 and outcome != "completed" else b"",
                process_witness=index != no_process_at,
                layout="profile" if outcome == "completed" and count == len(MODULE.NAMES)
                    else "partial-profile")
            manifest_rows.append({"name": name, "case_id": MODULE.case_id("cuda", name),
                "source_sha256": MODULE.CASE_SOURCE_SHA256[name], "run_case_sha256": case_hash})
        (metrics / "cases-manifest.json").write_text(json.dumps({"backend": "cuda",
            "cases": [{"name": name, "case_id": MODULE.case_id("cuda", name),
                "source_sha256": MODULE.CASE_SOURCE_SHA256[name],
                "run_case_sha256": next((r["run_case_sha256"] for r in manifest_rows if r["name"] == name),
                    "0" * 64)} for name in MODULE.NAMES]}), encoding="utf-8")
        zip_path = root / "metrics.zip"
        zip_path.write_bytes(b"authentic-metrics-fixture")
        run, job, artifact = binding_metadata(zip_path,
            run_conclusion="success" if outcome == "completed" and count == len(MODULE.NAMES) else "failure",
            job_conclusion="success" if outcome == "completed" and count == len(MODULE.NAMES) else "failure")
        return metrics, zip_path, run, job, artifact

    def test_frozen_target_matches_current_app8_run_receipt(self):
        self.assertEqual((MODULE.TARGET_RUN_ID, MODULE.TARGET_ATTEMPT, MODULE.TARGET_JOB_ID),
                         (37314391667, 1, 111777364558))
        self.assertEqual((MODULE.TARGET_CONTROL_SHA, MODULE.TARGET_APP_SHA, MODULE.TARGET_ENGINE_SHA),
                         ("2c820231926094566d1ed719c085ba7a7a2284a5",
                          "c1f86907ae41183fa8ddc9126a821df36cc597dd",
                          "25bd55cdb6a56c78b07584a12150c9f5d46be439"))
        self.assertEqual(MODULE.TARGET_RUN_ROOT,
                         r"E:\sceneworks-terminal\sc-23002-yue2-precision\37314391667-1")
        self.assertEqual((MODULE.TARGET_RUNNER, MODULE.TARGET_RUNNER_ID), ("cuda-windows-2", 2619))
        collector = SOURCE.with_name("yue2_app_install_release_processes.ps1").read_text(encoding="utf-8")
        self.assertIn("$env:YUE2_RELEASE_OLD_ROOT", collector)
        self.assertNotIn("37295993157", collector)
        self.assertNotIn("yue2-acceptance-ed59cf8e2088", collector)

    def test_collect_assembles_valid_partial_outcomes_without_claiming_full_capture(self):
        target = {"runId": MODULE.TARGET_RUN_ID, "attempt": MODULE.TARGET_ATTEMPT,
            "jobId": MODULE.TARGET_JOB_ID, "runner": MODULE.TARGET_RUNNER,
            "runnerId": MODULE.TARGET_RUNNER_ID, "runRoot": MODULE.TARGET_RUN_ROOT,
            "workerId": None, "hostname": "MICHAEL-TRX50",
            "selectedDevice": {"physicalIndex": 1, "cudaOrdinal": 0,
                "uuid": "GPU-e4b79931-7be6-f216-460a-f5405cfafffe",
                "pci": "00000000:C1:00.0", "luid": "luid_0x00000000_0x0001f78f"}}
        binding = {"target": target, "runConclusion": "failure", "jobConclusion": "failure",
            "allObservedCaseOutcomesCompleted": True, "captureRecordSetComplete": False,
            "captureAcceptanceEvaluated": False, "releaseScope": "known recorded case generations",
            "caseProcessWitnesses": [], "binaryHashTargets": []}
        census = {"physicalMode": "shared-gpu1", "validatedDevice": target["selectedDevice"]}
        api = {"id": MODULE.TARGET_RUN_ID}
        release_job = {"runner_name": MODULE.TARGET_RUNNER, "runner_id": MODULE.TARGET_RUNNER_ID}

        class Collector:
            def __init__(self, *_args, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "os",
             SimpleNamespace(name="nt", environ=os.environ)), \
             patch.dict("os.environ", {"GITHUB_REPOSITORY": "SceneWorks/inference",
                "GITHUB_JOB": "cuda_release_check", "GITHUB_RUN_ATTEMPT": "1",
                "RUNNER_NAME": MODULE.TARGET_RUNNER, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "1", "EXPECTED_APP_SHA": MODULE.TARGET_APP_SHA,
                "EXPECTED_ENGINE_SHA": MODULE.TARGET_ENGINE_SHA,
                "EXPECTED_CONTROL_SHA": MODULE.TARGET_CONTROL_SHA,
                "GITHUB_RUN_ID": "9001", "GITHUB_SHA": "c" * 40,
                "COMPUTERNAME": "MICHAEL-TRX50"}), \
             patch.object(MODULE, "verify_sources", return_value={"control_sha": MODULE.TARGET_CONTROL_SHA}), \
             patch.object(MODULE, "fetch_authenticated_target",
                return_value=(api, api, api, release_job, Path(directory) / "metrics.zip")), \
             patch.object(MODULE, "safe_extract_metrics", return_value={"extractedRoot": directory}), \
             patch.object(MODULE, "derive_run_binding", return_value=binding), \
             patch.object(MODULE, "rehash_recorded_binaries", return_value=[]), \
             patch.object(MODULE, "cuda_physical_census", return_value=(json.dumps(census), [])), \
             patch.object(MODULE, "retain_cuda_physical_evidence", return_value=[]), \
             patch.object(MODULE, "PowerShellSnapshotCollector", Collector), \
             patch.object(MODULE, "validate_release_pairs", return_value={
                 "processFiles": [], "refusalFiles": [], "before": {}, "after": {}}):
            result = MODULE.collect(Path(directory) / "result", Path(directory) / "app",
                Path(directory) / "engine", Path(directory) / "control")
            saved_binding = json.loads((Path(directory) / "result" / "case-binding.json").read_text())
        self.assertTrue(result["captureRecordsComplete"])
        self.assertFalse(result["captureRecordSetComplete"])
        self.assertFalse(result["captureAcceptanceEvaluated"])
        self.assertEqual(saved_binding["runConclusion"], "failure")
        self.assertEqual(saved_binding["jobConclusion"], "failure")

    def test_run_binding_uses_exact_eight_verified_case_generations_and_binary_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 8)
            binding = MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            self.assertEqual(binding["sourceCaseIds"], [MODULE.case_id("cuda", name) for name in MODULE.NAMES])
            self.assertTrue(binding["captureRecordSetComplete"])
            self.assertFalse(binding["captureAcceptanceEvaluated"])
            self.assertEqual(len(binding["caseProcessWitnesses"]), 8)
            self.assertEqual(len(binding["binaryHashTargets"]), 8)
            expected = {row["path"]: row["sha256"] for row in binding["binaryHashTargets"]}
            actual = MODULE.rehash_recorded_binaries(binding,
                lambda path: expected[path])
            self.assertEqual(len(actual), 8)
            witness = binding["caseProcessWitnesses"][0]
            row = globals()["row"](pid=witness["pid"], created=witness["preciseCreatedUtc"])
            row["parentPid"] = witness["parentPid"]
            snapshot_payload = snapshot([row], start="2026-10-05T13:40:00+00:00",
                end="2026-10-05T13:40:01+00:00")
            with self.assertRaisesRegex(ValueError, "old app install process"):
                MODULE.validate_snapshot(snapshot_payload,
                binding["target"]["runRoot"], recorded_generations=binding["caseProcessWitnesses"],
                job_started=binding["target"]["jobStartedUtc"])

    def test_precise_sampler_birth_matches_all_three_real_windows_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 3)
            binding = MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            witnesses = binding["caseProcessWitnesses"]
            self.assertEqual([item["preciseCreatedUtc"] for item in witnesses], [
                "2026-10-05T13:30:44.425979+00:00",
                "2026-10-05T13:33:24.899385+00:00",
                "2026-10-05T13:36:04.521593+00:00",
            ])
            for index, witness in enumerate(witnesses):
                process = globals()["row"](pid=witness["pid"], created=witness["preciseCreatedUtc"])
                process["parentPid"] = witness["parentPid"]
                with self.assertRaisesRegex(ValueError, "old app install process"):
                    MODULE.validate_snapshot(snapshot([process], start="2026-10-05T13:40:00+00:00",
                        end="2026-10-05T13:40:01+00:00"), recorded_generations=witnesses,
                        job_started=binding["target"]["jobStartedUtc"])

            first = witnesses[0]
            record_path = metrics / "partial-profile" / MODULE.NAMES[0] / "record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["measured"]["owned"]["process"]["createdUtc"] = "2026-10-05T13:30:44.426Z"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "millisecond truncation"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)

            reused_pid = globals()["row"](pid=first["pid"], created="2026-10-05T13:39:00+00:00")
            foreign = MODULE.validate_snapshot(snapshot([reused_pid], start="2026-10-05T13:40:00+00:00",
                end="2026-10-05T13:40:01+00:00"), recorded_generations=witnesses,
                job_started=binding["target"]["jobStartedUtc"])
            self.assertEqual(foreign["recordedGenerationMatches"], [])

    def test_current_runner_route_and_device_binding_are_exact_but_allow_same_host_runner_alias(self):
        current = {"id": 9001, "head_sha": "c" * 40}
        jobs = {"total_count": 1, "jobs": [{"name": "cuda_release_check", "status": "in_progress",
            "runner_name": "cuda-windows", "runner_id": 812}]}
        with patch.dict("os.environ", {"GITHUB_RUN_ID": "9001", "GITHUB_SHA": "c" * 40}):
            selected = MODULE.select_release_runner(jobs, current, "cuda_release_check")
            self.assertEqual(selected["runner_name"], "cuda-windows")
            jobs["jobs"][0]["runner_id"] = MODULE.TARGET_RUNNER_ID
            with self.assertRaisesRegex(ValueError, "release-check job"):
                MODULE.select_release_runner(jobs, current, "cuda_release_check")
        device = {"physicalIndex": 1, "cudaOrdinal": 0, "uuid": "GPU-e4b79931-7be6-f216-460a-f5405cfafffe",
                  "pci": "00000000:C1:00.0", "luid": "luid_0x00000000_0x0001f78f"}
        self.assertEqual(MODULE.verify_same_selected_device(device,
            {"physicalMode": "shared-gpu1", "validatedDevice": dict(device)}), device)
        changed = dict(device, luid="luid_0x00000000_0x00000000")
        with self.assertRaisesRegex(ValueError, "differs"):
            MODULE.verify_same_selected_device(device,
                {"physicalMode": "shared-gpu1", "validatedDevice": changed})

    def test_partial_capture_preserves_only_observed_prefix_and_does_not_invent_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(
                Path(directory), 2, outcome="failed", no_process_at=1)
            binding = MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            self.assertEqual(binding["sourceCaseIds"], [MODULE.case_id("cuda", name) for name in MODULE.NAMES[:2]])
            self.assertFalse(binding["captureRecordSetComplete"])
            self.assertFalse(binding["captureAcceptanceEvaluated"])
            self.assertEqual(binding["caseLayout"], "partial-profile")
            self.assertEqual([item["pid"] for item in binding["caseProcessWitnesses"]], [5232])
            self.assertEqual(binding["caseOutcomeStatuses"], ["completed", "failed"])
            self.assertEqual(binding["caseProcessWitnesses"][0]["faultBytes"], 0)
            self.assertEqual(binding["rawSamplerReceipts"][1]["sampleRows"], 1)
            self.assertEqual(binding["rawSamplerReceipts"][1]["faultsBytes"], 3)

    def test_no_records_yields_run_root_only_scope_and_nonprefix_or_wrong_source_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 0)
            binding = MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            self.assertEqual(binding["caseProcessWitnesses"], [])
            self.assertIn("run-root markers only", binding["releaseScope"])
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 2)
            (metrics / "partial-profile" / MODULE.NAMES[0] / "record.json").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous capture prefix"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            run["head_sha"] = "f" * 40
            with self.assertRaisesRegex(ValueError, "run identity/source"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)

    def test_binding_refuses_artifact_zip_and_source_or_executable_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 1)
            artifact["digest"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "artifact binding"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            artifact["digest"] = "sha256:" + MODULE.file_sha256(zip_path)
            (metrics / "sources.json").write_text(json.dumps({"control_sha": "f" * 40}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/pin"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)

    def test_fixed_case_source_and_selected_device_mutations_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, zip_path, run, job, artifact = self._binding_fixture(Path(directory), 1)
            manifest_path = metrics / "cases-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["source_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest source differs"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)
            manifest["cases"][0]["source_sha256"] = MODULE.CASE_SOURCE_SHA256[MODULE.NAMES[0]]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            record_path = metrics / "partial-profile" / MODULE.NAMES[0] / "record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["measured"]["owned"]["selectedLuid"] = "luid_0x00000000_0x00000000"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected device differs"):
                MODULE.derive_run_binding(metrics, run, job, artifact, zip_path)

    def test_binary_rehash_refuses_changed_or_out_of_tree_executable(self):
        binding = {"binaryHashTargets": [{"path": str(MODULE.PureWindowsPath(MODULE.TARGET_RUN_ROOT) /
            "target" / "release" / "deps" / "sceneworks_worker-0000000000000001.exe"), "sha256": "a" * 64}]}
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            MODULE.rehash_recorded_binaries(binding, lambda _path: "b" * 64)
        binding["binaryHashTargets"][0]["path"] = r"E:\other\sceneworks_worker-0000000000000001.exe"
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.rehash_recorded_binaries(binding, lambda _path: "a" * 64)

    def test_original_zip_digest_and_safe_extraction_refuse_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zip_path = root / "metrics.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.writestr("evidence/sources.json", "{}")
                archive.writestr("install/evidence/summary.json", "{}")
            artifact = {"id": MODULE.TARGET_ARTIFACT_ID, "name": MODULE.TARGET_ARTIFACT_NAME, "expired": False,
                "workflow_run": {"head_sha": MODULE.TARGET_CONTROL_SHA},
                "digest": "sha256:" + MODULE.file_sha256(zip_path)}
            result = MODULE.safe_extract_metrics(zip_path, artifact, root / "extracted")
            self.assertEqual(Path(result["extractedRoot"]).name, "evidence")
            self.assertEqual(Path(result["installEvidenceRoot"]).name, "install")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zip_path = root / "bad.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.writestr("../escape.json", "{}");
            artifact = {"id": MODULE.TARGET_ARTIFACT_ID, "name": MODULE.TARGET_ARTIFACT_NAME, "expired": False,
                "workflow_run": {"head_sha": MODULE.TARGET_CONTROL_SHA},
                "digest": "sha256:" + MODULE.file_sha256(zip_path)}
            with self.assertRaisesRegex(ValueError, "unsafe"):
                MODULE.safe_extract_metrics(zip_path, artifact, root / "extracted")

    def test_clean_pair_stops_at_original_two_observation_guarantee(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "datetime", FixedDateTime):
            evidence = Path(directory)
            collector = FakePairCollector(evidence, [snapshot_pair(1),
                                                       snapshot_pair(2, persistent=True)])
            result = MODULE.validate_release_pairs(evidence, collector)
            self.assertEqual(collector.read_count, 1)
            self.assertEqual(collector.decisions, [(1, False)])
            self.assertTrue(collector.finished)
            self.assertEqual(len(result["pairs"]), 1)
            self.assertEqual({item["name"] for item in result["processFiles"]}, {
                "process-snapshot-before.json", "process-snapshot-after.json"})
            self.assertFalse((evidence / "process-snapshot-2-before.json").exists())
            for item in result["processFiles"]:
                self.assertEqual(item["sha256"], MODULE.file_sha256(evidence / item["name"]))

    def test_transient_pair_requires_one_later_complete_pair_and_retains_refusal(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "datetime", FixedDateTime):
            evidence = Path(directory)
            collector = FakePairCollector(evidence, [snapshot_pair(1, inaccessible=True),
                                                       snapshot_pair(2), snapshot_pair(3, persistent=True)])
            result = MODULE.validate_release_pairs(evidence, collector)
            self.assertEqual(collector.read_count, 2)
            self.assertEqual(collector.decisions, [(1, True), (2, False)])
            self.assertEqual([pair["valid"] for pair in result["pairs"]], [False, True])
            refusal = evidence / "initial-refusal.json"
            self.assertEqual(result["refusalFiles"][0]["sha256"], MODULE.file_sha256(refusal))
            self.assertNotIn("commandLine", refusal.read_text(encoding="utf-8"))
            self.assertFalse((evidence / "process-snapshot-3-before.json").exists())

    def test_persistent_or_unresolved_transient_candidate_never_proves_release(self):
        with patch.object(MODULE, "datetime", FixedDateTime):
            for pairs in (
                [snapshot_pair(i, persistent=True) for i in range(1, 4)],
                [snapshot_pair(i, inaccessible=True) for i in range(1, 4)],
            ):
                with tempfile.TemporaryDirectory() as directory:
                    evidence = Path(directory)
                    collector = FakePairCollector(evidence, pairs)
                    with self.assertRaisesRegex(ValueError, "no final complete process snapshot pair"):
                        MODULE.validate_release_pairs(evidence, collector)
                    self.assertEqual(collector.read_count, 3)
                    self.assertTrue((evidence / "initial-refusal.json").exists())

    def test_known_rust_test_worker_is_retained_and_unresolved_identity_refuses(self):
        name = "sceneworks_worker-625c59022a418279.exe"
        self.assertIsNotNone(MODULE.RELEVANT_NAME.fullmatch(name))
        ps_source = SOURCE.with_name("yue2_app_install_release_processes.ps1").read_text(encoding="utf-8")
        self.assertIn("sceneworks_worker-[0-9a-f]{16}", ps_source)
        unknown = row(pid=5232, name=name, executable=None, command=None)
        with self.assertRaises(MODULE.TransientCandidateError):
            MODULE.validate_snapshot(snapshot([unknown]))
        with self.assertRaisesRegex(ValueError, "old app install process"):
            MODULE.validate_snapshot(snapshot([row(pid=5232, name=name,
                executable=MODULE.TARGET_RUN_ROOT + "\\target\\release\\deps\\" + name)]))

    def test_old_marker_and_malformed_later_row_override_transient_retry(self):
        with patch.object(MODULE, "datetime", FixedDateTime):
            for pair, expected in ((snapshot_pair(1, inaccessible=True, old_after_unknown=True),
                                    "old app install"),
                                   (snapshot_pair(1, inaccessible=True, malformed_after_unknown=True),
                                    "PID/parent")):
                with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                    evidence = Path(directory)
                    collector = FakePairCollector(evidence, [pair, snapshot_pair(2), snapshot_pair(3)])
                    with self.assertRaisesRegex(ValueError, expected):
                        MODULE.validate_release_pairs(evidence, collector)
                    self.assertEqual(collector.read_count, 1)
                    self.assertEqual(collector.decisions[-1], (1, False))

    def test_collector_witness_timing_and_freshness_errors_are_terminal(self):
        with patch.object(MODULE, "datetime", FixedDateTime):
            cases = []
            missing_before, missing_after = snapshot_pair(1, inaccessible=True)
            missing_before["rows"] = [row(pid=999, name="powershell.exe")]  # no PID-999 self witness
            missing_before["collectorPid"] = 123
            cases.append(([(missing_before, missing_after)], "collector process missing"))
            cases.append(([snapshot_pair(1, changed_collector=True)],
                          "collector PID/name/creation changed between process snapshots"))
            pair1 = snapshot_pair(1, inaccessible=True)
            recycled = list(snapshot_pair(2))
            for payload in recycled:
                next(item for item in payload["rows"] if item["pid"] == payload["collectorPid"])["createdUtc"] = \
                    "2026-10-05T10:58:00+00:00"
            cases.append(([pair1, tuple(recycled)],
                          "collector PID/name/creation changed between process pairs"))
            too_close = list(snapshot_pair(1))
            too_close[1]["queriedUtc"] = (datetime.fromisoformat(too_close[0]["completedUtc"])
                                           + timedelta(seconds=1)).isoformat()
            cases.append(([(tuple(too_close))], "process snapshots overlap, are too close, or are stale"))
            stale = list(snapshot_pair(1))
            stale[0]["completedUtc"] = (datetime.fromisoformat(stale[0]["queriedUtc"])
                                          + timedelta(seconds=31)).isoformat()
            cases.append(([(tuple(stale))], "snapshot stale or unbounded"))
            for pairs, expected in cases:
                with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                    evidence = Path(directory)
                    collector = FakePairCollector(evidence, [*pairs, snapshot_pair(3)])
                    with self.assertRaisesRegex(ValueError, expected):
                        MODULE.validate_release_pairs(evidence, collector)
                    self.assertEqual(collector.read_count,
                                     2 if "between process pairs" in expected else 1)

    def test_inaccessible_collector_argv_is_terminal_not_transient(self):
        with patch.object(MODULE, "datetime", FixedDateTime), tempfile.TemporaryDirectory() as directory:
            before, after = snapshot_pair(1, inaccessible=True)
            own = next(item for item in before["rows"] if item["pid"] == before["collectorPid"])
            own["commandLineAvailable"] = False
            own["commandLineLength"] = None
            own["commandLineSha256"] = None
            collector = FakePairCollector(Path(directory), [(before, after),
                                                               snapshot_pair(2), snapshot_pair(3)])
            with self.assertRaisesRegex(ValueError, "collector PID is not the accessible PowerShell process"):
                MODULE.validate_release_pairs(Path(directory), collector)
            self.assertEqual(collector.read_count, 1)
            self.assertFalse((Path(directory) / "initial-refusal.json").exists())

    def test_final_freshness_is_rechecked_after_collector_finish(self):
        with patch.object(MODULE, "datetime", MutableDateTime), tempfile.TemporaryDirectory() as directory:
            MutableDateTime.current = "2026-10-05T11:01:00+00:00"
            collector = FakePairCollector(Path(directory), [snapshot_pair(1), snapshot_pair(2)],
                finish_callback=lambda: setattr(MutableDateTime, "current", "2026-10-05T11:03:00+00:00"))
            with self.assertRaisesRegex(ValueError, "process release observation is not fresh"):
                MODULE.validate_release_pairs(Path(directory), collector)
            self.assertEqual(collector.read_count, 1)

    def test_foreign_and_gpu0_hosting_do_not_block_old_run_absence(self):
        foreign = row(command='node.exe --token dummy-secret C:\\foreign\\run.js')
        self.assertNotIn("dummy-secret", str(foreign))
        before = snapshot([foreign])
        after = snapshot([foreign], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00")
        self.assertTrue(MODULE.validate_pair(before, after)["noOldOwnedMatch"])
        self.assertFalse(MODULE.RELEVANT_NAME.fullmatch("LM Studio.exe"))

    def test_old_root_match_respects_directory_boundary(self):
        suffix = row(command='node.exe "' + MODULE.TARGET_RUN_ROOT + '-other\\server.js"')
        self.assertFalse(suffix["oldRootInCommandLine"])
        self.assertFalse(MODULE.has_old_root(suffix["executablePath"], MODULE.TARGET_RUN_ROOT))
        self.assertEqual(MODULE.validate_snapshot(snapshot([suffix]))["oldOwnedMatches"], [])

    def test_old_root_or_worker_in_either_snapshot_refuses(self):
        normal = snapshot([row()])
        old_root = snapshot([row(command='node.exe "' + MODULE.TARGET_RUN_ROOT + r'\install\server.js"')],
                            "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00")
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(normal, old_root)
        worker = snapshot([row(command='node.exe --worker-id ' + TEST_WORKER_ID)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(worker, snapshot([row()], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_exe = snapshot([row(name="git.exe", executable=MODULE.TARGET_RUN_ROOT + r"\tools\git.exe")])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(arbitrary_exe, snapshot([], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_command = snapshot([row(name="git.exe", command="git.exe --work-tree " + MODULE.TARGET_RUN_ROOT)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(arbitrary_command, snapshot([], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_worker = snapshot([row(name="git.exe", command="git.exe --worker " + TEST_WORKER_ID)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(arbitrary_worker, snapshot([], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))

    def test_new_candidate_with_inaccessible_identity_refuses_but_preexisting_foreign_is_allowed(self):
        unknown = row(created="2026-10-05T10:36:00+00:00", command=None)
        with self.assertRaisesRegex(ValueError, "newer candidate"):
            MODULE.validate_snapshot(snapshot([unknown]))
        old_foreign = row(created="2026-10-05T10:00:00+00:00", command=None)
        self.assertEqual(MODULE.validate_snapshot(snapshot([old_foreign]))["preexistingIncompleteCandidates"], 1)

    def test_incomplete_stale_or_duplicate_census_refuses(self):
        for invalid in (snapshot([row()], end="2026-10-05T10:59:59+00:00"),
                        snapshot([row(), row()]),
                        {**snapshot([row()]), "complete": False}):
            with self.assertRaises(ValueError):
                MODULE.validate_snapshot(invalid)
        broken = row()
        broken["parentPid"] = None
        with self.assertRaisesRegex(ValueError, "PID/parent"):
            MODULE.validate_snapshot(snapshot([broken]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            MODULE.validate_pair(snapshot([row()]), snapshot([row()]))
        with self.assertRaisesRegex(ValueError, "collector/completeness witness"):
            MODULE.validate_snapshot(snapshot([], witness=False))
        wrong_name = snapshot([])
        wrong_name["rows"][-1]["name"] = "cmd.exe"
        with self.assertRaisesRegex(ValueError, "collector PID is not"):
            MODULE.validate_snapshot(wrong_name)
        recycled = snapshot([row()], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00")
        recycled["rows"][-1]["createdUtc"] = "2026-10-05T11:00:02+00:00"
        with self.assertRaisesRegex(ValueError, "collector PID/name/creation changed"):
            MODULE.validate_pair(snapshot([row()]), recycled)

    def test_workflow_is_opt_in_and_collects_no_model(self):
        workflow = (SOURCE.parents[2] / ".github/workflows/yue2-app-precision-profile.yml").read_text(encoding="utf-8")
        self.assertIn("release_check_only:\n        type: boolean\n        required: false\n        default: false", workflow)
        self.assertIn("cuda:\n    if: inputs.backend == 'cuda' && !inputs.release_check_only", workflow)
        self.assertIn("metal:\n    if: inputs.backend == 'metal' && !inputs.release_check_only", workflow)
        self.assertIn("invalid_release_check:\n    if: inputs.release_check_only && inputs.backend != 'cuda'", workflow)
        section = workflow.split("  cuda_release_check:\n", 1)[1].split("\n  metal:\n", 1)[0]
        self.assertIn("if: inputs.backend == 'cuda' && inputs.release_check_only", section)
        self.assertIn("yue2_app_install_release_probe.py", section)
        self.assertIn("GH_TOKEN: ${{ github.token }}", section)
        self.assertNotIn("run-captures", section)
        self.assertNotIn("profile-install-only", section)
        self.assertIn("CUDA_VISIBLE_DEVICES: \"1\"", section)
        collector = SOURCE.with_name("yue2_app_install_release_processes.ps1").read_text(encoding="utf-8")
        self.assertIn("Get-CimInstance -ClassName Win32_Process -OperationTimeoutSec 15", collector)
        self.assertIn("commandLineSha256 = $digest", collector)
        self.assertIn("oldRootInCommandLine = $rootMatch", collector)
        self.assertLess(collector.index("$rootMatch = Has-Old-Root $command"),
                        collector.index("$relevant = $_.Name -match"))
        self.assertIn("if ($relevant -or $rootMatch -or $exeRootMatch -or $workerMatch -or $_.ProcessId -eq $PID)", collector)
        self.assertIn("totalCimCount = $processes.Count", collector)
        self.assertIn("collectorPid = $PID", collector)
        self.assertNotIn("commandLine = $command", collector)
        self.assertNotIn("Stop-Process", collector)
        self.assertNotIn("Set-CimInstance", collector)


if __name__ == "__main__":
    unittest.main()
