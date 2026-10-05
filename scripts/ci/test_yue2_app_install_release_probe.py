"""CPU-only release-observation contract tests; no Windows process or GPU calls."""

import importlib.util
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("yue2_app_install_release_probe.py")
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
            "oldRootInCommandLine": MODULE.has_old_root(command, MODULE.OLD_RUN_ROOT) if command else False,
            "oldRootInExecutable": MODULE.has_old_root(executable, MODULE.OLD_RUN_ROOT) if executable else False,
            "workerIdInCommandLine": MODULE.OLD_WORKER_ID.casefold() in command.casefold() if command else False}


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
                            command='node.exe "' + MODULE.OLD_RUN_ROOT + r'\child.exe"'))
        if malformed_after_unknown and suffix == "before":
            malformed = row(pid=24518, created=created.isoformat())
            malformed["parentPid"] = None
            rows.append(malformed)
        result = snapshot(rows, start.isoformat(), (start + timedelta(milliseconds=200)).isoformat())
        if changed_collector and suffix == "after":
            result["rows"][-1]["createdUtc"] = (start - timedelta(seconds=10)).isoformat()
        return result

    return make(first_start, "before"), make(second_start, "after")


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
    def test_frozen_target_matches_stopped_app8_run_receipt(self):
        self.assertEqual((MODULE.OLD_RUN_ID, MODULE.OLD_RUN_ATTEMPT, MODULE.OLD_JOB_ID),
                         ("37295993157", "1", "111717219895"))
        self.assertEqual((MODULE.OLD_CONTROL_SHA, MODULE.OLD_APP_SHA, MODULE.OLD_ENGINE_SHA),
                         ("6a14bcd709f68f8ebcb6a3b2a360fe04aa5b96d8",
                          "c1f86907ae41183fa8ddc9126a821df36cc597dd",
                          "25bd55cdb6a56c78b07584a12150c9f5d46be439"))
        self.assertEqual(MODULE.OLD_RUN_ROOT,
                         r"E:\sceneworks-terminal\sc-23002-yue2-precision\37295993157-1")
        self.assertEqual((MODULE.OLD_WORKER_ID, MODULE.OLD_RUNNER, MODULE.OLD_HOST),
                         ("yue2-acceptance-ed59cf8e2088", "cuda-windows", "MICHAEL-TRX50"))
        self.assertEqual(MODULE.OLD_JOB_STARTED.isoformat(), "2026-10-05T10:35:14+00:00")
        self.assertEqual(MODULE.OLD_JOB_COMPLETED, "2026-10-05T10:59:07+00:00")
        self.assertEqual(MODULE.OLD_METRICS_ZIP_SHA256,
                         "e4fe7f5ad2ed80b2f3294064f49f113ac5a83a0ef734a0bcfc3d5160cfead5ad")
        collector = SOURCE.with_name("yue2_app_install_release_processes.ps1").read_text(encoding="utf-8")
        self.assertIn(MODULE.OLD_RUN_ROOT, collector)
        self.assertIn(MODULE.OLD_WORKER_ID, collector)

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
        ps_source = SOURCE.with_name("yue2_app_install_release_processes.ps1").read_text()
        self.assertIn("sceneworks_worker-[0-9a-f]{16}", ps_source)
        unknown = row(pid=5232, name=name, executable=None, command=None)
        with self.assertRaises(MODULE.TransientCandidateError):
            MODULE.validate_snapshot(snapshot([unknown]))
        with self.assertRaisesRegex(ValueError, "old app install process"):
            MODULE.validate_snapshot(snapshot([row(pid=5232, name=name,
                executable=MODULE.OLD_RUN_ROOT + "\\target\\release\\deps\\" + name)]))

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
        suffix = row(command='node.exe "' + MODULE.OLD_RUN_ROOT + '-other\\server.js"')
        self.assertFalse(suffix["oldRootInCommandLine"])
        self.assertFalse(MODULE.has_old_root(suffix["executablePath"], MODULE.OLD_RUN_ROOT))
        self.assertEqual(MODULE.validate_snapshot(snapshot([suffix]))["oldOwnedMatches"], [])

    def test_old_root_or_worker_in_either_snapshot_refuses(self):
        normal = snapshot([row()])
        old_root = snapshot([row(command='node.exe "' + MODULE.OLD_RUN_ROOT + r'\install\server.js"')],
                            "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00")
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(normal, old_root)
        worker = snapshot([row(command='node.exe --worker-id ' + MODULE.OLD_WORKER_ID)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(worker, snapshot([row()], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_exe = snapshot([row(name="git.exe", executable=MODULE.OLD_RUN_ROOT + r"\tools\git.exe")])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(arbitrary_exe, snapshot([], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_command = snapshot([row(name="git.exe", command="git.exe --work-tree " + MODULE.OLD_RUN_ROOT)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(arbitrary_command, snapshot([], "2026-10-05T11:00:04+00:00", "2026-10-05T11:00:05+00:00"))
        arbitrary_worker = snapshot([row(name="git.exe", command="git.exe --worker " + MODULE.OLD_WORKER_ID)])
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
