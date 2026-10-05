"""CPU-only release-observation contract tests; no Windows process or GPU calls."""

import importlib.util
from hashlib import sha256
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).with_name("yue2_app_install_release_probe.py")
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("yue2_app_install_release_probe", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def row(pid=100, name="node.exe", created="2026-10-05T05:00:00+00:00",
        executable=r"C:\Program Files\nodejs\node.exe", command=r'"C:\Program Files\nodejs\node.exe" C:\foreign\run.js'):
    return {"pid": pid, "parentPid": 50, "name": name, "createdUtc": created,
            "executablePath": executable, "commandLineAvailable": command is not None,
            "commandLineLength": len(command) if command is not None else None,
            "commandLineSha256": sha256(command.encode()).hexdigest() if command is not None else None,
            "oldRootInCommandLine": MODULE.has_old_root(command, MODULE.OLD_RUN_ROOT) if command else False,
            "workerIdInCommandLine": MODULE.OLD_WORKER_ID.casefold() in command.casefold() if command else False}


def snapshot(rows, start="2026-10-05T06:00:00+00:00", end="2026-10-05T06:00:01+00:00"):
    return {"complete": True, "queriedUtc": start, "completedUtc": end, "rows": rows}


class ReleaseProbeTests(unittest.TestCase):
    def test_foreign_and_gpu0_hosting_do_not_block_old_run_absence(self):
        foreign = row(command='node.exe --token dummy-secret C:\\foreign\\run.js')
        self.assertNotIn("dummy-secret", str(foreign))
        before = snapshot([foreign])
        after = snapshot([foreign], "2026-10-05T06:00:04+00:00", "2026-10-05T06:00:05+00:00")
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
                            "2026-10-05T06:00:04+00:00", "2026-10-05T06:00:05+00:00")
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(normal, old_root)
        worker = snapshot([row(command='node.exe --worker-id ' + MODULE.OLD_WORKER_ID)])
        with self.assertRaisesRegex(ValueError, "old app install"):
            MODULE.validate_pair(worker, snapshot([row()], "2026-10-05T06:00:04+00:00", "2026-10-05T06:00:05+00:00"))

    def test_new_candidate_with_inaccessible_identity_refuses_but_preexisting_foreign_is_allowed(self):
        unknown = row(command=None)
        with self.assertRaisesRegex(ValueError, "newer candidate"):
            MODULE.validate_snapshot(snapshot([unknown]))
        old_foreign = row(created="2026-10-05T04:00:00+00:00", command=None)
        self.assertEqual(MODULE.validate_snapshot(snapshot([old_foreign]))["preexistingIncompleteCandidates"], 1)

    def test_incomplete_stale_or_duplicate_census_refuses(self):
        for invalid in (snapshot([row()], end="2026-10-05T05:59:59+00:00"),
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
        self.assertNotIn("commandLine = $command", collector)
        self.assertNotIn("Stop-Process", collector)
        self.assertNotIn("Set-CimInstance", collector)


if __name__ == "__main__":
    unittest.main()
