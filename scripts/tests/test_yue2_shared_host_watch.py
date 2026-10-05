"""Fail-closed external shared-host inventory watcher regressions."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_shared_host_watch as watch
import yue2_reviewed_gpu1 as gpu1

SHA = "a" * 40

def listener(name, rid, busy):
    return {"name": name, "id": rid, "status": "online", "busy": busy,
            "labels": [{"name": item} for item in ("self-hosted", "Windows", "cuda")]}


def own_snapshot():
    own = {"id": 7, "head_sha": SHA, "run_attempt": 1, "event": "workflow_dispatch",
           "path": ".github/workflows/yue2-precision-proof.yml", "status": "in_progress",
           "conclusion": None}
    job = {"id": 70, "run_id": 7, "run_attempt": 1, "head_sha": SHA,
           "name": "cuda", "status": "in_progress", "conclusion": None,
           "runner_name": "cuda-windows", "runner_id": 2313}
    return {"runners": {"org": [listener("cuda-windows", 2313, True),
                                listener("cuda-windows-2", 2619, False)],
                        "inference": [],
                        "app": [listener("cuda-windows-3", 23, False),
                                listener("cuda-windows-4", 24, False)]},
            "runs": {("SceneWorks/inference", 7): own},
            "jobs": {("SceneWorks/inference", 7): [job]}}


def reviewed_gpu1_snapshot():
    data = own_snapshot()
    data["runners"]["org"][0]["busy"] = True
    data["runners"]["org"][1]["busy"] = True
    own_job = data["jobs"][("SceneWorks/inference", 7)][0]
    own_job["runner_name"], own_job["runner_id"] = "cuda-windows-2", 2619
    run = {"id": gpu1.RUN, "head_sha": gpu1.HEAD, "run_attempt": 1,
           "event": "workflow_dispatch", "path": ".github/workflows/real-weights.yml",
           "repository": {"full_name": "SceneWorks/inference"}, "created_at": gpu1.CREATED,
           "name": "Real-weight validation", "status": "in_progress", "conclusion": None}
    selected = {"id": gpu1.JOB, "run_id": gpu1.RUN, "run_attempt": 1,
                "head_sha": gpu1.HEAD, "name": gpu1.NAME,
                "workflow_name": "Real-weight validation", "runner_name": gpu1.RUNNER,
                "runner_id": gpu1.RUNNER_ID, "started_at": gpu1.STARTED,
                "status": "in_progress", "conclusion": None, "completed_at": None}
    jobs = [selected] + [{"id": 1000 + i, "status": "completed", "conclusion": "skipped"}
                         for i in range(54)]
    data["runs"][("SceneWorks/inference", gpu1.RUN)] = run.copy()
    data["jobs"][("SceneWorks/inference", gpu1.RUN)] = [row.copy() for row in jobs]
    data["reviewed_gpu1"] = {"run": run, "job": selected.copy(),
                             "jobs": {"total_count": 55, "jobs": jobs},
                             "group": {"group_name": gpu1.GROUP, "total_count": 1,
                                       "group_members": [{"run_id": gpu1.RUN,
                                                          "status": "in_progress"}]}}
    return data


def transition_fixture():
    data = own_snapshot()
    data["checked_at"] = "2026-10-04T20:02:00Z"
    own = data["runs"][("SceneWorks/inference", 7)]
    own["created_at"] = "2026-10-04T20:00:00Z"
    direct = {**own, "repository": {"full_name": "SceneWorks/inference"}}
    job = {**data["jobs"][("SceneWorks/inference", 7)][0],
           "started_at": "2026-10-04T20:01:00Z"}
    data["jobs"][("SceneWorks/inference", 7)][0] = job
    binding = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
    return data, direct, binding


class SharedHostWatchTests(unittest.TestCase):
    def test_own_transition_is_only_retryable_after_foreign_inventory_validation(self):
        data, _, binding = transition_fixture()
        own_key = ("SceneWorks/inference", 7)
        del data["runs"][own_key]
        del data["jobs"][own_key]
        with self.assertRaises(watch.OwnedInventoryTransition):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=binding)

        for status in ("queued", "pending", "requested", "waiting"):
            queued, _, queued_binding = transition_fixture()
            queued["runs"][own_key]["status"] = status
            with self.subTest(status=status), self.assertRaises(watch.OwnedInventoryTransition):
                watch.classify(queued, 7, SHA, "yue2-precision-proof.yml",
                               mode="shared-gpu1", owned_binding=queued_binding)

        queued["runs"][own_key]["head_sha"] = "b" * 40
        with self.assertRaisesRegex(RuntimeError, "owned run/source/attempt changed"):
            watch.classify(queued, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=queued_binding)

        created, _, created_binding = transition_fixture()
        created["runs"][own_key]["created_at"] = "2026-10-04T20:00:01Z"
        with self.assertRaisesRegex(RuntimeError, "creation identity changed"):
            watch.classify(created, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=created_binding)
        terminal, _, terminal_binding = transition_fixture()
        terminal["runs"][own_key].update(status="completed", conclusion="success")
        with self.assertRaisesRegex(RuntimeError, "owned run/source/attempt/status changed"):
            watch.classify(terminal, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=terminal_binding)

        foreign, _, foreign_binding = transition_fixture()
        del foreign["runs"][own_key]
        del foreign["jobs"][own_key]
        foreign_key = ("SceneWorks/SceneWorks", 99)
        foreign["runs"][foreign_key] = {"id": 99, "status": "queued"}
        foreign["jobs"][foreign_key] = []
        with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
            watch.classify(foreign, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-host", owned_binding=foreign_binding)

        partial, _, partial_binding = transition_fixture()
        del partial["runs"][own_key]
        del partial["jobs"][own_key]
        partial["runners"]["app"][0]["id"] = 999
        with self.assertRaisesRegex(RuntimeError, "physical CUDA runner identity"):
            watch.classify(partial, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=partial_binding)

        active, _, active_binding = transition_fixture()
        active["runs"][own_key]["created_at"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "creation identity changed"):
            watch.classify(active, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", owned_binding=active_binding)

    def test_transition_retry_retains_sanitized_inventory_and_requires_exact_reauth(self):
        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            stale["privateToken"] = "do-not-retain"
            fresh = own_snapshot()
            fresh["checked_at"] = "2026-10-04T20:03:00Z"
            fresh["runs"][("SceneWorks/inference", 7)]["created_at"] = direct["created_at"]
            fresh["jobs"][("SceneWorks/inference", 7)][0]["started_at"] = binding["start"]
            terminal = {**direct, "status": "completed", "conclusion": "success"}
            output = Path(directory) / "watch"
            with patch.object(watch, "bind_owned_job", side_effect=[binding, binding]) as auth, \
                 patch.object(watch, "owned_run", side_effect=[direct, terminal]), \
                 patch.object(watch, "snapshot", side_effect=[stale, fresh]) as snapshots, \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                            70, "cuda-windows", 2313, mode="shared-gpu1")
            self.assertEqual((auth.call_count, snapshots.call_count), (2, 2))
            cancel.assert_not_called()
            retained = json.loads((output / "own-transition-0001-1.json").read_text(encoding="utf-8"))
            self.assertNotIn("privateToken", retained["inventory"])
            self.assertEqual(retained["inventory"]["runners"]["org"][0]["id"], 2313)
            self.assertTrue((output / "0001.json").is_file())

    def test_transition_retry_does_not_retry_unknown_foreign_or_cancel_terminal_run(self):
        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            foreign_key = ("SceneWorks/SceneWorks", 99)
            stale["runs"][foreign_key] = {"id": 99, "status": "queued"}
            stale["jobs"][foreign_key] = []
            with patch.object(watch, "bind_owned_job", return_value=binding) as auth, \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=stale) as snapshots, \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "bad",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-host")
            self.assertEqual((auth.call_count, snapshots.call_count), (1, 1))
            cancel.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            terminal = {**direct, "status": "completed", "conclusion": "success"}
            with patch.object(watch, "bind_owned_job", side_effect=[
                    binding, watch.OwnedBindingUnavailable("job ended")]), \
                 patch.object(watch, "owned_run", side_effect=[direct, terminal]), \
                 patch.object(watch, "snapshot", return_value=stale), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "terminal",
                            60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_not_called()
            self.assertTrue((Path(directory) / "terminal" / "terminal.json").is_file())

    def test_status_only_binding_refusal_keeps_cached_cancel_and_exact_job_terminal_ends_watch(self):
        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            queued_job = {**binding["job"], "status": "queued", "completed_at": None}
            with patch.object(watch, "bind_owned_job", side_effect=[
                    binding, watch.OwnedBindingUnavailable("owned job/runner/start binding unavailable")]), \
                 patch.object(watch, "owned_run", side_effect=[direct, direct, direct]), \
                 patch.object(watch, "snapshot", return_value=stale), \
                 patch.object(watch, "api", return_value=queued_job), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(watch.OwnedBindingUnavailable, "binding unavailable"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "queued",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_called_once_with(7, SHA, "yue2-precision-proof.yml", binding,
                                           identity_drift=False)

        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            completed_job = {**binding["job"], "status": "completed", "conclusion": "success",
                             "completed_at": "2026-10-04T20:05:00Z"}
            with patch.object(watch, "bind_owned_job", side_effect=[
                    binding, watch.OwnedBindingUnavailable("owned job/runner/start binding unavailable")]), \
                 patch.object(watch, "owned_run", side_effect=[direct, direct, direct]), \
                 patch.object(watch, "snapshot", return_value=stale), \
                 patch.object(watch, "api", return_value=completed_job), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "job-done",
                            60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_not_called()
            receipt = json.loads((Path(directory) / "job-done" / "job-terminal.json").read_text(encoding="utf-8"))
            self.assertIn("whole run may still be active", receipt["scope"])

    def test_transition_retry_is_bounded_and_job_drift_revokes_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            with patch.object(watch, "bind_owned_job", return_value=binding) as auth, \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=stale) as snapshots, \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "unresolved after bounded retries"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "retry",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            self.assertEqual((auth.call_count, snapshots.call_count), (3, 3))
            cancel.assert_called_once_with(7, SHA, "yue2-precision-proof.yml", binding,
                                           identity_drift=False)
            self.assertEqual(len(list((Path(directory) / "retry").glob("own-transition-*.json"))), 3)

        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            changed = {**binding, "start": "different",
                       "job": {**binding["job"], "started_at": "different"}}
            with patch.object(watch, "bind_owned_job", side_effect=[binding, changed]), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=stale), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "owned identity drifted"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "drift",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_called_once()
            self.assertTrue(cancel.call_args.kwargs["identity_drift"])

    def test_aggregate_creation_drift_revokes_cached_cancel_when_direct_api_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            data, direct, binding = transition_fixture()
            data["runs"][("SceneWorks/inference", 7)]["created_at"] = "changed"
            with patch.object(watch, "bind_owned_job", return_value=binding), \
                 patch.object(watch, "owned_run", side_effect=[
                     direct, subprocess.TimeoutExpired(["gh", "api"], 45)]), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "creation identity changed"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "drift",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_called_once_with(7, SHA, "yue2-precision-proof.yml", binding,
                                           identity_drift=True)

    def test_direct_immutable_job_drift_revokes_cached_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            stale, direct, binding = transition_fixture()
            del stale["runs"][("SceneWorks/inference", 7)]
            del stale["jobs"][("SceneWorks/inference", 7)]
            with patch.object(watch, "bind_owned_job", side_effect=[
                    binding, watch.OwnedIdentityDrift("owned job immutable identity changed")]), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=stale), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(watch.OwnedIdentityDrift, "immutable identity changed"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "job-drift",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_called_once_with(7, SHA, "yue2-precision-proof.yml", binding,
                                           identity_drift=True)

    def test_moving_paginated_status_is_rejected_then_fresh_snapshot_succeeds(self):
        moving = [{"total_count": 2, "workflow_runs": [{"id": 7}]}]
        with self.assertRaisesRegex(watch.InventorySnapshotError,
                                    "truncated paginated inventory") as error:
            watch.complete_pages(moving, "workflow_runs", "SceneWorks/inference:queued")
        self.assertEqual(error.exception.pages, moving)
        self.assertEqual(error.exception.source, "SceneWorks/inference:queued")
        duplicated = [{"total_count": 2, "workflow_runs": [{"id": 7}, {"id": 7}]}]
        with self.assertRaisesRegex(watch.InventorySnapshotError, "ambiguous paginated inventory"):
            watch.complete_pages(duplicated, "workflow_runs", "SceneWorks/inference:queued")
        data = own_snapshot()
        data["checked_at"] = "2026-10-04T20:02:00Z"
        data["runs"][("SceneWorks/inference", 7)]["created_at"] = "2026-10-04T20:00:00Z"
        direct = {**data["runs"][("SceneWorks/inference", 7)],
                  "repository": {"full_name": "SceneWorks/inference"},
                  "created_at": "2026-10-04T20:00:00Z"}
        job = data["jobs"][("SceneWorks/inference", 7)][0]
        job["started_at"] = "2026-10-04T20:01:00Z"
        bound = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
        terminal = {**direct, "status": "completed", "conclusion": "success"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "watch"
            with patch.object(watch, "bind_owned_job", return_value=bound) as auth, \
                 patch.object(watch, "owned_run", side_effect=[direct, terminal]), \
                 patch.object(watch, "snapshot", side_effect=[error.exception, data]) as snapshots, \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                            70, "cuda-windows", 2313, mode="shared-gpu1")
            self.assertEqual(snapshots.call_count, 2)
            self.assertEqual(auth.call_count, 2)
            cancel.assert_not_called()
            retained = json.loads((output / "inventory-attempt-0001-1.json").read_text(encoding="utf-8"))
            self.assertEqual(retained["pages"], moving)
            self.assertTrue((output / "0001.json").is_file())

    def test_shared_gpu1_retries_transport_but_persistent_incomplete_inventory_refuses(self):
        data = own_snapshot()
        data["checked_at"] = "2026-10-04T20:02:00Z"
        direct = {**data["runs"][("SceneWorks/inference", 7)],
                  "repository": {"full_name": "SceneWorks/inference"},
                  "created_at": "2026-10-04T20:00:00Z"}
        job = data["jobs"][("SceneWorks/inference", 7)][0]
        job["started_at"] = "2026-10-04T20:01:00Z"
        bound = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
        transport = subprocess.CalledProcessError(1, ["gh", "api"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "watch"
            with patch.object(watch, "bind_owned_job", return_value=bound) as auth, \
                 patch.object(watch, "owned_run", side_effect=[direct, {**direct, "status": "completed"}]), \
                 patch.object(watch, "snapshot", side_effect=[transport, data]) as snapshots, \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                            70, "cuda-windows", 2313, mode="shared-gpu1")
            self.assertEqual((auth.call_count, snapshots.call_count), (2, 2))
            cancel.assert_not_called()
        incomplete = watch.InventorySnapshotError("truncated paginated inventory", "queued", [])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(watch, "bind_owned_job", return_value=bound) as auth, \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", side_effect=incomplete) as snapshots, \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "truncated paginated inventory"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            self.assertEqual((auth.call_count, snapshots.call_count), (3, 3))
            cancel.assert_called_once()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(watch, "bind_owned_job", return_value=bound) as auth, \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", side_effect=incomplete) as snapshots, \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "truncated paginated inventory"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-host")
            auth.assert_called_once()
            snapshots.assert_called_once()
            cancel.assert_called_once()

    def test_inventory_retry_refuses_owned_identity_drift_without_cancellation(self):
        data = own_snapshot()
        direct = {**data["runs"][("SceneWorks/inference", 7)],
                  "repository": {"full_name": "SceneWorks/inference"},
                  "created_at": "2026-10-04T20:00:00Z"}
        job = data["jobs"][("SceneWorks/inference", 7)][0]
        job["started_at"] = "2026-10-04T20:01:00Z"
        bound = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
        moved = {**bound, "start": "changed", "job": {**job, "started_at": "changed"}}
        incomplete = watch.InventorySnapshotError("truncated paginated inventory", "queued", [])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(watch, "bind_owned_job", side_effect=[bound, moved]), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", side_effect=incomplete) as snapshots, \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "owned identity drifted"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            snapshots.assert_called_once()
            self.assertTrue(cancel.call_args.kwargs["identity_drift"])

    def test_inventory_retry_auth_transport_loss_cancels_only_cached_owned_run(self):
        data = own_snapshot()
        direct = {**data["runs"][("SceneWorks/inference", 7)],
                  "repository": {"full_name": "SceneWorks/inference"},
                  "created_at": "2026-10-04T20:00:00Z"}
        job = {**data["jobs"][("SceneWorks/inference", 7)][0],
               "started_at": "2026-10-04T20:01:00Z"}
        bound = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
        incomplete = watch.InventorySnapshotError("truncated paginated inventory", "queued", [])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "watch"
            with patch.object(watch, "bind_owned_job", side_effect=[
                    bound, subprocess.TimeoutExpired(["gh", "api"], 45)]), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", side_effect=incomplete), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                with self.assertRaises(subprocess.TimeoutExpired):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", output,
                                60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_called_once_with(7, SHA, "yue2-precision-proof.yml", bound,
                                           identity_drift=False)
            self.assertIn("inventory-attempt-0001-1.json", {p.name for p in output.iterdir()})

    def test_shared_gpu1_observes_foreign_jobs_without_revoking_owned_run(self):
        data = own_snapshot()
        data["runners"]["org"][1]["busy"] = True
        data["runners"]["app"][0]["busy"] = True
        data["runners"]["app"][1]["status"] = "offline"
        key = ("SceneWorks/SceneWorks", 99)
        data["runs"][key] = {"id": 99, "status": "queued", "head_sha": "b" * 40}
        data["jobs"][key] = []
        proof = watch.classify(data, 7, SHA, "yue2-precision-proof.yml", mode="shared-gpu1")
        self.assertEqual(proof["own_job"], 70)
        self.assertEqual(proof["foreign_runs_observed"],
                         [{"repository": key[0], "run_id": 99, "status": "queued", "job_count": 0}])
        with self.assertRaisesRegex(RuntimeError, "unaccounted busy"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml", mode="shared-host")
        data["runs"][("SceneWorks/inference", 7)]["head_sha"] = "c" * 40
        with self.assertRaisesRegex(RuntimeError, "owned run/source"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml", mode="shared-gpu1")

    def test_shared_gpu1_requires_exact_owned_runner_and_all_four_listener_identities(self):
        data = own_snapshot()
        data["jobs"][("SceneWorks/inference", 7)][0]["name"] = "cuda_diagnostic"
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                                        mode="shared-gpu1", own_job_name="cuda_diagnostic")["own_job"], 70)
        data["runners"]["org"][0]["busy"] = False
        with self.assertRaisesRegex(RuntimeError, "owned CUDA runner is not assigned"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", own_job_name="cuda_diagnostic")
        data["runners"]["org"][0]["busy"] = True
        data["runners"]["app"][0]["id"] = 999
        with self.assertRaisesRegex(RuntimeError, "physical CUDA runner identity"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-gpu1", own_job_name="cuda_diagnostic")

    def test_shared_gpu1_watch_keeps_owned_run_when_foreign_cuda_job_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            data = own_snapshot()
            data["runners"]["org"][1]["busy"] = True
            key = ("SceneWorks/inference", 99)
            data["runs"][key] = {"id": 99, "status": "in_progress", "head_sha": "b" * 40}
            data["jobs"][key] = [{"id": 990, "name": "foreign CUDA", "status": "in_progress",
                                  "runner_name": "cuda-windows-2", "runner_id": 2619}]
            direct = {**data["runs"][("SceneWorks/inference", 7)],
                      "repository": {"full_name": "SceneWorks/inference"},
                      "created_at": "2026-10-04T20:00:00Z"}
            job = data["jobs"][("SceneWorks/inference", 7)][0]
            job["started_at"] = "2026-10-04T20:01:00Z"
            binding = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
            terminal = {**direct, "status": "completed", "conclusion": "success"}
            with patch.object(watch, "bind_owned_job", return_value=binding), \
                 patch.object(watch, "owned_run", side_effect=[direct, terminal]), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(watch.time, "sleep"), \
                 patch.object(watch, "cancel_bound_run") as cancel:
                watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                            60, 30, 70, "cuda-windows", 2313, mode="shared-gpu1")
            cancel.assert_not_called()

    def test_unused_app_listeners_may_be_offline_but_owned_runner_must_be_online(self):
        data = own_snapshot()
        for row in data["runners"]["app"]:
            row["status"] = "offline"
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml")["own_job"], 70)
        data["runners"]["app"][0]["busy"] = True
        with self.assertRaisesRegex(RuntimeError, "unaccounted busy"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
        data["runners"]["app"][0]["busy"] = False
        data["runners"]["org"][0]["status"] = "offline"
        with self.assertRaisesRegex(RuntimeError, "owned CUDA runner is offline"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")

    def test_reviewed_gpu1_exact_job_and_clean_completion_transition(self):
        data = reviewed_gpu1_snapshot()
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                                        mode="gpu0-with-reviewed-gpu1")["reviewed_gpu1"], "active")
        for field, bad in (("head_sha", "b" * 40), ("runner_name", "cuda-windows-2"),
                           ("runner_id", 2619), ("started_at", "changed"),
                           ("status", "completed")):
            broken = reviewed_gpu1_snapshot()
            broken["reviewed_gpu1"]["job"][field] = bad
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                watch.classify(broken, 7, SHA, "yue2-precision-proof.yml",
                               mode="gpu0-with-reviewed-gpu1")
        data["runners"]["org"][0]["busy"] = False
        data["reviewed_gpu1"]["job"].update(status="completed", conclusion="success",
                                             completed_at="2026-10-04T15:00:00Z")
        data["reviewed_gpu1"]["jobs"]["jobs"][0] = data["reviewed_gpu1"]["job"].copy()
        data["jobs"][("SceneWorks/inference", gpu1.RUN)][0] = data["reviewed_gpu1"]["job"].copy()
        del data["reviewed_gpu1"]["group"]
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                                        mode="gpu0-with-reviewed-gpu1")["reviewed_gpu1"], "completed")
        # A newly queued reservation remains unknown after GPU1 is free.
        data["runs"][("SceneWorks/inference", 999)] = {"id": 999, "status": "queued"}
        data["jobs"][("SceneWorks/inference", 999)] = []
        with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="gpu0-with-reviewed-gpu1")

    def test_reviewed_gpu1_source_and_group_mutations_refuse(self):
        data = reviewed_gpu1_snapshot()
        data["reviewed_gpu1"]["group"]["group_members"][0]["run_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "old-group reservation"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="gpu0-with-reviewed-gpu1")

    def test_reviewed_gpu1_read_only_diagnostic_uses_same_other_runner(self):
        data = reviewed_gpu1_snapshot()
        data["jobs"][("SceneWorks/inference", 7)][0]["name"] = "cuda_diagnostic"
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                                        mode="gpu0-with-reviewed-gpu1",
                                        own_job_name="cuda_diagnostic")["own_runner"],
                         "cuda-windows-2")
        with self.assertRaisesRegex(RuntimeError, "unreviewed owned GPU job name"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="shared-host", own_job_name="cuda_diagnostic")
        data = reviewed_gpu1_snapshot()
        data["runners"]["app"][0]["busy"] = True
        with self.assertRaisesRegex(RuntimeError, "unaccounted busy"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml",
                           mode="gpu0-with-reviewed-gpu1")

    def test_reviewed_gpu1_watch_cancels_only_owned_on_new_unknown_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            data = reviewed_gpu1_snapshot()
            data["runs"][("SceneWorks/SceneWorks", 999)] = {"id": 999, "status": "queued"}
            data["jobs"][("SceneWorks/SceneWorks", 999)] = []
            own = data["runs"][("SceneWorks/inference", 7)]
            own["created_at"] = "2026-10-04T14:00:00Z"
            direct = {**own, "repository": {"full_name": "SceneWorks/inference"},
                      "created_at": own["created_at"]}
            job = data["jobs"][("SceneWorks/inference", 7)][0]
            job["started_at"] = "2026-10-04T14:01:00Z"
            binding = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
            with patch.object(watch, "bind_owned_job", return_value=binding), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "api", return_value=direct), \
                 patch.object(gpu1, "source"), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(watch.subprocess, "run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                                60, 30, 70, "cuda-windows-2", 2619,
                                mode="gpu0-with-reviewed-gpu1")
            cancel.assert_called_once()
            self.assertEqual(cancel.call_args.args[0],
                             ["gh", "run", "cancel", "7", "-R", "SceneWorks/inference"])
    def test_only_exact_owned_job_and_four_known_idle_other_runners_pass(self):
        data = own_snapshot()
        self.assertEqual(watch.classify(data, 7, SHA, "yue2-precision-proof.yml")["own_job"], 70)
        foreign = {"id": 8, "status": "queued", "head_sha": "b" * 40}
        data["runs"][("SceneWorks/inference", 8)] = foreign
        data["jobs"][("SceneWorks/inference", 8)] = []
        with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
        del data["runs"][("SceneWorks/inference", 8)]
        del data["jobs"][("SceneWorks/inference", 8)]
        data["runners"]["app"][0]["busy"] = True
        with self.assertRaisesRegex(RuntimeError, "unaccounted busy"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
        data["runners"]["app"][0]["busy"] = False
        data["runners"]["app"].append({"name": "cuda-windows-5", "id": 25,
                                          "status": "online", "busy": False,
                                          "labels": [{"name": "self-hosted"}]})
        with self.assertRaisesRegex(RuntimeError, "listener set changed"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")

    def test_source_qualified_allocated_cpu_job_may_coexist_but_unknown_cannot(self):
        data = own_snapshot()
        key = ("SceneWorks/SceneWorks", 9)
        data["runs"][key] = {"id": 9, "status": "in_progress"}
        data["jobs"][key] = [{"id": 90, "status": "in_progress", "labels": ["ubuntu-latest"],
                              "runner_name": "GitHub Actions 2", "runner_id": 100}]
        watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
        data["jobs"][key][0]["labels"] = ["cuda", "self-hosted"]
        with self.assertRaisesRegex(RuntimeError, "foreign CUDA or unknown"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
        data["jobs"][key][0]["labels"] = ["ubuntu-latest"]
        data["jobs"][key][0]["runner_name"] = "cuda-windows-3"
        with self.assertRaisesRegex(RuntimeError, "foreign CUDA or unknown"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")

    def test_exact_two_historical_zero_job_exceptions_only(self):
        data = own_snapshot()
        data["legacy_direct"] = {}
        for rid, (sha, attempt, created, updated) in watch.LEGACY_ZERO_JOB.items():
            row = {
                "id": rid, "head_sha": sha, "run_attempt": attempt,
                "created_at": created, "updated_at": updated, "status": "queued",
                "conclusion": None, "event": "pull_request", "path": ".github/workflows/ci.yml"}
            data["runs"][("SceneWorks/inference", rid)] = row
            data["legacy_direct"][("SceneWorks/inference", rid)] = {
                **row, "repository": {"full_name": "SceneWorks/inference"}}
            data["jobs"][("SceneWorks/inference", rid)] = []
        self.assertEqual(set(watch.classify(data, 7, SHA, "yue2-precision-proof.yml")
                             ["historical_zero_job_runs"]), set(watch.LEGACY_ZERO_JOB))
        first = next(iter(watch.LEGACY_ZERO_JOB))
        data["runs"][("SceneWorks/inference", first)]["updated_at"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "direct run readback"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")

    def test_nonterminal_run_with_only_completed_jobs_is_unknown(self):
        data = own_snapshot()
        key = ("SceneWorks/SceneWorks", 9)
        data["runs"][key] = {"id": 9, "status": "in_progress"}
        data["jobs"][key] = [{"id": 90, "status": "completed", "conclusion": "success"}]
        with self.assertRaisesRegex(RuntimeError, "no active assigned"):
            watch.classify(data, 7, SHA, "yue2-precision-proof.yml")

    def test_watch_cancels_only_authenticated_owned_run_on_foreign_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-watch"
            data = own_snapshot()
            data["runs"][("SceneWorks/SceneWorks", 9)] = {"id": 9, "status": "queued"}
            data["privateToken"] = "must-not-be-retained"
            direct = {**data["runs"][("SceneWorks/inference", 7)],
                      "repository": {"full_name": "SceneWorks/inference"}}
            bound = {"run": direct, "job": data["jobs"][("SceneWorks/inference", 7)][0],
                     "job_id": 70, "start": "2026-10-03T00:00:00Z"}
            bound["job"]["started_at"] = bound["start"]
            with patch.object(watch, "bind_owned_job", return_value=bound), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "api", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(watch.subprocess, "run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "foreign run has no allocated jobs"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                                70, "cuda-windows", 2313)
            cancel.assert_called_once()
            self.assertEqual(cancel.call_args.args[0],
                             ["gh", "run", "cancel", "7", "-R", "SceneWorks/inference"])
            self.assertTrue((output / "refusal.txt").is_file())
            retained = output / "inventory-refusal-0001.json"
            self.assertTrue(retained.is_file())
            self.assertNotIn("must-not-be-retained", retained.read_text(encoding="utf-8"))

    def test_failed_initial_binding_never_cancels_and_refusal_write_failure_still_cancels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "watch"
            with patch.object(watch, "bind_owned_job", side_effect=RuntimeError("initial auth failed")), \
                 patch.object(watch.subprocess, "run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "initial auth failed"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                                70, "cuda-windows", 2313)
                cancel.assert_not_called()
            data = own_snapshot()
            data["runs"][("SceneWorks/SceneWorks", 9)] = {"id": 9, "status": "queued"}
            direct = {**data["runs"][("SceneWorks/inference", 7)],
                      "repository": {"full_name": "SceneWorks/inference"}}
            job = data["jobs"][("SceneWorks/inference", 7)][0]
            job["started_at"] = "2026-10-03T00:00:00Z"
            bound = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
            original_write = Path.write_text
            def write(path, *args, **kwargs):
                if path.name == "refusal.txt":
                    raise OSError("disk failed")
                return original_write(path, *args, **kwargs)
            with patch.object(watch, "bind_owned_job", return_value=bound), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "api", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(Path, "write_text", write), \
                 patch.object(watch.subprocess, "run") as cancel:
                with self.assertRaisesRegex(OSError, "disk failed"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", output, 60, 30,
                                70, "cuda-windows", 2313)
                cancel.assert_called_once()

    def test_positive_owned_job_drift_revokes_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = own_snapshot()
            direct = {**data["runs"][("SceneWorks/inference", 7)],
                      "repository": {"full_name": "SceneWorks/inference"}}
            job = data["jobs"][("SceneWorks/inference", 7)][0]
            job["started_at"] = "2026-10-03T00:00:00Z"
            bound = {"run": direct, "job": job.copy(), "job_id": 70, "start": job["started_at"]}
            job["runner_name"] = "cuda-windows-2"
            with patch.object(watch, "bind_owned_job", return_value=bound), \
                 patch.object(watch, "owned_run", return_value=direct), \
                 patch.object(watch, "snapshot", return_value=data), \
                 patch.object(watch.subprocess, "run") as cancel:
                with self.assertRaisesRegex(RuntimeError, "owned job identity drifted"):
                    watch.watch(7, SHA, "yue2-precision-proof.yml", Path(directory) / "watch",
                                60, 30, 70, "cuda-windows", 2313)
                cancel.assert_not_called()

    def test_cancel_uses_only_cached_binding_on_api_loss_and_revokes_direct_drift(self):
        data = own_snapshot()
        direct = {**data["runs"][("SceneWorks/inference", 7)],
                  "created_at": "2026-10-03T00:00:00Z",
                  "repository": {"full_name": "SceneWorks/inference"}}
        job = {**data["jobs"][("SceneWorks/inference", 7)][0],
               "started_at": "2026-10-03T00:01:00Z"}
        binding = {"run": direct, "job": job, "job_id": 70, "start": job["started_at"]}
        with patch.object(watch, "api", side_effect=OSError("API unavailable")), \
             patch.object(watch.subprocess, "run") as cancel:
            watch.cancel_bound_run(7, SHA, "yue2-precision-proof.yml", binding,
                                   identity_drift=False)
            cancel.assert_called_once()
        with patch.object(watch, "api", return_value={**direct, "head_sha": "b" * 40}), \
             patch.object(watch.subprocess, "run") as cancel:
            watch.cancel_bound_run(7, SHA, "yue2-precision-proof.yml", binding,
                                   identity_drift=False)
            cancel.assert_not_called()
