"""Fail-closed external shared-host inventory watcher regressions."""
from pathlib import Path
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


class SharedHostWatchTests(unittest.TestCase):
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
            direct = {**own, "repository": {"full_name": "SceneWorks/inference"},
                      "created_at": "2026-10-04T14:00:00Z"}
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
