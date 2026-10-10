"""Exact source and job identity for the one reviewed GPU1 coexistence run."""

import base64
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import yue2_reviewed_gpu1 as gpu1


def run():
    return {"id": gpu1.RUN, "head_sha": gpu1.HEAD, "run_attempt": 1,
            "event": "workflow_dispatch", "path": ".github/workflows/real-weights.yml",
            "repository": {"full_name": "SceneWorks/inference"}, "created_at": gpu1.CREATED,
            "name": "Real-weight validation", "status": "in_progress", "conclusion": None}


def job():
    return {"id": gpu1.JOB, "run_id": gpu1.RUN, "run_attempt": 1,
            "head_sha": gpu1.HEAD, "name": gpu1.NAME,
            "workflow_name": "Real-weight validation", "runner_name": gpu1.RUNNER,
            "runner_id": gpu1.RUNNER_ID, "started_at": gpu1.STARTED,
            "status": "in_progress", "conclusion": None, "completed_at": None}


class ReviewedGpu1Tests(unittest.TestCase):
    def test_exact_active_and_completed_run_job(self):
        gpu1.run(run(), active=True)
        gpu1.job(job(), active=True)
        for key, bad in (("id", 1), ("head_sha", "b" * 40), ("run_attempt", 2),
                         ("created_at", "changed"), ("path", "other.yml"),
                         ("repository", {"full_name": "other/repo"})):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                gpu1.run({**run(), key: bad}, active=True)
        for key, bad in (("id", 1), ("head_sha", "b" * 40), ("runner_name", "cuda-windows-2"),
                         ("runner_id", 2619), ("started_at", "changed")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                gpu1.job({**job(), key: bad}, active=True)
        done_job = {**job(), "status": "completed", "conclusion": "success",
                    "completed_at": "2026-10-04T15:00:00Z"}
        gpu1.job(done_job, active=False)
        done_run = {**run(), "status": "completed", "conclusion": "success",
                    "updated_at": "2026-10-04T15:00:10Z"}
        gpu1.run(done_run, active=False)
        with self.assertRaises(RuntimeError):
            gpu1.job(done_job, active=True)

    def test_source_bytes_are_checked_not_names(self):
        path = next(iter(gpu1.SOURCES))
        raw = b"exact frozen workflow\n"
        with patch.dict(gpu1.SOURCES, {path: hashlib.sha256(raw).hexdigest()}):
            payload = {"path": path, "encoding": "base64",
                       "content": base64.encodebytes(raw).decode("ascii")}
            gpu1.source(payload, path)
            for broken in ({**payload, "path": "other"},
                           {**payload, "content": base64.b64encode(b"changed").decode("ascii")},
                           {**payload, "encoding": "utf-8"}):
                with self.assertRaises(RuntimeError):
                    gpu1.source(broken, path)


if __name__ == "__main__":
    unittest.main()
