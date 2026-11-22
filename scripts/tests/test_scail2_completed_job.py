"""Model-free completed Windows Job evidence and its GPU 0 continuation."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts.ci import scail2_plural_fit as runner
from scripts.ci import scail2_recover_completed_job as recovery
from scripts.ci.scail_plural import contract, cuda_stage
from scripts.tests.test_scail2_cuda_stage import fixture_with_rejected_growth


def completed_fixture():
    guard, pool = fixture_with_rejected_growth()
    guard["returncode"] = 0
    guard["admission"]["pollMillis"] = 250
    raw = guard["fit_observation"]
    card = raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    card.pop()
    end = raw["cuda_render_end"]
    end["zero"] = None
    tail = end["rejected_growth"]
    tail.extend((dict(tail[-1], read_ns=345, at_ns=355), dict(tail[-1], read_ns=375, at_ns=390)))
    guard["cuda_attribution"]["unknown_pid_polls"] += 2
    for series, name in ((guard["cuda_attribution"]["wddm_dedicated"], "dedicated"), (raw["wddm_shared"], "shared")):
        series["samples"] = [[i, 10] for i in range(100, 104)] + series["samples"]
        gap = raw["wddm_retry"][name]["events"][0]
        if name == "dedicated":
            gap.update(end_ns=380, local_filetime_after=181)
        else:
            gap.update(start_ns=380, local_filetime_before=181)
    claim = dict(release_metadata_verification={"status": "verified"}, release_background_activity={
        "samples": [{"selected_card": {"used_mib": 0, "utilization_percent": 0}} for _ in range(5)]
    })
    return guard, pool, claim


def scope(guard, pool, claim):
    uuid = pool["cudaDeviceUuid"]
    _, primary = cuda_stage.tenant_growth(guard, uuid)
    return cuda_stage.render_scope(guard, pool, uuid, primary, completed_job=(claim, "owned-job-completion"))


class CompletedJob(unittest.TestCase):
    def test_real_order_breaks_clock_ties_without_synthesizing_a_zero(self):
        guard, pool, claim = completed_fixture()
        before = copy.deepcopy(guard)
        result = scope(guard, pool, claim)
        self.assertEqual(result["terminal_rejected_attempts"], 4)
        self.assertEqual(result["shared_missing_after_output"], 1)
        self.assertEqual(result["dedicated_missing_after_output"], 1)
        self.assertNotIn("zero_card_index", result)
        self.assertEqual(guard, before)
        with self.assertRaisesRegex(ValueError, "without a verified owned zero"):
            cuda_stage.render_scope(guard, pool, pool["cudaDeviceUuid"], 1)

    def test_unlocated_gaps_missing_polls_and_failed_guards_are_ineligible(self):
        for problem in ("before-read", "overlap", "filetime", "unrecorded", "missing-attempt",
                        "crossed-job", "cap", "reserve", "reused-card", "release", "return",
                        "shared-backwards", "both-backwards", "noninteger", "late-clock"):
            with self.subTest(problem=problem):
                guard, pool, claim = completed_fixture()
                raw = guard["fit_observation"]
                attempts = raw["cuda_render_end"]["rejected_growth"]
                gap = raw["wddm_retry"]["dedicated"]["events"][0]
                shared = raw["wddm_retry"]["shared"]["events"][0]
                if problem == "before-read":
                    gap["start_ns"] -= 1
                elif problem == "overlap":
                    shared["start_ns"] -= 1
                elif problem == "filetime":
                    shared["local_filetime_before"] -= 1
                elif problem == "shared-backwards":
                    shared["local_filetime_after"] = shared["local_filetime_before"] - 1
                elif problem == "both-backwards":
                    gap["local_filetime_after"] = shared["local_filetime_after"] = 0
                elif problem == "noninteger":
                    shared["end_ns"] = float(shared["end_ns"])
                elif problem == "late-clock":
                    shared["local_filetime_after"] += 2_500_001
                elif problem == "unrecorded":
                    raw["wddm_shared"]["samples"].pop(0)
                elif problem == "missing-attempt":
                    attempts.pop()
                elif problem == "crossed-job":
                    attempts[1]["created_filetime"] += 1
                elif problem == "cap":
                    guard["admission"]["childGpuCapBytes"] = 77
                elif problem == "reserve":
                    attempts[-1]["free_bytes"] = 49
                elif problem == "reused-card":
                    claim["release_background_activity"]["samples"][4]["selected_card"]["used_mib"] = 1
                elif problem == "release":
                    claim["release_metadata_verification"]["status"] = "inconclusive-driver-query-timing"
                elif problem == "return":
                    guard["returncode"] = 1
                with self.assertRaises(ValueError):
                    scope(guard, pool, claim)

    def test_only_exact_gpu0_continuation_adds_the_archived_pointer(self):
        with patch.object(runner.proof, "repository_identity", return_value={"revision": "r", "tree": "t"}):
            full = dict(selection=contract.SELECTION, policy=contract.policy("candle-cuda", cuda_uuid=contract.GPU0))
            first = runner.plan("candle-cuda", full, "r")
            subset = dict(full, continuation=contract.CONTINUATION)
            follow = runner.plan("candle-cuda", subset, "r")
            self.assertEqual(first["counts"], [6, 2, 3, 4, 5])
            self.assertNotIn("recovered_six", first)
            self.assertEqual(first["cuda_completion"], "supervisor-owned-windows-job")
            self.assertNotIn("cuda_completion", follow)
            self.assertEqual(follow["counts"], [2, 3, 4, 5])
            self.assertEqual(follow["recovered_six"], contract.RECOVERED_SIX)
            self.assertEqual(follow["fixture_module_sha256"], first["fixture_module_sha256"])
            for backend, request in (("mlx", subset), ("candle-cuda", dict(subset, counts=[2])),
                                     ("candle-cuda", dict(subset, policy=contract.policy("candle-cuda"))),
                                     ("candle-cuda", dict(subset, policy=dict(full["policy"], deadlineSeconds=1)))):
                with self.assertRaises(ValueError):
                    runner.plan(backend, request, "r")

    @unittest.skipUnless(os.environ.get("SCAIL_COMPLETED_JOB_ARCHIVE"), "original Actions archive not present")
    def test_original_sealed_six_and_independent_mutations(self):
        archive = Path(os.environ["SCAIL_COMPLETED_JOB_ARCHIVE"])
        cert = recovery.recover(archive)
        candidate = cert["candidate_row_pending_physical_review"]
        self.assertEqual(candidate["witness"]["gross_host_bytes"], 58_836_885_504)
        self.assertEqual(candidate["witness"]["gross_device_bytes"], 57_048_903_680)
        self.assertEqual(candidate["cuda_completed_job"]["pool_card_index"], 2311)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            altered = root / "altered.zip"
            altered.write_bytes(archive.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "authenticated"):
                recovery.recover(altered)
            # Only extract an already authenticated exact ZIP.
            with ZipFile(archive) as captured:
                captured.extractall(root)
            guard_path = root / "count-6.supervisor.json"
            captured = json.loads(guard_path.read_bytes())
            policy = runner.tier1.read_policy(root, "policy.json", contract.policy("candle-cuda", cuda_uuid=contract.GPU0))
            for problem in ("missing-attempt", "earlier-gap", "foreign-job", "native-pool", "both-backwards"):
                with self.subTest(problem=problem):
                    guard = copy.deepcopy(captured)
                    raw = guard["fit_observation"]
                    tail = raw["cuda_render_end"]["rejected_growth"]
                    if problem == "missing-attempt":
                        tail.pop(0)
                    elif problem == "earlier-gap":
                        raw["wddm_retry"]["dedicated"]["events"][0]["start_ns"] = tail[-2]["read_ns"]
                    elif problem == "foreign-job":
                        tail[-1]["pid"] = 1
                    elif problem == "both-backwards":
                        for gauge in raw["wddm_retry"].values():
                            gauge["events"][-1]["local_filetime_after"] = 0
                    else:
                        tail[-1]["pool"]["poolReservedHighBytes"] -= 1
                    guard_path.write_text(json.dumps(guard))
                    with self.assertRaises(ValueError):
                        recovery.replay_files(root, policy)
