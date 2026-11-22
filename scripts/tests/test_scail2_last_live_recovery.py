"""A missing final live WDDM read must be followed by the real, owned Job zero."""

import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from zipfile import ZipFile

from scripts.ci import scail2_recover_last_live as recovery
from scripts.ci.scail_plural import cuda_stage
from scripts.tests.test_scail2_owned_zero_recovery import fixture


def scope(guard, pool, claim, archive):
    _, n = cuda_stage.tenant_growth(guard, pool["cudaDeviceUuid"])
    return cuda_stage.render_scope(guard, pool, pool["cudaDeviceUuid"], n, fresh_cuda=True,
                                   completed_job=(claim, "owned-job-completion"),
                                   archived_owned_zero=archive, last_rejection_live=archive)


class LastLiveRecovery(unittest.TestCase):
    def test_archive_only_and_every_live_observation_paired(self):
        guard, pool, claim = fixture()
        raw = guard["fit_observation"]
        end = raw["cuda_render_end"]
        end["rejected_growth"][-1]["after"] = end["zero"]["before"] = "live"
        middle = dict(end["rejected_growth"][0], read_ns=321, at_ns=322)
        end["rejected_growth"].insert(1, middle)
        guard["cuda_attribution"]["unknown_pid_polls"] += 1
        dedicated = guard["cuda_attribution"]["wddm_dedicated"]["samples"]
        shared = raw["wddm_shared"]["samples"]
        dedicated[:0] = [[20, 5], [40, 5], [60, 4]]
        shared[:0] = [[21, 0], [41, 0], [61, 0]]
        self.assertIsNone(scope(guard, pool, claim, False))
        self.assertEqual(scope(guard, pool, claim, True)["terminal_rejected_read_ns"], 325)
        shared.pop(1)
        with self.assertRaises(ValueError):
            scope(guard, pool, claim, True)

    @unittest.skipUnless(os.environ.get("SCAIL_LAST_LIVE_ARCHIVE"), "requires the immutable five-count ZIP")
    def test_original_archive_and_raw_mutations(self):
        archive = Path(os.environ["SCAIL_LAST_LIVE_ARCHIVE"])
        cert = recovery.recover(archive)
        canonical = (json.dumps(cert, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                         "ca4696364a2a4c951b76f58879e0d25b7e48fdd4d6df5d9e7b605c313cbfd02d")
        with ZipFile(archive) as z:
            original = json.loads(z.read("count-3.supervisor.json"))
            pool = json.loads(z.read("count-3.cuda-pool.json"))
            claim = json.loads(z.read("receipt.json"))["counts"]["3"]["claim"]
        self.assertIsNone(scope(original, pool, claim, False))
        self.assertEqual(scope(original, pool, claim, True),
                         cert["candidate_row_pending_physical_review"]["cuda_render_scope"])
        for problem in ("missing-pair", "cross-gauge", "bad-filetime", "earlier-attempt",
                        "earlier-retirement", "wrong-zero", "live-job", "nonquiet-release"):
            with self.subTest(problem=problem):
                guard, lease = copy.deepcopy((original, claim))
                raw = guard["fit_observation"]
                end = raw["cuda_render_end"]
                shared = raw["wddm_shared"]["samples"]
                if problem == "missing-pair":
                    shared.pop(1)
                    guard["cuda_attribution"]["wddm_dedicated"]["samples"].pop(1)
                elif problem == "cross-gauge":
                    shared[0][0] = guard["cuda_attribution"]["wddm_dedicated"]["samples"][1][0]
                elif problem == "bad-filetime":
                    raw["wddm_retry"]["shared"]["events"][0]["local_filetime_after"] = 0
                elif problem == "earlier-attempt":
                    raw["wddm_retry"]["dedicated"]["events"][0]["start_ns"] = end["rejected_growth"][-2]["read_ns"]
                elif problem == "earlier-retirement":
                    end["rejected_growth"][-1]["after"] = end["zero"]["before"] = "exited_empty"
                elif problem == "wrong-zero":
                    end["zero"]["created_filetime"] += 1
                elif problem == "live-job":
                    guard["owned_completion"]["job_member_pids"] = [guard["pid"]]
                else:
                    lease["release_background_activity"]["samples"][0]["selected_card"]["used_mib"] = 2
                with self.assertRaises(ValueError):
                    scope(guard, pool, lease, True)
