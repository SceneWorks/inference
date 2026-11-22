"""The archived terminal read can tie a gauge only on its own synchronous attempt."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.ci import scail2_recover_owned_zero as recovery
from scripts.ci.scail_plural import cuda_stage as stage
from scripts.tests.test_scail2_cuda_stage import fixture_with_rejected_growth


def fixture():
    guard, pool = fixture_with_rejected_growth()
    raw = guard["fit_observation"]
    end = raw["cuda_render_end"]
    end["rejected_growth"][-1]["after"] = end["zero"]["before"] = "exited_empty"
    gaps = [raw["wddm_retry"][k]["events"][0] for k in ("dedicated", "shared")]
    gaps[0].update(start_ns=325, end_ns=330)
    gaps[1].update(start_ns=330, end_ns=340, local_filetime_before=185, local_filetime_after=190)
    guard["admission"]["pollMillis"] = 250
    guard["returncode"] = 0
    guard["owned_completion"] = dict(root_pid=guard["pid"], scope="supervisor-owned-windows-job",
                                     root_returncode=0, waited_returncode=0, owned_pids=[], job_member_pids=[])
    claim = dict(owned_execution_completion=guard["owned_completion"].copy(),
                 release_metadata_verification={"status": "verified"}, release_background_activity={
                     "samples": [{"selected_card": {"used_mib": 0, "utilization_percent": 0}} for _ in range(5)]})
    return guard, pool, claim


def scope(guard, pool, claim, archived=True, fresh=False):
    _, primary = stage.tenant_growth(guard, pool["cudaDeviceUuid"])
    return stage.render_scope(guard, pool, pool["cudaDeviceUuid"], primary,
                              completed_job=(claim, "owned-job-completion"), archived_owned_zero=archived, fresh_cuda=fresh)


class OwnedZeroRecovery(unittest.TestCase):
    def test_fresh_gpu0_locates_only_the_verified_terminal_tie(self):
        guard, pool, claim = fixture()
        self.assertEqual(scope(guard, pool, claim, archived=False, fresh=True), scope(guard, pool, claim))
        raw = guard["fit_observation"]
        event = raw["wddm_retry"]["dedicated"]["events"][0]
        event.update(start_ns=150, end_ns=160)
        self.assertIsNone(scope(guard, pool, claim, archived=False, fresh=True))
        guard, pool, claim = fixture()
        guard["fit_observation"]["wddm_retry"]["shared"]["events"][0]["local_filetime_after"] = 0
        self.assertIsNone(scope(guard, pool, claim, archived=False, fresh=True))
        guard, pool, claim = fixture()
        guard["fit_observation"]["cuda_render_end"]["rejected_growth"][-1]["free_bytes"] = 1
        with self.assertRaises(ValueError):
            scope(guard, pool, claim, archived=False, fresh=True)

    def test_fresh_strict_post_output_gap_validates_both_clocks(self):
        for gauge in ("dedicated", "shared"):
            guard, pool, claim = fixture()
            diagnostics = guard["fit_observation"]["wddm_retry"]
            diagnostics["dedicated"]["events"][0]["start_ns"] += 1
            self.assertIsNotNone(scope(guard, pool, claim, archived=False, fresh=True))
            diagnostics[gauge]["events"][0]["local_filetime_after"] = 0
            self.assertIsNone(scope(guard, pool, claim, archived=False, fresh=True))
            self.assertIsNotNone(scope(guard, pool, claim, archived=False))

    def test_archived_tied_read_only_and_raw_record_preserved(self):
        guard, pool, claim = fixture()
        before = json.dumps(guard)
        self.assertIsNone(scope(guard, pool, claim, archived=False))
        result = scope(guard, pool, claim)
        self.assertEqual((result["terminal_rejected_read_ns"], result["zero_card_index"]), (325, 3))
        self.assertEqual((result["dedicated_missing_after_output"], result["shared_missing_after_output"]), (1, 1))
        self.assertEqual(before, json.dumps(guard))

    def test_other_reads_earlier_gaps_or_unverified_release_do_not_qualify(self):
        problems = ("ordinary-card-tie", "previous-rejection", "zero-last-growth", "gauge-overlap",
                    "filetime-crossing", "dedicated-filetime-length", "shared-filetime-length",
                    "shared-clock-reversed", "late-retry", "post-zero-read", "missing-dedicated", "other-shared",
                    "different-pid", "changed-creation", "wrong-pool", "unfinished-job", "wrong-zero",
                    "failed-unlock-survey", "nonquiet-unlock", "breached-reserve", "breached-cap")
        for problem in problems:
            with self.subTest(problem=problem):
                guard, pool, claim = fixture()
                raw = guard["fit_observation"]
                end = raw["cuda_render_end"]
                final = end["rejected_growth"][-1]
                ded, shared = (raw["wddm_retry"][k]["events"][0] for k in ("dedicated", "shared"))
                if problem == "ordinary-card-tie":
                    ded["start_ns"] = raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"][-2][0]
                elif problem == "previous-rejection":
                    ded["start_ns"] = end["rejected_growth"][-2]["read_ns"]
                elif problem == "zero-last-growth":
                    final.update(child_growth_bytes=0, global_used_upper_bytes=50)
                elif problem == "gauge-overlap":
                    ded["end_ns"] += 1
                elif problem == "filetime-crossing":
                    shared["local_filetime_before"] -= 1
                elif problem == "dedicated-filetime-length":
                    ded["local_filetime_after"] = ded["local_filetime_before"] + 2_500_001
                elif problem == "shared-filetime-length":
                    shared["local_filetime_after"] = shared["local_filetime_before"] + 2_500_001
                elif problem == "shared-clock-reversed":
                    shared["local_filetime_after"] = shared["local_filetime_before"] - 1
                elif problem == "late-retry":
                    shared["end_ns"] += 1
                elif problem == "post-zero-read":
                    final["at_ns"] = end["zero"]["at_ns"]
                elif problem == "missing-dedicated":
                    guard["cuda_attribution"]["wddm_dedicated"]["missing_active_polls"] += 1
                elif problem == "other-shared":
                    shared["previous"] = raw["wddm_shared"]["samples"][0]
                elif problem == "different-pid":
                    final["pid"] += 1
                elif problem == "changed-creation":
                    final["created_filetime"] += 1
                elif problem == "wrong-pool":
                    final["pool"]["poolUsedBytes"] -= 1
                elif problem == "unfinished-job":
                    guard["owned_completion"]["root_returncode"] = 1
                elif problem == "wrong-zero":
                    end["zero"]["before"] = "live"
                elif problem == "failed-unlock-survey":
                    claim["release_metadata_verification"] = {"status": "inconclusive-driver-query-timing"}
                elif problem == "nonquiet-unlock":
                    claim["release_background_activity"]["samples"][0]["selected_card"]["used_mib"] = 1
                elif problem == "breached-reserve":
                    raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"][0][1] = 1
                elif problem == "breached-cap":
                    guard["admission"]["childGpuCapBytes"] = 70
                with self.assertRaises(ValueError):
                    scope(guard, pool, claim)

    @unittest.skipUnless(os.environ.get("SCAIL_OWNED_ZERO_ARCHIVE"), "original four-count archive not present")
    def test_authenticate_exact_archive_replay_original_verdicts_and_derive_per_count(self):
        archive = Path(os.environ["SCAIL_OWNED_ZERO_ARCHIVE"])
        certificates = recovery.recover(archive)
        expected = {"3": "c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d",
                    "5": "bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7"}
        self.assertEqual(set(expected), set(certificates))
        for key, cert in certificates.items():
            encoded = (json.dumps(cert, sort_keys=True, separators=(",", ":")) + "\n").encode()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected[key])
        # Reproduce the strict interior (non-tied) post-output path on the two
        # authentic records. A malformed clock on either gauge must refuse it.
        with ZipFile(archive) as raw:
            for count in (3, 5):
                claim = json.loads(raw.read("receipt.json"))["counts"][str(count)]["claim"]
                pool = json.loads(raw.read(f"count-{count}.cuda-pool.json"))
                for gauge in ("dedicated", "shared"):
                    for field, change in (("local_filetime_after", lambda _: 0),
                                          ("local_filetime_after", lambda v: v + 2_500_001),
                                          ("start_ns", float)):
                        with self.subTest(count=count, gauge=gauge, field=field):
                            guard = json.loads(raw.read(f"count-{count}.supervisor.json"))
                            diagnostics = guard["fit_observation"]["wddm_retry"]
                            diagnostics["dedicated"]["events"][0]["start_ns"] += 1
                            self.assertIsNotNone(scope(guard, pool, claim, archived=False, fresh=True))
                            event = diagnostics[gauge]["events"][0]
                            event[field] = change(event[field])
                            self.assertIsNone(scope(guard, pool, claim, archived=False, fresh=True))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "changed.zip"
            path.write_bytes(archive.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "authenticated"):
                recovery.recover(path)
