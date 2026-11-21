"""Offline process, time and publication contracts for the one GPU0 Mochi completion pair."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.ci import mochi_pair_completion as boundary
from scripts.ci import video_tier1_comparison as proof, video_tier1_cuda_diagnostic as diagnostic
from scripts.ci import wddm_process_memory as wddm
from scripts.tests import test_video_tier1_cuda_diagnostic as contracts

UUID = diagnostic.GPU0_HARDWARE["uuid"]
ADAPTER = {"cuda_uuid": UUID, "luid": "luid_0x00000000_0x0001f603", "node_mask": 1, "physical_node": 0, "counter": wddm.COUNTER}
WINDOW = {"start_ns": 4, "end_ns": 306}


def evidence():
    offset = 1000
    packet = {"pid": 14, "case": boundary.CASE, "uuid": UUID, "ready_utc_ns": offset + 13}
    ready = {"packet": packet, "seen_utc_ns": offset + 14, "seen_ns": 14, "local_filetime": 1035}
    ack = {"ready": packet, "sample": [1040, 36], "sample_count": 2, "created_filetime": boundary.UNIX_EPOCH_FILETIME + 1,
           "adapter": ADAPTER, "end_local_filetime": 1050, "ack_ns": 15, "ack_utc_ns": offset + 16}
    native = {"ack": deepcopy(ack), "release_utc_ns": offset + 20}
    release = {"packet": deepcopy(native), "sample_count": 2, "seen_ns": 25, "seen_utc_ns": offset + 26, "local_filetime": 1100, "retry_count": 0}
    miss = {"start_ns": 29, "end_ns": 31, "local_filetime_before": 1130, "local_filetime_after": 1140,
            "retried": True, "retry_sample": None, "previous": [1040, 36], "recovered": False}
    ev = {"diagnostics": {"attempts": 1, "events": [miss]}, "ready": ready, "release": release, "ack": ack,
          "reason": diagnostic.RETRY_REASON, "pid": 14, "created_filetime": boundary.UNIX_EPOCH_FILETIME + 1, "adapter": ADAPTER}
    raw = {"adapter": ADAPTER, "pid": 14, "created_filetime": boundary.UNIX_EPOCH_FILETIME + 1, "error": None,
           "missing_active_polls": 1, "samples": [[1010, 39], [1040, 36]]}
    guard = {"pid": 14, "fit_observation": ev, "cuda_attribution": {"wddm_dedicated": raw}, "owned_completion":
             {"scope": "supervisor-owned-windows-job", "root_pid": 14, "root_returncode": 0, "waited_returncode": 0,
              "job_member_pids": [], "owned_pids": []}}
    return guard, native


def grade(guard, native):
    diagnostic.verify_retry(guard, SimpleNamespace(poll_millis=125), ADAPTER)
    live = boundary.live_record(guard, native, ADAPTER, WINDOW)
    return wddm.sampled_peak(live, pid=14, adapter=ADAPTER)


class MochiCompletionTests(unittest.TestCase):
    def test_pair_contains_two_fresh_exact_roles_and_the_retained_gpu0_receipt(self):
        _, previous = contracts.TargetedCudaTests().binding(proof.cohorts.MOCHI_SVD, name=diagnostic.MOCHI_P1, uuid=UUID)
        cases, extension = diagnostic.continuation(diagnostic.MOCHI_P1, proof.cohorts.MOCHI_SVD, previous)
        self.assertEqual(cases, [boundary.CASE])
        self.assertEqual(list(proof.schedule(boundary.CASE, extension)), [
            (1, arm, f"{boundary.CASE}-memory-p1-{arm}", 1) for arm in ("base", "head")])
        self.assertEqual(extension["protocol"]["previous_gpu0"]["receipt_sha256"],
                         "823f91359bb1c22447c8dbba5c624e21ef6eed059b0c40ce2909910ffeabe011")
        for other in ("cuda-svd-euler-cfg", "cuda-ti2v5b-t2v-heun"):
            self.assertEqual(list(proof.schedule(other, extension)), [])
        with self.assertRaises(ValueError):
            diagnostic.continuation(diagnostic.MOCHI_P1, proof.cohorts.WAN, previous)

    def test_separates_only_observed_teardown_and_preserves_every_raw_poll(self):
        guard, native = evidence()
        original = deepcopy(guard)
        self.assertEqual(grade(guard, native), 39)
        self.assertEqual(guard, original)
        raw = guard["cuda_attribution"]["wddm_dedicated"]
        self.assertIsNone(wddm.sampled_peak(raw, pid=14, adapter=ADAPTER))
        witness = guard["fit_observation"]
        early = dict(witness["diagnostics"]["events"][0], start_ns=17, end_ns=19,
                     local_filetime_before=1060, local_filetime_after=1070)
        witness["diagnostics"]["events"].insert(0, early)
        witness["diagnostics"]["attempts"] = raw["missing_active_polls"] = 2
        witness["release"]["retry_count"] = 1
        self.assertIsNone(grade(guard, native))
        witness["release"]["retry_count"] = 0
        with self.assertRaisesRegex(ValueError, "straddled"):
            grade(guard, native)

    def test_tampering_late_allocation_error_and_pid_substitution_cannot_pass(self):
        for path, change in [
                (("owned_completion", "root_pid"), 15), (("owned_completion", "owned_pids"), [14]),
                (("fit_observation", "ready", "local_filetime"), 1040),
                (("fit_observation", "ack", "sample_count"), 1),
                (("fit_observation", "release", "sample_count"), 1),
                (("fit_observation", "release", "seen_ns"), WINDOW["end_ns"] + 1),
                (("fit_observation", "release", "seen_utc_ns"), 19),
                (("cuda_attribution", "wddm_dedicated", "samples"), [[1010, 39], [1040, 36], [1120, 37]]),
                (("cuda_attribution", "wddm_dedicated", "samples"), [[1010, 39], [1040, 36], [1150, 2]])]:
            with self.subTest(path=path, change=change):
                guard, native = evidence()
                row = guard
                for key in path[:-1]:
                    row = row[key]
                row[path[-1]] = change
                with self.assertRaises(ValueError):
                    grade(guard, native)
        guard, native = evidence()
        with self.assertRaises(ValueError):
            grade(guard, dict(native, release_utc_ns=native["release_utc_ns"]+1))
        guard["cuda_attribution"]["wddm_dedicated"]["error"] = "lost Job"
        self.assertIsNone(grade(guard, native))

    def test_controller_only_acknowledges_a_new_fresh_positive_owned_sample(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            packet = {"pid": 14, "case": boundary.CASE, "uuid": UUID, "ready_utc_ns": 1}
            (out / "mochi-ready.json").write_text(json.dumps(packet))
            job = MagicMock()
            job.members.return_value, job.root_identity.return_value = {14}, (12345, False)
            gauge = wddm.DedicatedSamples(ADAPTER, UUID)
            observer = boundary.CompletionSamples(gauge, SimpleNamespace(poll_millis=125), out)
            with patch.object(wddm, "_query") as query, patch.object(boundary, "utc_ns", side_effect=[10, 20, 30]), \
                    patch.object(boundary, "local_filetime", side_effect=[100, 105, 110, 125, 130]):
                query.return_value.__enter__.return_value.side_effect = [[98, 30], [98, 30], [106, 0], [116, 44], None]
                for _ in range(3):
                    observer.poll(job, cuda_active=True)
                    self.assertFalse((out / "mochi-ack.json").exists())
                observer.poll(job, cuda_active=True)
                ack = json.loads((out / "mochi-ack.json").read_bytes())
                self.assertEqual((ack["ready"], ack["sample"], ack["created_filetime"]), (packet, [116, 44], 12345))
                (out / "mochi-release.json").write_text(json.dumps({"ack": ack, "release_utc_ns": ack["ack_utc_ns"]+1}))
                observer.check_release()
                self.assertEqual(observer.release["sample_count"], 3)
                observer.close()
