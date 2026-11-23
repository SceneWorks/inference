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
from scripts.ci.scail_plural import completed_job, contract, cuda_stage
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
    # The final attempt's card-wide bound (50) stays within the recorded dedicated peak (50).
    tail.extend((dict(tail[-1], read_ns=345, at_ns=355),
                 dict(tail[-1], read_ns=375, at_ns=390, global_used_upper_bytes=50)))
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


def scope(guard, pool, claim, *, fresh=False, basis="owned-job-completion"):
    uuid = pool["cudaDeviceUuid"]
    _, primary = cuda_stage.tenant_growth(guard, uuid)
    return cuda_stage.render_scope(guard, pool, uuid, primary, completed_job=(claim, basis), fresh_cuda=fresh)


MS = 1_000_000
UNIX_OFFSET_NS = 1_795_000_000_000_000_000  # UTC minus monotonic
LOCAL_OFFSET_NS = -5 * 3600 * 1_000_000_000  # Eastern Standard Time, as on the GPU 0 runner


def filetime(monotonic_ns):
    return (monotonic_ns + UNIX_OFFSET_NS + completed_job.FILETIME_UNIX_EPOCH_NS + LOCAL_OFFSET_NS) // 100


def clocked_exit_fixture():
    """Run 45281400027 count 4's shape on real clocks.

    The final attempt's card read saw 1 MiB of child growth, then the root had left the
    Job before either WDDM gauge read it (invalid allocation, Job exited empty). Every
    earlier card read and attempt holds its own dedicated and shared sample: PDH local
    FILETIMEs 2 ms and 1 ms before each monotonic record, with the supervisor's UTC
    growth stamp 1 ms after card row 0.
    """
    guard, pool, claim = completed_fixture()
    raw = guard["fit_observation"]
    end = raw["cuda_render_end"]
    card = raw["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    for row in card:
        row[0] *= MS
    for entry in end["pool_growth"] + end["rejected_growth"]:
        for key in ("read_ns", "at_ns"):
            entry[key] *= MS
    end["pool"]["at_ns"] *= MS
    end["rejected_growth"][-1].update(detail=cuda_stage.RejectedPoolSample.INVALID_ALLOCATION, after="exited_empty")
    growth, = (e for e in guard["samples"] if e["phase"] == "cuda-global-used-upper-bound-sample")
    growth["at_ns"] = card[0][0] + UNIX_OFFSET_NS + MS
    records = [row[0] for row in card] + [e["at_ns"] for e in end["rejected_growth"]]
    for name, series, lead in (("dedicated", guard["cuda_attribution"]["wddm_dedicated"], 2 * MS),
                               ("shared", raw["wddm_shared"], MS)):
        values = [value for _, value in series["samples"]]
        series["samples"] = [[filetime(at - lead), value] for at, value in zip(records, values)]
        series["missing_active_polls"] = 0
        raw["wddm_retry"][name] = {"attempts": 0, "events": []}
    return guard, pool, claim


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
                    # Owned memory did not return to the admission reading and the Job is unverified.
                    claim["release_background_activity"]["samples"][4]["selected_card"]["used_mib"] = (
                        completed_job.RELEASE_BASELINE_TOLERANCE_MIB + 1)
                elif problem == "release":
                    claim["release_metadata_verification"]["status"] = "inconclusive-driver-query-timing"
                elif problem == "return":
                    guard["returncode"] = 1
                with self.assertRaises(ValueError):
                    scope(guard, pool, claim)

    def _owned_empty(self, guard, claim):
        completion = dict(scope="supervisor-owned-windows-job", root_pid=guard["pid"], root_returncode=0,
                          waited_returncode=0, owned_pids=[], job_member_pids=[])
        guard["owned_completion"] = completion
        claim["owned_execution_completion"] = dict(completion)

    def _cards(self, claim, phase, used, utilization=0):
        claim[phase + "_background_activity"] = {"samples": [
            {"selected_card": {"used_mib": used, "utilization_percent": utilization}} for _ in range(5)]}

    def test_foreign_idle_allocation_at_admission_and_release_is_accepted(self):
        # Release is never an idle-card check: memory resident before admission stays resident.
        guard, pool, claim = completed_fixture()
        self._cards(claim, "admission", 498)
        self._cards(claim, "release", 498)
        self.assertEqual(scope(guard, pool, claim)["terminal_rejected_attempts"], 4)

    def test_unowned_memory_after_a_verified_empty_job_is_accepted(self):
        # Run 45247572882 count 3: 19 MiB at admission, an unrelated idle 498 MiB at release,
        # 0% utilization, and the owned Windows Job verified empty.
        guard, pool, claim = completed_fixture()
        self._cards(claim, "admission", 19)
        self._cards(claim, "release", 498)
        self._owned_empty(guard, claim)
        self.assertEqual(scope(guard, pool, claim)["terminal_rejected_attempts"], 4)

    def test_owned_memory_not_returning_to_the_admission_baseline_is_rejected(self):
        for job in ("unverified", "member-left", "other-root"):
            with self.subTest(job=job):
                guard, pool, claim = completed_fixture()
                self._cards(claim, "admission", 19)
                self._cards(claim, "release", 19 + completed_job.RELEASE_BASELINE_TOLERANCE_MIB + 1)
                if job != "unverified":
                    self._owned_empty(guard, claim)
                    if job == "member-left":
                        claim["owned_execution_completion"]["job_member_pids"] = [guard["pid"] + 1]
                    else:
                        claim["owned_execution_completion"]["root_pid"] = guard["pid"] + 1
                with self.assertRaisesRegex(ValueError, "quiet CUDA release"):
                    scope(guard, pool, claim)
        guard, pool, claim = completed_fixture()
        self._cards(claim, "admission", 19)
        self._cards(claim, "release", 19 + completed_job.RELEASE_BASELINE_TOLERANCE_MIB)
        self.assertEqual(scope(guard, pool, claim)["terminal_rejected_attempts"], 4)

    def test_busy_release_card_is_rejected_even_after_an_empty_job(self):
        guard, pool, claim = completed_fixture()
        self._cards(claim, "release", 0, utilization=99)
        self._owned_empty(guard, claim)
        with self.assertRaisesRegex(ValueError, "quiet CUDA release"):
            scope(guard, pool, claim)

    def test_launch_read_without_wddm_is_accepted_and_gaps_after_allocation_are_not(self):
        mib = 1 << 20
        # 35 zero-owned launch reads; the gauge has a counter only from the 2nd read (34 samples).
        card = [[i, 9 * mib, 0] for i in range(35)] + [[35 + i, 8 * mib, 490 * mib] for i in range(20)]
        samples = [[i, 264 * 1024] for i in range(34)] + [[34 + i, 490 * mib] for i in range(20)]
        self.assertEqual(completed_job.unsampled_launch_reads(card, samples), 1)
        # Every launch read sampled but one sample missing after allocation: nothing is excused,
        # so the terminal count stays one short and is rejected.
        launch = [[i, 264 * 1024] for i in range(35)]
        self.assertEqual(completed_job.unsampled_launch_reads(card, launch + samples[35:]), 0)
        self.assertEqual(completed_job.unsampled_launch_reads(card, launch + samples[34:]), 0)
        # No zero-owned launch read, or two missing launch samples: nothing is excused.
        self.assertEqual(completed_job.unsampled_launch_reads(card[35:], samples[34:]), 0)
        self.assertEqual(completed_job.unsampled_launch_reads(card, samples[1:]), 0)
        # A gap after allocation still fails the terminal witness (the fixture's own gauges).
        guard, pool, claim = completed_fixture()
        guard["cuda_attribution"]["wddm_dedicated"]["samples"].pop(-2)
        with self.assertRaises(ValueError):
            scope(guard, pool, claim)

    def test_final_attempt_after_the_job_exit_needs_no_wddm_reading(self):
        guard, pool, claim = clocked_exit_fixture()
        result = scope(guard, pool, claim, fresh=True)
        self.assertEqual((result["dedicated_missing_after_output"], result["shared_missing_after_output"]), (0, 0))
        # Archived captures and the older GPU 0 continuation keep their original verdict.
        for basis in ("owned-job-completion", "archived-supervisor-return"):
            with self.assertRaisesRegex(ValueError, "unlocated WDDM gap during render"):
                scope(guard, pool, claim, basis=basis)

    def test_exit_excuse_holds_only_when_each_earlier_attempt_has_its_own_pair(self):
        for dropped in range(6):
            with self.subTest(dropped=dropped):
                # Drop an earlier pair and add one in the final attempt: counts match, clocks do not.
                guard, pool, claim = clocked_exit_fixture()
                end = guard["fit_observation"]["cuda_render_end"]
                final_records = end["rejected_growth"][-1]["at_ns"]
                for series, lead, value in ((guard["cuda_attribution"]["wddm_dedicated"], 2 * MS, 12),
                                            (guard["fit_observation"]["wddm_shared"], MS, 6)):
                    series["samples"].pop(dropped)
                    series["samples"].append([filetime(final_records - lead), value])
                with self.assertRaisesRegex(ValueError, "unlocated WDDM gap during render"):
                    scope(guard, pool, claim, fresh=True)
        with self.subTest(problem="extra-earlier-pair"):
            # A second pair inside card read 1 and none for the attempt before the exit.
            guard, pool, claim = clocked_exit_fixture()
            record = guard["fit_observation"]["card_samples_monotonic_ns_free_owned_or_fallback_bytes"][1][0]
            for series, lead in ((guard["cuda_attribution"]["wddm_dedicated"], MS + MS // 2),
                                 (guard["fit_observation"]["wddm_shared"], MS // 2)):
                series["samples"].insert(2, [filetime(record - lead), 10])
                series["samples"].pop()
            with self.assertRaisesRegex(ValueError, "unlocated WDDM gap during render"):
                scope(guard, pool, claim, fresh=True)
        for problem in ("unequal-lengths", "shared-crosses-next-dedicated", "still-member"):
            with self.subTest(problem=problem):
                guard, pool, claim = clocked_exit_fixture()
                raw = guard["fit_observation"]
                ded, shared = guard["cuda_attribution"]["wddm_dedicated"]["samples"], raw["wddm_shared"]["samples"]
                if problem == "unequal-lengths":
                    shared.pop()
                elif problem == "shared-crosses-next-dedicated":
                    shared[2][0] = ded[3][0]
                else:
                    raw["cuda_render_end"]["rejected_growth"][-1]["detail"] = (
                        cuda_stage.RejectedPoolSample.EXCEEDS_CURRENT_USAGE)
                with self.assertRaisesRegex(ValueError, "unlocated WDDM gap during render"):
                    scope(guard, pool, claim, fresh=True)

    def test_exit_excuse_is_bounded_by_the_final_card_read(self):
        for problem in ("above-peak", "slow-attempt"):
            with self.subTest(problem=problem):
                guard, pool, claim = clocked_exit_fixture()
                final = guard["fit_observation"]["cuda_render_end"]["rejected_growth"][-1]
                if problem == "above-peak":
                    final["global_used_upper_bytes"] = 51  # recorded dedicated peak is 50
                else:
                    final["at_ns"] = final["read_ns"] + 250 * MS + 1
                with self.assertRaisesRegex(ValueError, "not bounded by its card read"):
                    scope(guard, pool, claim, fresh=True)

    def test_fresh_terminal_gap_must_be_bounded_by_its_attempts_card_read(self):
        scope(*completed_fixture(), fresh=True)  # bound equal to the recorded dedicated peak
        for problem in ("above-peak", "slow-attempt"):
            with self.subTest(problem=problem):
                guard, pool, claim = completed_fixture()
                final = guard["fit_observation"]["cuda_render_end"]["rejected_growth"][-1]
                if problem == "above-peak":
                    final["global_used_upper_bytes"] = 51
                else:
                    final["at_ns"] = final["read_ns"] + 250 * MS + 1
                with self.assertRaisesRegex(ValueError, "not bounded by its card read"):
                    scope(guard, pool, claim, fresh=True)
                # The older GPU 0 continuation replays its stored receipts unchanged.
                self.assertEqual(scope(guard, pool, claim)["terminal_rejected_attempts"], 4)

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
                    guard_path.write_text(json.dumps(guard), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        recovery.replay_files(root, policy)
