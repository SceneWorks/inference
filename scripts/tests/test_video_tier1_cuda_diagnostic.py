"""Metadata-only checks for the frozen CUDA diagnostic and its evidence provenance."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from scripts.ci import video_tier1_comparison as proof
from scripts.ci import video_tier1_cuda_diagnostic as diagnostic
from scripts.ci import video_tier1_followup as followup


class TargetedCudaTests(unittest.TestCase):
    def binding(self, selection, tracked=None, *, name=followup.NAME, uuid=followup.CUDA_UUID):
        def read(_head, path):
            return (proof.ROOT / path).read_bytes()
        policy = dict(schemaVersion=1, backend="windows-cuda", deadlineSeconds=180, pollMillis=125,
                      termGraceMillis=2000, hostFreeReserveBytes=8 * proof.GIB, childFootprintCapBytes=20 * proof.GIB,
                      cudaDeviceUuid=uuid, gpuFreeReserveBytes=6 * proof.GIB, childGpuCapBytes=12 * proof.GIB,
                      **dict.fromkeys(("stdoutCapBytes", "stderrCapBytes", "eventCapBytes"), 4 * proof.MIB))
        _, _, prior = followup.bind("controller", "candle-cuda", selection, read, policy, name)
        return diagnostic.bind("controller", "candle-cuda", selection, tracked or read, prior)

    def test_continuation_has_three_independent_fixed_schedules_and_frozen_rank_interval(self):
        for cohort, name, count in ((proof.cohorts.WAN, diagnostic.TIMING_CONTINUATION, 80),
                                    (proof.cohorts.MOCHI_SVD, diagnostic.TIMING_CONTINUATION, 80),
                                    (proof.cohorts.MOCHI_SVD, diagnostic.MEMORY_CONTINUATION, 6)):
            _, previous = self.binding(cohort)
            cases, current = diagnostic.continuation(name, cohort, previous)
            self.assertEqual(len(proof.phases(cases, current)), count)
            for case in cases:
                retained = current["retained"][case]
                rows = list(proof.schedule(case, current))
                if name == diagnostic.TIMING_CONTINUATION:
                    self.assertEqual(len(rows), 80)
                    self.assertEqual([r[1] for r in rows[::2]].count("base"), 20)
                else:
                    self.assertEqual(tuple(r[0] for r in rows[::2]), diagnostic.CONTINUATION_MEMORY[case])
                data = {}
                for position, arm, key, repeats in rows:
                    ns = 110 if arm == "head" and position > 26 else 100
                    data[key] = {"initial": {"sha256": retained["initial_sha256"]},
                        "terminal": {"sha256": retained["terminal_sha256"]}, "trace": [],
                        "trace_sha256": retained["trace_sha256"], "samples": [{"outer_steps": 4, "elapsed_ns": ns}] * repeats,
                        "peaks": {"pool": 100}, "cuda_process": {"lower": 50, "upper": 200, "basis": "wddm-bounded",
                        "dedicated": {"adapter": retained["dedicated_adapter"], "sampled_peak_bytes": 100}}}
                result = proof.compare_case(case, current, data)
                if name == diagnostic.TIMING_CONTINUATION:
                    self.assertEqual(result["timing"]["interval"], [1, 1.1])
                    self.assertFalse(result["accepted"])
                    self.assertNotIn("dedicated_process_memory", result)
                    data[f"{case}-timing-p27-head"]["samples"] = data[f"{case}-timing-p27-base"]["samples"]
                    self.assertTrue(proof.compare_case(case, current, data)["accepted"])
                else:
                    self.assertNotIn("timing", result)
                    self.assertEqual(len(result["dedicated_process_memory"]["pairs"]), len(rows) // 2)
                    self.assertTrue(result["accepted"])
        _, prior = self.binding(proof.cohorts.WAN)
        with self.assertRaises(ValueError):
            diagnostic.continuation(diagnostic.MEMORY_CONTINUATION, proof.cohorts.WAN, prior)

    def test_gpu0_continuations_bind_fresh_card_and_preserve_original_gpu1_scope(self):
        gpu0 = diagnostic.GPU0_HARDWARE
        for cohort, fresh, old, count in (
            (proof.cohorts.WAN, diagnostic.GPU0_TIMING, diagnostic.TIMING_CONTINUATION, 80),
            (proof.cohorts.MOCHI_SVD, diagnostic.GPU0_TIMING, diagnostic.TIMING_CONTINUATION, 80),
            (proof.cohorts.MOCHI_SVD, diagnostic.GPU0_MEMORY, diagnostic.MEMORY_CONTINUATION, 6),
        ):
            _, previous = self.binding(cohort, name=fresh, uuid=gpu0["uuid"])
            cases, extension = diagnostic.continuation(fresh, cohort, previous)
            old_cases, history = diagnostic.continuation(old, cohort, previous)
            self.assertEqual(cases, old_cases)
            self.assertEqual(proof.phases(cases, extension), proof.phases(cases, history))
            self.assertEqual(len(proof.phases(cases, extension)), count)
            self.assertEqual(extension["protocol"]["retained_hardware"]["uuid"], followup.CUDA_UUID)
            self.assertEqual(extension["protocol"]["fresh_hardware"], gpu0)
            self.assertEqual(extension["protocol"]["measurement_source"], diagnostic.GPU0_PROTOCOL_SOURCE)
            self.assertNotIn("fresh_hardware", history["protocol"])
            for case in cases:
                saved = extension["retained"][case]
                gpu1_adapter = saved["dedicated_adapter"]
                adapter = dict(gpu1_adapter, cuda_uuid=gpu0["uuid"], luid="luid_0x00000000_0x00001234")
                proof.check_followup_device(extension, gpu0, adapter)
                for changed in (dict(gpu0, index="1"), dict(gpu0, driver_version="other"),
                                dict(gpu0, name="other"), dict(gpu0, **{"memory.total": "1000"})):
                    with self.assertRaises(ValueError):
                        proof.check_followup_device(extension, changed, adapter)
                with self.assertRaises(ValueError):
                    proof.check_followup_device(history, gpu0, adapter)
                with self.assertRaises(ValueError):
                    proof.check_followup_device(extension, gpu0, gpu1_adapter)
                observations = {}
                for _, _, role, calls in proof.schedule(case, extension):
                    observations[role] = {"initial": {"sha256": saved["initial_sha256"]},
                        "terminal": {"sha256": saved["terminal_sha256"]}, "trace": [], "trace_sha256": saved["trace_sha256"],
                        "samples": [{"elapsed_ns": 100, "outer_steps": 4}] * calls, "peaks": {"pool": 101},
                        "cuda_process": {"lower": 50, "upper": 200, "basis": "wddm-bounded",
                                         "dedicated": {"adapter": adapter, "sampled_peak_bytes": 100}}}
                result = proof.compare_case(case, extension, observations, adapter=adapter)
                self.assertTrue(result["accepted"])
                self.assertEqual(result["fresh_cuda_uuid"], gpu0["uuid"])
                self.assertEqual(result["retained_cuda_uuid"], followup.CUDA_UUID)
                observations[role]["cuda_process"]["dedicated"]["adapter"] = dict(adapter, luid=gpu1_adapter["luid"])
                with self.assertRaises(ValueError):
                    proof.compare_case(case, extension, observations, adapter=adapter)
            with self.assertRaises(ValueError):
                self.binding(cohort, name=fresh, uuid=followup.CUDA_UUID)
            with self.assertRaises(ValueError):
                self.binding(cohort, name=old, uuid=gpu0["uuid"])
        with self.assertRaises(ValueError):
            diagnostic.continuation(diagnostic.GPU0_MEMORY, proof.cohorts.WAN, previous)

    def test_gpu0_role_refuses_gpu1_admission_even_with_a_matching_policy_hash(self):
        from scripts.tests.test_wan_tier1_comparison import request
        with TemporaryDirectory() as directory:
            raw_policy = request("candle-cuda", proof.BASE)["policy"]
            raw_policy["cudaDeviceUuid"] = diagnostic.GPU0_HARDWARE["uuid"]
            policy = proof.read_policy(Path(directory), "policy.json", raw_policy)
        admitted = proof.runtime_guarded_admission(policy)
        admitted.update(hostAvailableBytes=100 * proof.GIB, gpuAvailableBytes=50 * proof.GIB)
        guard = {"returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 100, "admission": admitted}
        observed = {"result": {}, "initial_le_f32_hex": "", "terminal_le_f32_hex": ""}
        with patch.object(proof.evidence, "capture", return_value={"samples": [{"elapsed_ns": 100}]}), \
             patch.object(proof.proof, "verify_execution", return_value={}):
            for delta in ({"cudaDeviceUuid": followup.CUDA_UUID}, {"pollMillis": policy.poll_millis + 1},
                          {"backend": "linux-cuda"}):
                with self.assertRaisesRegex(ValueError, "GPU0 child admission"):
                    proof.evaluate(observed, "cuda-ti2v5b-t2v-heun", "candle-cuda", policy,
                                   dict(proof.SETTINGS, campaign=diagnostic.GPU0_TIMING), "",
                                   dict(guard, admission=dict(admitted, **delta)), {})

    def test_gpu0_release_replays_both_supported_producer_outcomes_on_the_claimed_card(self):
        from scripts.ci.model_proof_resources import SHARED_CUDA_HEADROOM_ACTIVITY
        card = {"index": 0, "total_mib": 97887, "used_mib": 97100, "free_mib": 787, "utilization_percent": 9}
        rows = [{"sample_number": i, "selected_card": card, "predicates": {"unique_uuid": True},
                 "selected_index_matches_admission": True, "post_owned_exit_primary_metadata_valid": True}
                for i in range(1, 6)]
        activity = {"status": "retained-existing-samples", "activity_profile": SHARED_CUDA_HEADROOM_ACTIVITY,
                    "admission_headroom_is_diagnostic_only": True, "samples": rows, "sample_count": len(rows)}
        claim = {"device_uuid": diagnostic.GPU0_HARDWARE["uuid"], "physical_index": 0,
                 "activity_profile": SHARED_CUDA_HEADROOM_ACTIVITY, "admission_baseline": {"admitted_total_mib": 97887},
                 "release_metadata_verification": {"status": "verified"}, "release_utilization_percent": [9] * 5,
                 "release_background_activity": activity}
        proof.verify_gpu0_release(claim)
        partial = dict(activity, samples=rows[:2], sample_count=2)
        with self.assertRaises(ValueError):
            proof.verify_gpu0_release(dict(claim, release_background_activity=partial))
        claim.pop("release_utilization_percent")
        claim.update(release_background_activity=partial, release_metadata_verification={
            "status": "inconclusive-driver-query-timing", "error": "driver timeout", "diagnostic": partial})
        proof.verify_gpu0_release(claim)
        for replacement in (dict(rows[0], selected_card=dict(card, index=1)),
                            dict(rows[0], post_owned_exit_primary_metadata_valid=False)):
            with self.assertRaises(ValueError):
                proof.verify_gpu0_release(dict(claim, release_background_activity=dict(partial, samples=[replacement, rows[1]])))
        with self.assertRaises(ValueError):
            proof.verify_gpu0_release(dict(claim, release_metadata_verification={"status": "inconclusive-driver-query-timing",
                "error": "driver timeout", "diagnostic": activity}))

    def test_native_conditioning_is_complete_disjoint_and_within_the_owned_child(self):
        for case, count in diagnostic.TIMING_CALLS.items():
            row = {"conditioning_ns": [20] * count, "conditioning_end_ns": 21 * count,
                   "measured_start_ns": 21 * count + 1, "measured_end_ns": 31 * count + 2}
            measured = [{"elapsed_ns": 10}] * count
            diagnostic.verify_conditioning(row, measured, case, 1)
            for delta in ({"conditioning_ns": [20] * (count - 1)}, {"conditioning_ns": [22] * count},
                          {"measured_start_ns": 1}, {"measured_end_ns": 30 * count},
                          {"measured_end_ns": 2_000_000_000}, {"conditioning_ns": [True] * count}):
                with self.assertRaises(ValueError):
                    diagnostic.verify_conditioning(dict(row, **delta), measured, case, 1)

    def test_dedicated_only_retry_is_immediate_fresh_and_keeps_owned_pid_identity(self):
        from scripts.ci import wddm_process_memory as w
        from scripts.ci.scail_plural import wddm_retry
        _, extension = self.binding(proof.cohorts.MOCHI_SVD)
        adapter = extension["retained"]["cuda-svd-euler-cfg"]["dedicated_adapter"]
        policy = SimpleNamespace(poll_millis=125)
        for stamp, missing in ((110, 0), (104, 1), (100, 1)):
            with self.subTest(stamp=stamp), patch.object(w, "_query") as query, \
                 patch.object(wddm_retry.time, "monotonic_ns", side_effect=[100, 150, 160, 165, 170]), \
                 patch.object(wddm_retry.time, "sleep", side_effect=AssertionError("delayed retry")), \
                 patch.object(wddm_retry, "local_filetime", side_effect=[105, 115]):
                job = MagicMock()
                job.members.return_value = {42}
                job.root_identity.return_value = (99, False)
                query.return_value.__enter__.return_value.side_effect = [[100, 10], None, [stamp, 10]]
                cuda = SimpleNamespace(dedicated=w.DedicatedSamples(adapter, adapter["cuda_uuid"]))
                monitor = diagnostic.DedicatedRetryObserver(cuda, policy)
                for _ in range(2):
                    cuda.dedicated.poll(job, cuda_active=True)
                raw = cuda.dedicated.record
                observation = monitor.fit_observation()
                evidence = dict(fit_observation=observation, pid=42, cuda_attribution={"wddm_dedicated": raw})
                diagnostic.verify_retry(evidence, policy, adapter)
                self.assertEqual(raw["missing_active_polls"], missing)
                self.assertEqual(observation["reason"], diagnostic.RETRY_REASON)
                for change in ({"pid": 24}, {"created_filetime": 111}, {"adapter": {}}, {"reason": "other"}):
                    with self.assertRaises(ValueError):
                        diagnostic.verify_retry(dict(evidence, fit_observation=dict(observation, **change)), policy, adapter)
                cuda.dedicated.close()

    def test_exact_adjacent_schedule_and_retained_sources_across_all_cuda_cohorts(self):
        recovered, unresolved, original = 0, 0, 0
        for cohort, nroles in ((proof.cohorts.WAN, 40), (proof.cohorts.MOCHI_SVD, 52), (proof.cohorts.SCAIL, 2)):
            cases, extension = self.binding(cohort)
            self.assertEqual(len(proof.phases(cases, extension)), nroles)
            for case, retained in extension["retained"].items():
                order = list(proof.schedule(case, extension))
                if not order:
                    self.assertNotIn(case, cases)
                timing = [r for r in order if "-timing-" in r[2]]
                if timing:
                    self.assertEqual(len(timing), 40)
                    self.assertEqual({r[3] for r in timing}, {diagnostic.TIMING_CALLS[case]})
                    self.assertEqual([r[1] for r in timing[::2]].count("head"), 10)
                memory = [r for r in order if "-memory-" in r[2]]
                self.assertEqual([r[0] for r in memory[::2]], list(diagnostic.MEMORY.get(case, ())))
                self.assertTrue(all(r[3] == 1 for r in memory))
                for a, b in zip(order[::2], order[1::2], strict=True):
                    self.assertEqual((a[0], a[3]), (b[0], b[3]))
                    self.assertEqual((a[1], b[1]), ("base", "head") if a[0] % 2 else ("head", "base"))
                for slot in retained["dedicated_metrics"]["pairs"]:
                    recovered += slot["source"] == "follow-up" and slot["verdict"] == "pass"
                    unresolved += slot["verdict"] == "inconclusive"
                    original += slot["source"] == "retained"
        self.assertEqual((recovered, unresolved, original), (10, 7, 18))

    def test_manifest_and_actual_handle_source_are_frozen(self):
        for path in [diagnostic.MANIFEST, *diagnostic.ROOT_HANDLE_SOURCE]:
            def tracked(head, name):
                return b"changed" if name == path else (proof.ROOT / name).read_bytes()
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.binding(proof.cohorts.WAN, tracked)

    def test_twenty_pair_interval_and_all_frozen_calls(self):
        for calls in diagnostic.TIMING_CALLS.values():
            def measure(ratios):
                base = [{"samples": [{"elapsed_ns": 100}] * calls} for _ in range(20)]
                head = [{"samples": [{"elapsed_ns": n}] * calls} for n in ratios]
                return followup.timing(base, head, 1.02, pairs=20, calls=calls, ranks=(6, 15))
            self.assertEqual(measure([102] * 15 + [110] * 5)["verdict"], "pass")
            r = measure([100] * 14 + [110] * 6)
            self.assertEqual((r["median"], r["interval"], r["verdict"]), (1, [1, 1.1], "inconclusive"))
            self.assertEqual(measure([100] * 5 + [103] * 15)["verdict"], "fail")
        self.assertEqual(1 - 2 * sum(math.comb(20, k) for k in range(6)) / 2**20, 0.95861053466796875)

    def test_targeted_results_combine_only_the_elected_domains(self):
        cases, extension = self.binding(proof.cohorts.SCAIL)
        self.assertEqual(cases, ["cuda-scail2-replacement-dpmpp"])
        case = cases[0]
        retained = extension["retained"][case]
        role = {"initial": {"sha256": retained["initial_sha256"]}, "terminal": {"sha256": retained["terminal_sha256"]},
                "trace_sha256": retained["trace_sha256"], "trace": [], "peaks": {"pool": 100},
                "samples": [{"elapsed_ns": 500, "outer_steps": 8}], "cuda_process": {"lower": 50, "upper": 200,
                "basis": "wddm-bounded", "dedicated": {"adapter": retained["dedicated_adapter"], "sampled_peak_bytes": 100}}}
        roles = {key: deepcopy(role) for key in proof.phases(cases, extension)}
        result = proof.compare_case(case, extension, roles)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["timing"]["source"], "follow-up")
        self.assertEqual(result["dedicated_process_memory"]["pairs"][3]["source"], "targeted-diagnostic")
        self.assertEqual(result["followup_receipt_sha256"], retained["followup_receipt_sha256"])
        roles[case + "-memory-p4-head"]["cuda_process"]["dedicated"]["sampled_peak_bytes"] = None
        self.assertFalse(proof.compare_case(case, extension, roles)["accepted"])

    def test_inconclusive_job_requires_an_ordered_prefix_before_reusing_completed_slots(self):
        cases, extension = self.binding(proof.cohorts.SCAIL)
        contract = {"cases": cases, "followup": extension}
        phases = proof.phases(cases, extension)
        roles = [{"phase": phase, "start_ns": 10 + 10 * i, "end_ns": 16 + 10 * i}
                 for i, phase in enumerate(phases)]
        proof.verify_execution_order(roles[:1], contract, complete=False)
        for bad in (roles[1:], roles[::-1], [roles[0], dict(roles[1], start_ns=15)]):
            with self.assertRaises(ValueError):
                proof.verify_execution_order(bad, contract, complete=False)

    def test_operational_failures_keep_raw_windows_but_only_clean_completed_miss_can_be_reused(self):
        from scripts.tests.test_wan_tier1_comparison import request
        cases, extension = self.binding(proof.cohorts.SCAIL)
        case = cases[0]
        source = extension["retained"][case]
        req = dict(request("candle-cuda", proof.cohorts.VIDEO_BASE), selection=proof.cohorts.SCAIL,
                   followup=diagnostic.NAME)
        req["policy"]["cudaDeviceUuid"] = followup.CUDA_UUID
        contract = dict(request=req, backend="candle-cuda", cases=cases, followup=extension,
                        fixture_contracts={case: {}}, fixture_source_sha256={"base": "hash", "head": "hash"},
                        claim_budget_seconds=572.125, release_reserve_seconds=32.125)
        for issue in (None, "changed case fixture", "changed final fixture", "release failed", "missing log", "cleanup failed"):
            with self.subTest(issue=issue), TemporaryDirectory() as tmp, ExitStack() as patches:
                root = Path(tmp)
                args = SimpleNamespace(output=root / "evidence")
                patches.enter_context(patch.dict(os.environ, RUNNER_TEMP=tmp, RUNNER_NAME="runner",
                    MODEL_PROOF_RUNNER_ENVIRONMENT="self-hosted", GITHUB_RUN_ID="7", GITHUB_RUN_ATTEMPT="1",
                    GITHUB_ENV=str(root / "bindings")))
                def mock(module, method, **kw):
                    return patches.enter_context(patch.object(module, method, **kw))
                mock(proof.proof, "require_host")
                mock(proof.paired, "hardware", return_value=extension["prior_hardware"])
                mock(proof, "cuda_adapter", return_value=source["dedicated_adapter"])
                mock(proof.cohorts, "fixture")
                mock(proof, "inspect_fixture", side_effect=[source["fixture"],
                    {} if issue == "changed case fixture" else source["fixture"],
                    {} if issue == "changed final fixture" else source["fixture"]])
                mock(proof, "build", return_value=(root, {"scail2": root / "binary"}, {}, {"fixture_source_sha256": "hash"}))
                mock(proof, "execution_claim", return_value=nullcontext({"post_execution_verified": issue != "release failed"}))
                def simulate(binary, cwd, out, *unused, **kwargs):
                    out.mkdir()
                    for name in ("stdout.log", "stderr.log", "supervisor.json", "cuda-pool.json"):
                        if issue != "missing log" or name != "stderr.log":
                            (out / name).write_text("{}")
                mock(proof.paired, "invoke", side_effect=simulate)
                mock(proof, "read_observed", return_value={"raw": True})
                evaluation = mock(proof, "evaluate", return_value={"evaluated": True})
                mock(proof, "compare_case", return_value={"accepted": False, "initial_noise": "pass", "parity": "pass"})
                if issue == "cleanup failed":
                    mock(proof.paired, "remove_work", side_effect=OSError("failed cleanup"))
                with self.assertRaises((ValueError, OSError)):
                    proof.execute(args, contract)
                receipt = json.loads((args.output / "receipt.json").read_bytes())
                self.assertEqual(len(receipt["observations"]), 2)
                self.assertEqual(receipt["eligible_for_domain_reuse"], issue is None)
                self.assertEqual(bool(receipt["comparisons"]), issue is None)
                provenance = mock(proof, "check_complete_provenance")
                def verify(row):
                    proof.proof.write_json(args.output / "receipt.json", row)
                    with patch.dict(os.environ, VIDEO_PAIR_RECEIPT_SHA256=proof.sha(args.output / "receipt.json")):
                        proof.verify(args, contract)
                evaluation.reset_mock()
                verify(receipt)
                self.assertEqual(provenance.call_count, int(issue is None))
                self.assertEqual(evaluation.call_count, 2 if issue is None else 0)
                if issue is None:
                    for change in ({"failure": "other"}, {"log_errors": []}, {"cleanup_failure": "failed"},
                                   {"final_fixtures": {}}, {"eligible_for_domain_reuse": False}):
                        with self.assertRaises(ValueError):
                            verify(dict(receipt, **change))

    def test_selected_card_clock_samples_are_outside_each_timing_role(self):
        uuid = followup.CUDA_UUID
        with patch("scripts.media_campaign_windows.trusted_nvidia_smi", return_value="trusted-smi"), \
             patch.object(diagnostic.time, "monotonic_ns", side_effect=[100, 300]), \
             patch.object(subprocess, "check_output", side_effect=[f"{uuid}, 600, 1000, P2, 54, 7\n",
                                                                  subprocess.TimeoutExpired("trusted-smi", 2)]) as read:
            before = diagnostic.clock_snapshot(uuid)
            start = before["monotonic_ns"] + 1
            after = diagnostic.clock_snapshot(uuid)
            order = {"start_ns": start, "end_ns": after["monotonic_ns"],
                     "clock_diagnostics": {"before": before, "after": after}}
            diagnostic.verify_clocks(order, uuid)
            self.assertEqual(before["values"]["clocks.current.sm"], "600")
            self.assertEqual(after["status"], "TimeoutExpired")
            self.assertIn("--id=" + uuid, read.call_args.args[0])
            self.assertEqual(read.call_args.kwargs["timeout"], 2)
            after["monotonic_ns"] = start - 1
            with self.assertRaises(ValueError):
                diagnostic.verify_clocks(order, uuid)
