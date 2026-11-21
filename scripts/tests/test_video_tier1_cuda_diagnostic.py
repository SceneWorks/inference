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
from unittest.mock import patch

from scripts.ci import video_tier1_comparison as proof
from scripts.ci import video_tier1_cuda_diagnostic as diagnostic
from scripts.ci import video_tier1_followup as followup


class TargetedCudaTests(unittest.TestCase):
    def binding(self, selection, tracked=None):
        def read(_head, path):
            return (proof.ROOT / path).read_bytes()
        policy = dict(schemaVersion=1, backend="windows-cuda", deadlineSeconds=180, pollMillis=125,
                      termGraceMillis=2000, hostFreeReserveBytes=8 * proof.GIB, childFootprintCapBytes=20 * proof.GIB,
                      cudaDeviceUuid=followup.CUDA_UUID, gpuFreeReserveBytes=6 * proof.GIB, childGpuCapBytes=12 * proof.GIB,
                      **dict.fromkeys(("stdoutCapBytes", "stderrCapBytes", "eventCapBytes"), 4 * proof.MIB))
        _, _, prior = followup.bind("controller", "candle-cuda", selection, read, policy)
        return diagnostic.bind("controller", "candle-cuda", selection, tracked or read, prior)

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
