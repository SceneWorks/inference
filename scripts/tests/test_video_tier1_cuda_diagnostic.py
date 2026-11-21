"""Metadata-only checks for the frozen CUDA diagnostic and its evidence provenance."""
from copy import deepcopy
import math
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
