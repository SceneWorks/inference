"""Frozen SW-6954 case, schedule and receipt arithmetic. No device or model execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.ci import video_tier1_cohorts as cohorts
from scripts.ci import video_tier1_comparison as proof
from scripts.ci import video_tier1_followup as followup


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (proof.ROOT / followup.MANIFEST).read_bytes()
        cls.manifest = json.loads(cls.raw)

    def binding(self, selection, backend, **overrides):
        policy = {"schemaVersion": 1, "backend": proof.proof.POLICY_BACKENDS[backend], "deadlineSeconds": 180,
                  "pollMillis": 125, "termGraceMillis": 2000, "hostFreeReserveBytes": 8 * proof.GIB,
                  "childFootprintCapBytes": 20 * proof.GIB, "stdoutCapBytes": 4 * proof.MIB,
                  "stderrCapBytes": 4 * proof.MIB, "eventCapBytes": 4 * proof.MIB}
        if backend == "candle-cuda":
            policy.update(cudaDeviceUuid=followup.CUDA_UUID, gpuFreeReserveBytes=6*proof.GIB, childGpuCapBytes=12*proof.GIB)
        policy.update(overrides)
        return followup.bind("test-head", backend, selection, lambda *_: self.raw, policy)

    def test_exact_repair_slots_and_fresh_child_counts(self):
        lanes = ((cohorts.WAN, "mlx", 48), (cohorts.WAN, "candle-cuda", 26),
                 (cohorts.MOCHI_SVD, "candle-cuda", 46), (cohorts.SCAIL, "candle-cuda", 34))
        timing_cases, memory_slots = set(), 0
        for selection, backend, count in lanes:
            cases, head, binding = self.binding(selection, backend)
            self.assertEqual({r["head"] for r in binding["retained"].values()}, {head})
            self.assertEqual(len(proof.phases(cases, binding)), count)
            for case in cases:
                schedule = list(proof.schedule(case, binding))
                for a, b in zip(schedule[::2], schedule[1::2], strict=True):
                    slot, arm, name, repeats = a
                    self.assertEqual((a[0], a[3]), (b[0], b[3]))
                    self.assertEqual((arm, b[1]), ("base", "head") if slot % 2 else ("head", "base"))
                    if repeats == 16:
                        timing_cases.add(case)
                        self.assertIn("-timing-", name)
                    else:
                        memory_slots += 1
                        self.assertEqual(binding["retained"][case]["dedicated_pairs"][slot-1], "inconclusive")
        self.assertEqual(timing_cases, followup.TIMING)
        self.assertEqual(memory_slots, 17)
        with self.assertRaisesRegex(ValueError, "no authorized"):
            self.binding(cohorts.SCAIL, "mlx")
        with self.assertRaisesRegex(ValueError, "exact physical"):
            self.binding(cohorts.WAN, "candle-cuda", childGpuCapBytes=13*proof.GIB)

    def test_every_synchronized_call_contributes_and_interval_uses_frozen_ranks(self):
        def role(last=100):
            return {"samples": [{"elapsed_ns": 100} for _ in range(15)] + [{"elapsed_ns": last}]}
        base = [role() for _ in range(12)]
        head = [role() for _ in range(9)] + [role(180) for _ in range(3)]
        result = followup.timing(base, head, 1.02)
        self.assertEqual(result["interval"], [1., 1.05])
        self.assertEqual(result["median"], 1.)
        self.assertEqual(result["verdict"], "inconclusive")
        head[9] = role()
        self.assertEqual(followup.timing(base, head, 1.02)["verdict"], "pass")
        self.assertEqual(followup.timing(base, [role(180)] * 12, 1.02)["verdict"], "fail")
        with self.assertRaises(ValueError):
            followup.timing(base[:-1], head[:-1], 1.02)
        head[0]["samples"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            followup.timing(base, head, 1.02)

    def test_retained_and_replacement_memory_cannot_mask_a_regression_or_missing_observation(self):
        case = "cuda-mochi-masked-euler"
        historical = self.manifest["cases"][case]
        base = {"initial": {"sha256": historical["initial_sha256"]},
                "terminal": {"sha256": historical["terminal_sha256"]}, "trace": [],
                "trace_sha256": historical["trace_sha256"], "peaks": {"pool": 100},
                "samples": [{"elapsed_ns": 500, "outer_steps": 4}],
                "cuda_process": {"lower": 50, "upper": 200, "basis": "wddm-bounded",
                    "dedicated": {"adapter": historical["dedicated_adapter"], "sampled_peak_bytes": 100}}}
        data = {name: deepcopy(base) for _, _, name, _ in proof.schedule(case, True)}
        def compare():
            return followup.compare(case, historical, data, proof.SETTINGS)
        result = compare()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["timing"], {"source": "retained", **historical["timing"]})
        self.assertEqual([p["source"] for p in result["dedicated_process_memory"]["pairs"]],
                         ["follow-up", "follow-up", "retained", "retained", "follow-up"])
        peak = data[case + "-memory-p1-head"]["cuda_process"]["dedicated"]
        for value, verdict in ((103, "fail"), (None, "inconclusive")):
            peak["sampled_peak_bytes"] = value
            result = compare()
            self.assertFalse(result["accepted"])
            self.assertEqual(result["dedicated_process_memory"]["verdict"], verdict)
        peak["sampled_peak_bytes"] = 100
        data[case + "-memory-p1-head"]["terminal"]["sha256"] = "incorrect"
        self.assertFalse(compare()["accepted"])
        self.assertEqual(compare()["parity"], "fail")
        peak["adapter"] = {}
        with self.assertRaisesRegex(ValueError, "retained adapter"):
            compare()
