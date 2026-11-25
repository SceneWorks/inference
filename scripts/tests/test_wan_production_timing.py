"""Offline checks for the single merged-tree CUDA Wan timing comparison."""
from collections import Counter
import json
import os
from unittest import TestCase
from unittest.mock import patch

from scripts.ci import video_tier1_comparison as proof
from scripts.ci import video_tier1_cuda_diagnostic as targeted
from scripts.ci.wan_pair import production_timing as wan


class ProductionTiming(TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((proof.ROOT / proof.followup.MANIFEST).read_bytes())
        cls.extension = wan.bind({"retained": manifest["cases"]}, targeted.GPU0_HARDWARE, 256, 40)
        cls.anchor = cls.extension["retained"][wan.CASE]

    def test_verified_role_order_balances_position_predecessor_and_linear_drift(self):
        roles = list(proof.schedule(wan.CASE, self.extension))
        pairs = []
        for first, second in zip(roles[::2], roles[1::2], strict=True):
            self.assertEqual((first[0], first[3], second[3]), (second[0], 256, 256))
            pairs.append((first[1], second[1]))
        ab, ba = proof.ARMS, proof.ARMS[::-1]
        self.assertEqual(pairs, [ab, ba, ba, ab] * 10)
        self.assertEqual(Counter(zip(pairs, pairs[1:])), {(ab, ba): 10, (ba, ab): 10, (ba, ba): 10, (ab, ab): 9})
        for start in range(0, 40, 4):
            self.assertEqual(sum(i for i in range(4) if pairs[start+i] == ab), 3)
        contract = {"cases": [wan.CASE], "followup": self.extension,
                    "request": {"policy": {"cudaDeviceUuid": targeted.GPU0_HARDWARE["uuid"]}}}
        def windows(extension):
            return [{"phase": phase, "start_ns": 2*i+1, "end_ns": 2*i+2}
                    for i, phase in enumerate(proof.phases([wan.CASE], extension))]
        with patch.object(targeted, "verify_clocks"):
            proof.verify_execution_order(windows(self.extension), contract, complete=True)
            with self.assertRaisesRegex(ValueError, "order"):
                proof.verify_execution_order(windows({"name": targeted.GPU0_TIMING}), contract, complete=True)

    def test_actual_merged_arm_and_unmodified_baseline_are_fixed(self):
        policy = dict(schemaVersion=1, backend="windows-cuda", deadlineSeconds=180, pollMillis=125,
                      termGraceMillis=2000, hostFreeReserveBytes=8*proof.GIB, childFootprintCapBytes=20*proof.GIB,
                      cudaDeviceUuid=targeted.GPU0_HARDWARE["uuid"], gpuFreeReserveBytes=6*proof.GIB,
                      childGpuCapBytes=12*proof.GIB, **dict.fromkeys(("stdoutCapBytes", "stderrCapBytes", "eventCapBytes"), 4*proof.MIB))
        request = {"selection": proof.SELECTION, "base": wan.BASE, "followup": wan.NAME, "policy": policy}
        # PR checkouts need not contain either historical target; exercise both checked ancestry edges
        # and the two actual planned checkouts without relying on the runner's shallow Git history.
        head = "a" * 40
        def tracked(_, path):
            return (proof.ROOT / path).read_bytes()
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT="1"), patch.object(proof.proof, "repository_identity",
              return_value={"tree": "head-tree", "revision": head}), patch.object(proof, "tracked", side_effect=tracked), \
             patch.object(proof, "overlay_hashes", return_value="digest"), patch.object(proof.paired, "git", return_value="tree"), \
             patch.object(proof.subprocess, "run") as git:
            plan = proof.plan("candle-cuda", request, head)
            for ancestor in (wan.BASE, wan.FEATURE):
                git.assert_any_call(["git", "merge-base", "--is-ancestor", ancestor, head], cwd=proof.ROOT, check=True)
            self.assertEqual((plan["base"], plan["head"], plan["cases"]), (wan.BASE, head, [wan.CASE]))
            for updated in (dict(base=proof.BASE), dict(selection=proof.cohorts.MOCHI_SVD),
                            dict(policy=dict(policy, cudaDeviceUuid=proof.followup.CUDA_UUID))):
                with self.assertRaises(ValueError):
                    proof.plan("candle-cuda", dict(request, **updated), head)
            with patch.dict(os.environ, GITHUB_RUN_ATTEMPT="2"), self.assertRaises(ValueError):
                proof.plan("candle-cuda", request, head)

    def test_head_phase_precedes_sync_and_baseline_keeps_native_heun(self):
        # Expand the actual templates the same way as overlay_source's Git-object reads.
        read = lambda _, path: (proof.ROOT / path).read_bytes()
        def git_read(argv, **kwargs):
            return read(None, argv[-1].split(":", 1)[1])
        with patch.object(proof.subprocess, "check_output", side_effect=git_read):
            blobs = [proof.compiled_overlay("controller", "candle-cuda", arm, "wan", "head", self.extension)[1]
                     for arm in proof.ARMS]
        base, head = blobs
        self.assertNotIn(b"dense_curated_phase", base)
        self.assertIn(b'Some("heun"), &request.cancel', base)
        self.assertEqual(head.count(b"dense_curated_phase"), 1)
        prepared = head.index(b"let phase = crate::dense_curated_phase")
        sync = head.index(b"device.synchronize()?;", prepared)
        start = head.index(b"let start = Instant::now();", sync)
        call = head.index(b"let latent = model.denoise(", start)
        end = head.index(b"let elapsed = start.elapsed()", call)
        self.assertIn(b"&request, phase.as_ref()", head[call:end])
        self.assertIn(b"device.synchronize()?;", head[call:end])
        for blob in blobs:
            self.assertIn(b"pair.measure_conditioned(", blob)
        with self.assertRaisesRegex(ValueError, "audited fixture"):
            wan.prepare_overlay("head", head, "0"*64, b"bad")

    def test_fresh_forty_pair_verdict_excludes_historical_prices_and_rejects_bit_or_rank_miss(self):
        adapter = {"cuda_uuid": targeted.GPU0_HARDWARE["uuid"]}
        def observation(arm, slot):
            elapsed = 100 if arm == "base" or slot <= 27 else 106
            return {"initial": {"sha256": self.anchor["initial_sha256"]},
                    "terminal": {"sha256": self.anchor["terminal_sha256"]},
                    "trace_sha256": self.anchor["trace_sha256"],
                    "cuda_process": {"dedicated": {"adapter": adapter}},
                    "samples": [{"elapsed_ns": elapsed, "outer_steps": 4}] * 256}
        rows = {name: observation(arm, slot) for slot, arm, name, _ in proof.schedule(wan.CASE, self.extension)}
        with patch("scripts.ci.wddm_process_memory.validate_adapter", return_value=adapter):
            def compare():
                return proof.compare_case(wan.CASE, self.extension, rows, adapter=adapter)
            result = compare()
            self.assertTrue(result["accepted"])
            self.assertNotIn("retained_peaks", result)
            self.assertEqual(result["previous_comparison"]["timing_verdict"], "inconclusive")
            rows[f"{wan.CASE}-timing-p27-head"]["samples"] = rows[f"{wan.CASE}-timing-p28-head"]["samples"]
            self.assertEqual(compare()["timing"]["verdict"], "inconclusive")
            rows[f"{wan.CASE}-timing-p1-head"]["terminal"]["sha256"] = "0"*64
            self.assertEqual(compare()["parity"], "fail")
