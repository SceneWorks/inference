"""No model compute: terminal pairing decisions and native receipt contracts."""

import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ci import ltx25_tier2_fixture as ltx
from scripts.ci import sga_terminal_comparison as comparison
from scripts.ci import wan_adapter_tier2_fixture as wan

UUID = "GPU-a123bcde-4321-abcd-abcd-0123456789ab"
REQUEST = {"selection": comparison.SELECTION, "profile": "ltx25", "variant": "dev",
           "cuda_device_uuid": UUID, "download_missing": True}


class TerminalTests(unittest.TestCase):
    def test_request_is_one_known_route_on_one_declared_device(self):
        self.assertIs(comparison.profile_for("candle-cuda", REQUEST), ltx)
        for request in (dict(REQUEST, variant="../nested"), dict(REQUEST, profile="../ltx25"),
                        dict(REQUEST, download_missing=1), dict(REQUEST, cuda_device_uuid="0"),
                        dict(REQUEST, unrelated="argv")):
            with self.assertRaises(ValueError):
                comparison.profile_for("candle-cuda", request)
        with self.assertRaises(ValueError):
            comparison.profile_for("mlx", REQUEST)
        request = {key: value for key, value in REQUEST.items() if key != "cuda_device_uuid"}
        with patch.dict(comparison.PROFILES, {("mlx", "ltx25"): "ltx25_tier2_fixture"}):
            self.assertIs(comparison.profile_for("mlx", dict(request, runner="nax")), ltx)
            with self.assertRaises(ValueError):
                comparison.profile_for("mlx", dict(request, runner="nax-primary"))
        mac_limits = comparison.sample_limits(ltx, dict(ltx.policy("dev", UUID), backend="darwin-mlx"))
        self.assertNotIn("cuda-pool.json", mac_limits)

    def test_pair_order_and_exact_decisions(self):
        base = {"identity": {"v": "same", "a": "same"}, "timing": {"e2e_ns": 1000}, "peaks": {"device": 1000}}
        samples = {name: base for name, _ in comparison.SAMPLES}
        self.assertEqual(comparison.summarize(samples)["status"], "pass")
        self.assertEqual(list(samples)[:6], ["warmup-base", "warmup-head", "pair-1-base", "pair-1-head", "pair-2-head", "pair-2-base"])
        samples["pair-4-head"] = dict(base, timing={"e2e_ns": 1021})
        self.assertEqual(comparison.summarize(samples)["status"], "inconclusive")
        samples["pair-4-head"] = dict(base, peaks={"device": 1021})
        self.assertEqual(comparison.summarize(samples)["status"], "fail")
        samples["pair-4-head"] = dict(base, identity={"v": "drift", "a": "same"})
        with self.assertRaisesRegex(ValueError, "output"):
            comparison.summarize(samples)
        del samples["pair-4-head"]
        with self.assertRaisesRegex(ValueError, "every frozen sample"):
            comparison.summarize(samples)

    def test_ltx_stages_adapter_route_and_decoded_content(self):
        for variant, stage1 in ltx.STAGE1.items():
            total = stage1 + 3
            case = ltx.CASE[variant]
            adapter_count = 1660 if variant == "dev" else 0
            row = {"case": case, "cuda_device_uuid": UUID, "e2e_ns": 1000,
                   "progress": [[index, total] for index in range(1, total + 1)],
                   "video": {"size": [512, 512], "frames": 17, "fps": 24, "rgb_sha256": "1" * 64},
                   "audio": {"channels": 2, "sample_rate": 24000, "samples": 48000, "f32_sha256": "2" * 64}}
            base, refine = ltx.SCHEDULE[variant], ltx.SCHEDULE["refinement"]
            log = (f"SGA_LTX_ADAPTER {adapter_count} 0\nSGA_LTX_STAGE 0 {stage1} 200 {base}\n"
                   f"SGA_LTX_STAGE 1 3 100 {refine}\nSGA_LTX_REAL_COMPLETE {case}\n")
            ltx.preflight(f"SGA_LTX_NATIVE_CASE {case} {base} {refine}\n", variant)
            with TemporaryDirectory() as directory:
                result = Path(directory) / "result.json"
                result.write_text(json.dumps(row), encoding="utf-8")
                observed = ltx.observe(log, result, UUID, variant)
                self.assertEqual(observed["identity"]["steps"], [stage1, 3])
                self.assertEqual(observed["timing"]["denoise_ns_per_outer_step"], 300 / total)
                for malformed in (log.replace(" 3 100", " 2 100"), log.replace(" 0\n", " 1\n", 1), log + log):
                    with self.assertRaises(ValueError):
                        ltx.observe(malformed, result, UUID, variant)
                result.write_text(json.dumps(dict(row, progress=[])), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ltx.observe(log, result, UUID, variant)

    def test_wan_variant_and_packed_model_are_bound_to_backend(self):
        cuda = dict(REQUEST, profile="wan", variant="a14b-t2v-candle")
        mlx = {key: value for key, value in cuda.items() if key != "cuda_device_uuid"}
        mlx = dict(mlx, variant="a14b-t2v-mlx", runner="nax")
        self.assertIs(comparison.profile_for("candle-cuda", cuda), wan)
        self.assertIs(comparison.profile_for("mlx", mlx), wan)
        for backend, request, wrong in (("candle-cuda", cuda, mlx), ("mlx", mlx, cuda)):
            with self.assertRaises(ValueError):
                comparison.profile_for(backend, dict(request, variant=wrong["variant"]))
        self.assertEqual(wan.policy(cuda["variant"], UUID)["deadlineSeconds"], 240)
        self.assertNotEqual(wan.contract(cuda["variant"])["sources"]["model"],
                            wan.contract(mlx["variant"])["sources"]["model"])

    def test_wan_candle_receipt_requires_both_experts_to_install_adapters(self):
        variant = "a14b-t2v-candle"
        trace = [f"SGA_WAN_OUTER_STEP {i} {struct.unpack('<Q', struct.pack('<d', t))[0]} 1000"
                 for i, t in enumerate((1000.0, 980.0, 925.0, 807.0), 1)]
        row = {"device": {"kind": "candle-cuda", "index": 0}, "cuda_device_uuid": UUID,
               "end_to_end_ns": 12000, "frame_rgb8_sha256": [f"{i:064x}" for i in range(17)],
               "progress": ["renderer", "1/4", "2/4", "3/4", "renderer", "4/4", "renderer", "decode"]}
        with TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps(row), encoding="utf-8")
            log = "\n".join(["SGA_WAN_EXPERT_ADAPTERS High 8", *trace[:3], "SGA_WAN_EXPERT_ADAPTERS Low 8",
                             trace[3], "SGA_WAN_REAL_COMPLETE " + wan.CASE[variant]])
            sample = wan.observe(log, result, UUID, variant)
            self.assertEqual(sample["identity"]["adapters"], [{"expert": "High", "applied": 8},
                                                              {"expert": "Low", "applied": 8}])
            self.assertEqual(sample["timing"]["outer_step_ns"], 1000)
            for wrong in (log.replace("Low 8", "Low 0"), log.replace("Low 8", "High 8"),
                          log.replace("SGA_WAN_EXPERT_ADAPTERS Low 8\n" + trace[3],
                                      trace[3] + "\nSGA_WAN_EXPERT_ADAPTERS Low 8"),
                          log.replace(trace[-1].split()[2], trace[0].split()[2])):
                with self.assertRaises(ValueError):
                    wan.observe(wrong, result, UUID, variant)


if __name__ == "__main__":
    unittest.main()
