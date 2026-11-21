"""No model compute: terminal pairing decisions and LTX receipt contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.ci import ltx25_tier2_fixture as ltx
from scripts.ci import sga_terminal_comparison as comparison

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


if __name__ == "__main__":
    unittest.main()
