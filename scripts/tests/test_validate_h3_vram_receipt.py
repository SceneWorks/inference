import importlib.util
import json
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "ci" / "validate_h3_vram_receipt.py"
SPEC = importlib.util.spec_from_file_location("h3_receipt", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def receipt(**overrides):
    value = {
        "model": "minimax_h3", "tier": "q4", "peakOwner": "denoise",
        "peakGb": 10.0, "trueMemHighGib": 10.0, "denoiseMemHighGib": 10.0,
        "decodeMemHighGib": 1.0, "preDecodeGb": 9.0, "preDecodeAbsGb": 9.0,
        "decodeGb": 1.0, "steadyGb": 1.0, "loadPeakGb": 0.0, "baselineGb": 0.1,
        "middleFrameStd": 4.0, "seconds": 1.0, "vramMeasuredPixels": 1,
        "frames": 1, "width": 1, "height": 1, "steps": 1,
    }
    value.update(overrides)
    return MODULE.PREFIX + json.dumps(value)


class H3ReceiptTests(unittest.TestCase):
    def test_accepts_one_complete_tier_local_receipt(self):
        self.assertEqual(MODULE.parse_receipt([receipt()], "q4")["tier"], "q4")

    def test_rejects_duplicate_malformed_and_split_receipts(self):
        cases = {
            "duplicate": [receipt(), receipt()],
            "malformed": [MODULE.PREFIX + "{"],
            "wrong tier": [receipt(tier="q8")],
            "illegal owner": [receipt(peakOwner="unknown")],
            "contaminated baseline": [receipt(baselineGb=1.0)],
            "duplicate JSON member": [MODULE.PREFIX + '{"model":"minimax_h3","model":"minimax_h3"}'],
            "split fields": [MODULE.PREFIX + json.dumps({"tier": "q4", "peakGb": 1}), MODULE.PREFIX + json.dumps({"baselineGb": 0.1})],
        }
        for name, lines in cases.items():
            with self.subTest(name=name):
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    MODULE.parse_receipt(lines, "q4")
