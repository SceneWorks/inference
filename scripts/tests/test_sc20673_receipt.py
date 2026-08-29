import json, subprocess, sys, tempfile, unittest
from pathlib import Path
CHECK = Path(__file__).parents[1] / "check_sc20673_receipt.py"
RECEIPT = Path(__file__).parents[2] / "docs/architecture/receipts/sc-20673-coverage.json"
class ReceiptTests(unittest.TestCase):
    def test_current_receipt_passes(self):
        self.assertEqual(subprocess.run([sys.executable, str(CHECK)]).returncode, 0)
    def test_missing_axis_is_rejected(self):
        j = json.loads(RECEIPT.read_text()); j["axes"].pop("GQA")
        self.assertNotIn("GQA", j["axes"])
        self.assertIn("process_boundary_proxy", j["timing"]["first_dispatch"])
    def test_provenance_and_timing_are_explicit(self):
        j = json.loads(RECEIPT.read_text()); self.assertIn("host", j["provenance"]); self.assertIn("not isolated", j["timing"]["first_dispatch"])
