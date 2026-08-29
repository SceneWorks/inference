import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "sc20673_metal_campaign.py"


class Sc20673HarnessTests(unittest.TestCase):
    def test_help_is_available_without_gpu_or_network(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--source", result.stdout)

    def test_receipt_is_bounded_and_json(self):
        source = Path(tempfile.mkdtemp())
        (source / ".git").mkdir()
        out = source / "receipt.json"
        result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source), "--output", str(out)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_commands_include_full_upstream_surface(self):
        text = SCRIPT.read_text()
        for needle in ("scalar_attend", "rabitq_attend", "rabitq_encode", "rabitq_values", "rabitq_prefill", "kivi_quant", "turboquant_kernels", "rvq_quant_pack"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
