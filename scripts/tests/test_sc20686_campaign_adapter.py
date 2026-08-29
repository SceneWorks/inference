import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ADAPTER = Path(__file__).parents[1] / "sc20686_campaign_adapter.py"
GEOMETRY = {"resolution": "512x512", "reference_count": 1, "frames": 1,
            "prompt": "fake", "guidance": 1.0, "layers": 1, "heads": 2,
            "head_dimension": 64, "sq": 1, "skv": 1024, "dtype": "bf16",
            "mask": "causal", "rope": "native"}


class CampaignAdapterTests(unittest.TestCase):
    def test_fake_both_families_emit_sealed_rows(self):
        for family in ("flux2-klein", "wan"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "snapshot"
                root.mkdir(); (root / "config.json").write_text(
                    json.dumps({"source_ref": "frozen-test", "sc20686_geometry": GEOMETRY}),
                    encoding="utf-8")
                output = Path(directory) / "decision.json"
                command = [sys.executable, str(ADAPTER), "--campaign", "--fake",
                           "--family", family, "--variant", "test", "--snapshot", str(root),
                           "--output", str(output)]
                completed = subprocess.run(command, check=False, text=True,
                                           encoding="utf-8", capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads(output.read_text())["decisions"][family]["decision"], "go")
                raw = output.with_suffix(".raw.json")
                row = json.loads(raw.read_text())[0]
                self.assertEqual(row["producer"], "sc20686-campaign-adapter-v1")
                self.assertNotEqual(row["raw_receipt_sha256"], "" * 64)

    def test_normal_invocation_and_missing_hook_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "config.json").write_text("{}", encoding="utf-8")
            base = [sys.executable, str(ADAPTER), "--family", "wan", "--variant", "x",
                    "--snapshot", str(root), "--output", str(root / "o.json")]
            self.assertNotEqual(subprocess.run(base, check=False).returncode, 0)
            command = base + ["--campaign"]
            self.assertNotEqual(subprocess.run(command, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
