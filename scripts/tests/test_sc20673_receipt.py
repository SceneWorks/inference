from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECK = Path(__file__).parents[1] / "check_sc20673_receipt.py"
RECEIPTS = Path(__file__).parents[2] / "docs/architecture/receipts"
FILES = ("sc-20673-coverage.json", "sc-20673-metal-reproduction.json")


def _seal(root: Path, name: str, data: dict) -> None:
    path = root / name
    encoded = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {name}\n", encoding="utf-8"
    )


class Sc20673ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in FILES:
            shutil.copy2(RECEIPTS / name, self.root / name)
            shutil.copy2(RECEIPTS / f"{name}.sha256", self.root / f"{name}.sha256")

    def tearDown(self):
        self.temp.cleanup()

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(CHECK), "--root", str(self.root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def _load(self) -> tuple[dict, dict]:
        coverage = json.loads((self.root / FILES[0]).read_text(encoding="utf-8"))
        receipt = json.loads((self.root / FILES[1]).read_text(encoding="utf-8"))
        return coverage, receipt

    def _seal_pair(self, coverage: dict, receipt: dict) -> None:
        receipt["coverage"] = coverage
        _seal(self.root, FILES[0], coverage)
        _seal(self.root, FILES[1], receipt)

    def test_current_receipt_passes(self):
        self.assertEqual(self._run().returncode, 0)

    def test_checksum_drift_is_rejected(self):
        with (self.root / FILES[0]).open("ab") as handle:
            handle.write(b"\n")
        self.assertNotEqual(self._run().returncode, 0)

    def test_missing_axis_is_rejected_after_resealing(self):
        coverage, receipt = self._load()
        coverage["axes"].pop("GQA")
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_host_version_mismatch_is_rejected(self):
        coverage, receipt = self._load()
        receipt["host"]["mlx"] = "mismatched"
        _seal(self.root, FILES[1], receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_stale_upstream_result_is_rejected_after_resealing(self):
        coverage, receipt = self._load()
        coverage["upstream_results"]["rabitq_decode"]["speedup"][0] += 1.0
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_missing_probe_metric_is_rejected_after_resealing(self):
        coverage, receipt = self._load()
        row = coverage["probe"]["probes"][0]
        row["metrics"].pop("first_eval_compile_and_dispatch_s")
        coverage["probe_results"][row["name"]] = row["metrics"]
        receipt["probe"] = coverage["probe"]
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_nonpositive_physical_bytes_are_rejected_after_resealing(self):
        coverage, receipt = self._load()
        row = coverage["probe"]["probes"][0]
        field = next(iter(row["physical_bytes"]))
        row["physical_bytes"][field] = 0
        coverage["physical_bytes"][row["name"]] = row["physical_bytes"]
        receipt["probe"] = coverage["probe"]
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_positive_but_wrong_physical_bytes_are_rejected_after_resealing(self):
        coverage, receipt = self._load()
        row = coverage["probe"]["probes"][0]
        row["physical_bytes"]["compressed_persistent_bytes"] += 1
        coverage["physical_bytes"][row["name"]] = row["physical_bytes"]
        receipt["probe"] = coverage["probe"]
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_nonfinite_metric_is_rejected_after_resealing(self):
        coverage, receipt = self._load()
        row = coverage["probe"]["probes"][0]
        row["metrics"]["steady_dispatch_sync_median_s"] = float("nan")
        coverage["probe_results"][row["name"]] = row["metrics"]
        receipt["probe"] = coverage["probe"]
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)

    def test_immutable_ref_drift_is_rejected_after_resealing(self):
        coverage, receipt = self._load()
        coverage["provenance"]["upstream_commit"] = "0" * 40
        receipt["upstream"]["commit"] = "0" * 40
        self._seal_pair(coverage, receipt)
        self.assertNotEqual(self._run().returncode, 0)


if __name__ == "__main__":
    unittest.main()
