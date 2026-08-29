"""Weightless regression coverage for the sealed SC-20684 source contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check_sc20684_krea_realtime_contract.py"
MANIFEST_NAME = "sc-20684-krea-realtime-compressed-kv-contract.json"
ADR_NAME = "SC_20684_KREA_REALTIME_COMPRESSED_KV_SOURCE_MAP.md"


def write_sealed_manifest(root: Path, data: dict) -> None:
    docs = root / "docs/architecture"
    raw = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    manifest = docs / MANIFEST_NAME
    manifest.write_bytes(raw)
    (docs / f"{MANIFEST_NAME}.sha256").write_text(
        f"{hashlib.sha256(raw).hexdigest()}  {MANIFEST_NAME}\n", encoding="utf-8"
    )


class KreaRealtimeContractTests(unittest.TestCase):
    def copied_contract_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        docs = root / "docs/architecture"
        docs.mkdir(parents=True)
        for name in (MANIFEST_NAME, f"{MANIFEST_NAME}.sha256", ADR_NAME):
            shutil.copy2(ROOT / "docs/architecture" / name, docs / name)
        return temp, root

    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root), "--source-root", str(ROOT)],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )

    def test_checked_in_contract_is_sealed_and_current(self) -> None:
        result = self.run_checker(ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mutated_copy_rejects_mask_gap_even_when_resealed(self) -> None:
        temp, root = self.copied_contract_root()
        self.addCleanup(temp.cleanup)
        manifest = root / "docs/architecture" / MANIFEST_NAME
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["rejectionFallbacks"] = [row for row in data["rejectionFallbacks"] if row["id"] != "mask"]
        write_sealed_manifest(root, data)
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fallback taxonomy must be complete and reasoned", result.stdout)

    def test_mutated_copy_rejects_stale_mapping_and_checksum_drift(self) -> None:
        temp, root = self.copied_contract_root()
        self.addCleanup(temp.cleanup)
        manifest = root / "docs/architecture" / MANIFEST_NAME
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["sourceMappings"][0]["needle"] = "removed current Krea seam"
        write_sealed_manifest(root, data)
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sourceMappings must cover exactly the current Krea seams", result.stdout)
        manifest.write_bytes(manifest.read_bytes() + b" ")
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest checksum mismatch", result.stdout)

    def test_mutated_copy_rejects_missing_device_receipt_axis_even_when_resealed(self) -> None:
        temp, root = self.copied_contract_root()
        self.addCleanup(temp.cleanup)
        manifest = root / "docs/architecture" / MANIFEST_NAME
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["receipt"]["requiredFields"].remove("scoreMatrixBytes")
        write_sealed_manifest(root, data)
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("receipt contract must remain unproduced and complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
