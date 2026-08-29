"""Regression coverage for the SC-20674 fail-closed contract checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/architecture/sc-20674-compressed-kv-contract.json"
CHECKER = ROOT / "scripts/check_compressed_kv_contract.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("compressed_kv_contract", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load_checker()


class CompressedKvContractTests(unittest.TestCase):
    def test_checked_in_contract_is_sealed_and_complete(self) -> None:
        result = subprocess.run([sys.executable, "scripts/check_compressed_kv_contract.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_operation_or_stale_mapping_fails_closed(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["operationRoutes"] = data["operationRoutes"][1:]
        self.assertIn("operationRoutes must cover exactly requiredOperations", CONTRACT.errors_for(data))
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["sourceMappings"] = data["sourceMappings"][1:]
        self.assertIn("sourceMappings must cover exactly the required current seams", CONTRACT.errors_for(data))
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["sourceMappings"][0]["needle"] = "removed cache seam"
        self.assertIn("stale source needle: crates/llm/mlx-llm/src/primitives/kv_cache.rs: removed cache seam", CONTRACT.errors_for(data))


if __name__ == "__main__":
    unittest.main()
