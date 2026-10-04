"""Regression tests for the SC-20672 provenance validator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/architecture/sc-20672-veloxquant-provenance.json"
CHECKER = ROOT / "scripts/check_veloxquant_source_audit.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("veloxquant_source_audit", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def run_git(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


class VeloxquantSourceAuditTests(unittest.TestCase):
    def test_checked_in_audit_is_valid_and_byte_sealed(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_veloxquant_source_audit.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SC-20672 VeloxQuant source audit: OK", result.stdout)

    def test_source_mappings_do_not_require_the_historical_product_pin(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for mapping in data["localSourceMappings"]:
                source = checkout / mapping["path"]
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"// {mapping['sourceNeedle']}\n", encoding="utf-8")

            self.assertEqual(run_git(["init", "-q"], checkout).returncode, 0)
            self.assertEqual(run_git(["add", "."], checkout).returncode, 0)
            self.assertEqual(
                run_git(
                    [
                        "-c",
                        "user.name=SC-20672 test",
                        "-c",
                        "user.email=sc-20672@example.invalid",
                        "commit",
                        "-qm",
                        "current checkout only",
                    ],
                    checkout,
                ).returncode,
                0,
            )

            historical_pin = data["localProvenance"]["productInferenceRevision"]
            first_path = data["localSourceMappings"][0]["path"]
            self.assertNotEqual(
                run_git(["cat-file", "-e", f"{historical_pin}:{first_path}"], checkout).returncode,
                0,
                "the test checkout must not contain the historical product pin",
            )
            self.assertEqual(VALIDATOR.source_mapping_errors(data, checkout), [])

            missing_needle = checkout / first_path
            missing_needle.write_text("// wrong source\n", encoding="utf-8")
            self.assertEqual(
                VALIDATOR.source_mapping_errors(data, checkout),
                [
                    "local source mapping is missing 'fn update' in checked-out source: "
                    "crates/llm/mlx-llm/src/primitives/kv_cache.rs"
                ],
            )


if __name__ == "__main__":
    unittest.main()
