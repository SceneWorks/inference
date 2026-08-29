"""Regression tests for the SC-20672 provenance validator."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class VeloxquantSourceAuditTests(unittest.TestCase):
    def test_checked_in_audit_is_valid_and_byte_sealed(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_veloxquant_source_audit.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SC-20672 VeloxQuant source audit: OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
