"""Weightless closure tests for the SC-20684 real-weight campaign launcher."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "sc20684_krea_realtime_campaign.py"
SPEC = importlib.util.spec_from_file_location("sc20684_campaign", SCRIPT)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

SOURCE = {
    "repositoryHead": "a" * 40,
    "files": {path: "b" * 64 for path in campaign.SOURCE_FILES},
}
MODEL = {
    "repository": campaign.MODEL_REPOSITORY,
    "revision": campaign.MODEL_REVISION,
    "variant": "q4",
    "configSha256": "c" * 64,
    "files": {
        name: {"size": 1, "mtimeNs": 1}
        for name in ("dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
    },
}


def observation(mode: str, tier: str, run_id: str) -> dict:
    return {
        "schemaVersion": 1,
        "producer": "mlx-gen-krea-realtime/sc20684",
        "runId": run_id,
        "case": {"mode": mode, "cacheTier": tier},
        "source": SOURCE,
        "model": MODEL,
        "toolchain": {
            "os": "Darwin 25.0",
            "arch": "arm64",
            "rustc": "rustc 1.90",
            "cargo": "cargo 1.90",
            "mlx": "0.30.0",
            "metalDevice": "Apple M-series",
        },
        "geometry": {
            "batch": 1,
            "heads": 40,
            "queryTokens": 8,
            "keyTokens": 16,
            "headDim": 128,
            "groupSize": 64,
            "mask": "block-causal",
        },
        "compiledHandle": {
            "identity": f"sc20684/krea-packed-affine-{tier}-d128-g64-v1",
            "retainedBytes": 128,
            "compiled": True,
            "acceptedDispatches": 40,
        },
        "bytes": {
            "persistent": 4096,
            "retainedHandle": 128,
            "boundedScratch": 4096,
            "denseWindow": 0,
            "scoreMatrix": 0,
        },
        "timing": {"label": "product-median", "wallMs": 4.0, "compileMs": 1.0, "dispatchMs": 2.0},
        "parity": {"status": "pass", "candidateTier": tier, "maxAbsError": 0.01, "tolerance": 0.1},
        "quality": {"status": "pass", "candidateTier": tier, "metric": "clip-paired", "acknowledged": True},
        "fallback": {"count": 0, "reason": None},
        "cancellation": {"status": "pass", "requests": 1, "partialStateMutation": False, "scratchReleased": True},
        "output": {"status": "generated", "sha256": "d" * 64},
    }


class KreaRealtimeCampaignTests(unittest.TestCase):
    def validate(self, mode: str = "t2v", tier: str = "q8", run_id: str = "run") -> dict:
        return campaign._validate_observation(
            observation(mode, tier, run_id),
            expected_mode=mode,
            expected_tier=tier,
            run_id=run_id,
            expected_source=SOURCE,
            expected_snapshot=MODEL,
        )

    def complete_rows(self) -> list[dict]:
        rows = []
        for number, (mode, tier) in enumerate(campaign.CASES):
            run_id = f"run-{number}"
            rows.append({"mode": mode, "cacheTier": tier, "launcherElapsedNs": 1, "observation": self.validate(mode, tier, run_id)})
        return rows

    def test_help_is_available_without_weights_or_accelerator(self) -> None:
        import subprocess
        import sys

        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--product-command", result.stdout)
        self.assertIn("SC20684_KREA_PROVIDER_OBSERVATION", result.stdout)

    def test_complete_matrix_closes_only_after_all_modes_and_tiers(self) -> None:
        rows = self.complete_rows()
        campaign.validate_matrix(rows)
        with self.assertRaisesRegex(campaign.CampaignError, "incomplete"):
            campaign.validate_matrix(rows[:-1])
        malformed = [*rows]
        malformed[-1] = {**malformed[-1], "mode": "t2v", "cacheTier": "q8"}
        with self.assertRaisesRegex(campaign.CampaignError, "exactly once"):
            campaign.validate_matrix(malformed)

    def test_identity_drift_and_dense_bytes_are_refused(self) -> None:
        row = observation("t2v", "q8", "run")
        row["source"] = {**SOURCE, "repositoryHead": "e" * 40}
        with self.assertRaisesRegex(campaign.CampaignError, "source identity drift"):
            self.validate_from(row)
        row = observation("t2v", "q8", "run")
        row["bytes"]["denseWindow"] = 1
        with self.assertRaisesRegex(campaign.CampaignError, "dense window"):
            self.validate_from(row)

    def test_missing_or_malformed_rows_fail_closed(self) -> None:
        row = observation("i2v", "q4", "run")
        del row["cancellation"]
        with self.assertRaisesRegex(campaign.CampaignError, "provider observation fields differ"):
            self.validate_from(row, mode="i2v", tier="q4")
        row = observation("i2v", "q4", "run")
        row["timing"]["dispatchMs"] = -1
        with self.assertRaisesRegex(campaign.CampaignError, "timing.dispatchMs"):
            self.validate_from(row, mode="i2v", tier="q4")

    def test_q8_and_q4_cannot_substitute_one_another(self) -> None:
        row = observation("v2v", "q4", "run")
        row["compiledHandle"]["identity"] = "sc20684/krea-packed-affine-q8-d128-g64-v1"
        with self.assertRaisesRegex(campaign.CampaignError, "launched cache tier"):
            self.validate_from(row, mode="v2v", tier="q4")
        row = observation("v2v", "q4", "run")
        row["quality"]["candidateTier"] = "q8"
        with self.assertRaisesRegex(campaign.CampaignError, "tier-substituted"):
            self.validate_from(row, mode="v2v", tier="q4")

    def test_atomic_publication_never_leaves_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt"
            campaign.publish(output, source=SOURCE, model=MODEL, rows=self.complete_rows())
            self.assertTrue((output / "receipt.json").is_file())
            self.assertTrue((output / "receipt.json.sha256").is_file())
            sealed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "terminal-complete")
            self.assertEqual(len(sealed["matrix"]), 6)
            with self.assertRaisesRegex(campaign.CampaignError, "already exists"):
                campaign.publish(output, source=SOURCE, model=MODEL, rows=self.complete_rows())
            partial = Path(temporary) / "incomplete"
            with self.assertRaisesRegex(campaign.CampaignError, "incomplete"):
                campaign.publish(partial, source=SOURCE, model=MODEL, rows=self.complete_rows()[:-1])
            self.assertFalse(partial.exists())

    def validate_from(self, row: dict, mode: str = "t2v", tier: str = "q8") -> dict:
        return campaign._validate_observation(
            row,
            expected_mode=mode,
            expected_tier=tier,
            run_id="run",
            expected_source=SOURCE,
            expected_snapshot=MODEL,
        )


if __name__ == "__main__":
    unittest.main()
