"""Weightless closure tests for the SC-20684 real-weight campaign launcher."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "sc20684_krea_realtime_campaign.py"
ROOT = SCRIPT.parents[1]
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
    input_by_mode = {
        "t2v": {"kind": "text-only", "frameCount": 0, "vaeEncoding": "none", "v2vStrength": None},
        "i2v": {"kind": "deterministic-gradient-still", "frameCount": 1, "vaeEncoding": "WanVae.encode-mode", "v2vStrength": None},
        "v2v": {"kind": "deterministic-smooth-motion-clip", "frameCount": 25, "vaeEncoding": "WanVae.encode-sample", "v2vStrength": 0.6},
    }
    return {
        "schemaVersion": 2,
        "producer": "mlx-gen-krea-realtime/sc20684",
        "runId": run_id,
        "case": {"mode": mode, "cacheTier": tier},
        "source": SOURCE,
        "model": MODEL,
        "input": {**input_by_mode[mode], "sha256": "1" * 64, "width": 832, "height": 480},
        "schedule": {
            "kind": "source-owned-self-forcing",
            "timesteps": (
                [882.3529052734375, 803.5714111328125, 681.8181762695312, 468.75, 0.0]
                if mode == "v2v"
                else [1000.0, 937.0, 833.0, 625.0, 0.0]
            ),
            "steps": 5,
            "seed": 7,
        },
        "toolchain": {
            "os": "Darwin 25.0",
            "arch": "arm64",
            "rustc": "rustc 1.90",
            "cargo": "cargo 1.90",
            "mlx": "0.30.0",
            "hardwareModel": "Mac16,7",
            "metalDevice": "Apple M5 Max",
        },
        "geometry": {
            "batch": 1,
            "heads": 40,
            "queryTokens": 8,
            "keyTokens": 16,
            "headDim": 128,
            "groupSize": 64,
            "mask": "block-causal",
            "width": 832,
            "height": 480,
            "frames": 25,
            "latentFrames": 7,
            "generatedLatentFrames": 6 if mode == "i2v" else 7,
        },
        "compiledHandle": {
            "identity": f"sc20684/krea-packed-affine-{tier}-d128-g64-v1",
            "retainedBytes": 128,
            "compiled": True,
            "acceptedDispatches": 5,
        },
        "bytes": {
            "persistent": 4096,
            "retainedHandle": 128,
            "boundedScratch": 4096,
            "denseWindow": 0,
            "scoreMatrix": 0,
        },
        "timing": {
            "label": "fresh-process-full-schedule-loaded-product",
            "processWallMs": 80.0,
            "candidateWallMs": 45.0,
            "loadMs": 10.0,
            "conditioningMs": 2.0,
            "packedGenerationMs": 20.0,
            "denseGenerationMs": 21.0,
            "packedDecodeMs": 10.0,
            "denseDecodeMs": 11.0,
            "requestFirstFrameAvailableMs": 32.0,
            "coldProcessFirstFrameAvailableMs": 42.0,
            "meanOutputFrameMs": 1.2,
            "meanOutputFps": 1000.0 / 1.2,
            "coldEvaluatedPackedForwardMs": 3.0,
            "steadyEvaluatedPackedForwardMeanMs": 2.0,
            "steadyEvaluatedPackedForwardCount": 4,
            "compileUpperBoundMs": 1.0,
            "acceptedPackedAppendMeanMs": 0.5,
            "acceptedPackedAppendCount": 5,
            "progressStepCount": 10 if mode == "i2v" else 15,
            "steadyDenoiseStepMeanMs": 2.5,
            "steadyFullChunkMs": 240.0,
            "steadyDenoiseEquivalentFps": 50.0,
        },
        "memory": {
            **{name: {"physFootprintBytes": 100, "physFootprintPeakBytes": 200} for name in ("processStart", "weightsLoaded", "packedTerminal", "candidateTerminal", "verificationTerminal", "release")},
            "mlx": {
                "weightsLoadedActiveBytes": 10,
                "weightsLoadedCacheBytes": 2,
                "candidateTerminalActiveBytes": 90,
                "candidateTerminalCacheBytes": 18,
                "verificationTerminalActiveBytes": 100,
                "verificationTerminalCacheBytes": 20,
                "exactActivePeakBytes": 150,
                "sampledActivePeakBytes": 140,
                "sampledCachePeakBytes": 30,
                "sampledFootprintPeakBytes": 160,
                "footprintPeakActiveBytes": 140,
                "footprintPeakCacheBytes": 20,
                "sampleCount": 10,
                "periodicSampleCount": 8,
                "samplingSpanMicros": 1000,
                "intervalMicros": 100,
                "maxGapMicros": 100,
                "releaseActiveBytes": 90,
                "releaseCacheBytes": 10,
            },
            "releaseVerified": True,
        },
        "parity": {"status": "pass", "candidateTier": tier, "maxAbsError": 0.01, "tolerance": 0.1},
        "quality": {
            "status": "pass",
            "candidateTier": tier,
            "metric": "paired-rgb-and-temporal-delta",
            "maxAbsRgbU8": 2,
            "maxAbsRgbU8Tolerance": 32,
            "meanAbsRgbU8": 0.5,
            "meanAbsRgbU8Tolerance": 1.0,
            "packedMeanTemporalDelta": 4.0,
            "denseMeanTemporalDelta": 3.5,
            "temporalDeltaDrift": 0.5,
            "temporalDeltaDriftTolerance": 4.0,
            "acknowledged": True,
        },
        "fallback": {"count": 0, "reason": None},
        "cancellation": {"status": "pass", "requests": 1, "partialStateMutation": False, "scratchReleased": True},
        "output": {
            "status": "generated",
            "sha256": "d" * 64,
            "denseSha256": "e" * 64,
            "artifacts": [
                {"path": f"frame-{index:03}.ppm", "sha256": f"{index + 1:064x}", "bytes": index + 1}
                for index in range(25)
            ],
        },
    }


def baseline_observation(mode: str, tier: str, run_id: str, candidate: dict) -> dict:
    return {
        "schemaVersion": 2,
        "producer": "mlx-gen-krea-realtime/sc20684-dense-baseline",
        "runId": run_id,
        "case": {"mode": mode, "cacheTier": tier},
        **{key: candidate[key] for key in ("source", "model", "input", "schedule", "toolchain", "geometry")},
        "timing": {
            "label": "fresh-process-full-schedule-dense-read-window-baseline",
            "processWallMs": 50.0,
            "loadMs": 11.0,
            "conditioningMs": 2.0,
            "generationMs": 22.0,
            "decodeMs": 10.0,
            "requestFirstFrameAvailableMs": 34.0,
            "coldProcessFirstFrameAvailableMs": 45.0,
            "meanOutputFrameMs": 32.0 / 25.0,
            "meanOutputFps": 1000.0 / (32.0 / 25.0),
            "progressStepCount": 10 if mode == "i2v" else 15,
            "steadyDenoiseStepMeanMs": 3.0,
            "steadyFullChunkMs": 300.0,
            "steadyDenoiseEquivalentFps": 40.0,
        },
        "memory": {
            **{name: {"physFootprintBytes": 120, "physFootprintPeakBytes": 220} for name in ("processStart", "weightsLoaded", "generationTerminal", "release")},
            "mlx": {
                "weightsLoadedActiveBytes": 10,
                "weightsLoadedCacheBytes": 2,
                "generationTerminalActiveBytes": 120,
                "generationTerminalCacheBytes": 22,
                "exactActivePeakBytes": 170,
                "sampledActivePeakBytes": 160,
                "sampledCachePeakBytes": 30,
                "sampledFootprintPeakBytes": 180,
                "footprintPeakActiveBytes": 160,
                "footprintPeakCacheBytes": 20,
                "sampleCount": 10,
                "periodicSampleCount": 8,
                "samplingSpanMicros": 1000,
                "intervalMicros": 100,
                "maxGapMicros": 100,
                "releaseActiveBytes": 110,
                "releaseCacheBytes": 20,
            },
            "releaseVerified": True,
        },
        "output": {"status": "generated", "sha256": candidate["output"]["denseSha256"]},
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
            candidate = self.validate(mode, tier, run_id)
            baseline_run_id = f"baseline-{number}"
            baseline = campaign._validate_baseline_observation(
                baseline_observation(mode, tier, baseline_run_id, candidate),
                expected_mode=mode,
                expected_tier=tier,
                run_id=baseline_run_id,
                candidate=candidate,
            )
            rows.append({
                "mode": mode,
                "cacheTier": tier,
                "launcherElapsedNs": 1,
                "artifactDirectory": f"artifacts/{mode}-{tier}",
                "transcripts": {
                    role: {
                        stream: {"path": f"transcripts/{mode}-{tier}.{role}.{stream}.log", "sha256": "0" * 64, "bytes": 0}
                        for stream in ("stdout", "stderr")
                    }
                    for role in ("paired", "dense-baseline")
                },
                "comparison": {
                    "candidatePhysFootprintPeakBytes": 200,
                    "baselinePhysFootprintPeakBytes": 220,
                    "physFootprintReductionFraction": 20 / 220,
                    "candidateMlxFootprintPeakBytes": 160,
                    "baselineMlxFootprintPeakBytes": 180,
                    "mlxFootprintReductionFraction": 20 / 180,
                    "candidateRequestFirstFrameMs": 32.0,
                    "baselineRequestFirstFrameMs": 34.0,
                    "candidateSteadyDenoiseEquivalentFps": 50.0,
                    "baselineSteadyDenoiseEquivalentFps": 40.0,
                },
                "baseline": baseline,
                "observation": candidate,
            })
        return rows

    def write_evidence(self, evidence_root: Path, rows: list[dict]) -> None:
        import hashlib

        for row in rows:
            directory = evidence_root / "artifacts" / f"{row['mode']}-{row['cacheTier']}"
            directory.mkdir(parents=True)
            for index, artifact in enumerate(row["observation"]["output"]["artifacts"]):
                raw = bytes([index]) * (index + 1)
                (directory / artifact["path"]).write_bytes(raw)
                artifact["bytes"] = len(raw)
                artifact["sha256"] = hashlib.sha256(raw).hexdigest()
            transcript_dir = evidence_root / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            for role, transcripts in row["transcripts"].items():
                for stream, identity in transcripts.items():
                    raw = f"{row['mode']}/{row['cacheTier']} {role} {stream}\n".encode()
                    path = evidence_root / identity["path"]
                    path.write_bytes(raw)
                    identity["bytes"] = len(raw)
                    identity["sha256"] = hashlib.sha256(raw).hexdigest()

    def test_help_is_available_without_weights_or_accelerator(self) -> None:
        import subprocess
        import sys

        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--product-command", result.stdout)
        self.assertIn("SC20684_KREA_PROVIDER_OBSERVATION", result.stdout)

    def test_launcher_and_provider_seam_have_one_owned_protocol(self) -> None:
        provider = (
            ROOT / "crates/media/mlx-gen/mlx-gen-krea-realtime/tests/generate_smoke.rs"
        ).read_text(encoding="utf-8")
        cache = (
            ROOT / "crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs"
        ).read_text(encoding="utf-8")
        launcher = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("fn sc20684_packed_campaign_observer()", provider)
        self.assertIn("SC20684_KREA_PROVIDER_OBSERVATION", provider)
        self.assertIn("SC20684_KREA_BASELINE_OBSERVATION", provider)
        self.assertIn("generate_latents_conditioned_into", provider)
        self.assertIn("packed_metal_route_receipt", cache)
        self.assertIn("KREA_SC20684_Q4_QUALITY_ARM", launcher)
        self.assertNotIn("/Volumes/Data", launcher)
        self.assertNotIn('add_argument("--parity"', launcher)
        self.assertNotIn('add_argument("--quality"', launcher)

    def test_complete_matrix_closes_only_after_all_modes_and_tiers(self) -> None:
        rows = self.complete_rows()
        campaign.validate_matrix(rows)
        with self.assertRaisesRegex(campaign.CampaignError, "incomplete"):
            campaign.validate_matrix(rows[:-1])
        malformed = [*rows]
        malformed[-1] = {**malformed[-1], "mode": "t2v", "cacheTier": "q8"}
        with self.assertRaisesRegex(campaign.CampaignError, "exactly once"):
            campaign.validate_matrix(malformed)
        drifted = self.complete_rows()
        drifted[0]["comparison"]["candidateSteadyDenoiseEquivalentFps"] = 1.0
        with self.assertRaisesRegex(campaign.CampaignError, "comparison drift"):
            campaign.validate_matrix(drifted)

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
        row["timing"]["packedGenerationMs"] = -1
        with self.assertRaisesRegex(campaign.CampaignError, "timing.packedGenerationMs"):
            self.validate_from(row, mode="i2v", tier="q4")

    def test_old_schema_zero_conditioning_and_conflated_timing_fail_closed(self) -> None:
        row = observation("i2v", "q8", "run")
        row["schemaVersion"] = 1
        with self.assertRaisesRegex(campaign.CampaignError, "schema or producer"):
            self.validate_from(row, mode="i2v")
        row = observation("i2v", "q8", "run")
        row["input"]["sha256"] = ""
        with self.assertRaisesRegex(campaign.CampaignError, "input identity"):
            self.validate_from(row, mode="i2v")
        row = observation("i2v", "q8", "run")
        row["timing"]["requestFirstFrameAvailableMs"] = row["timing"]["processWallMs"]
        with self.assertRaisesRegex(campaign.CampaignError, "request first-frame"):
            self.validate_from(row, mode="i2v")

    def test_v2v_receipt_cannot_relabel_the_strength_schedule_as_t2v(self) -> None:
        row = observation("v2v", "q8", "run")
        row["schedule"]["timesteps"] = [1000.0, 937.0, 833.0, 625.0, 0.0]
        with self.assertRaisesRegex(campaign.CampaignError, "five-step schedule"):
            self.validate_from(row, mode="v2v")

    def test_i2v_receipt_cannot_overgenerate_past_the_reference_plus_continuation(self) -> None:
        row = observation("i2v", "q8", "run")
        row["geometry"]["generatedLatentFrames"] = 7
        with self.assertRaisesRegex(campaign.CampaignError, "generated-latent geometry"):
            self.validate_from(row, mode="i2v")

    def test_q8_and_q4_cannot_substitute_one_another(self) -> None:
        row = observation("v2v", "q4", "run")
        row["compiledHandle"]["identity"] = "sc20684/krea-packed-affine-q8-d128-g64-v1"
        with self.assertRaisesRegex(campaign.CampaignError, "launched cache tier"):
            self.validate_from(row, mode="v2v", tier="q4")
        row = observation("v2v", "q4", "run")
        row["quality"]["candidateTier"] = "q8"
        with self.assertRaisesRegex(campaign.CampaignError, "tier-substituted"):
            self.validate_from(row, mode="v2v", tier="q4")

    def test_partial_dense_fallback_and_false_release_are_refused(self) -> None:
        row = observation("t2v", "q8", "run")
        row["fallback"] = {"count": 1, "reason": "geometry"}
        with self.assertRaisesRegex(campaign.CampaignError, "dense fallback"):
            self.validate_from(row)
        row = observation("t2v", "q8", "run")
        row["memory"]["mlx"]["releaseActiveBytes"] = 101
        with self.assertRaisesRegex(campaign.CampaignError, "contradicts"):
            self.validate_from(row)
        candidate = self.validate()
        baseline = baseline_observation("t2v", "q8", "baseline", candidate)
        baseline["output"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(campaign.CampaignError, "differs"):
            campaign._validate_baseline_observation(
                baseline,
                expected_mode="t2v",
                expected_tier="q8",
                run_id="baseline",
                candidate=candidate,
            )

    def test_atomic_publication_never_leaves_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt"
            rows = self.complete_rows()
            evidence_root = Path(temporary) / "evidence-source"
            self.write_evidence(evidence_root, rows)
            campaign.publish(output, source=SOURCE, model=MODEL, rows=rows, evidence_root=evidence_root)
            self.assertTrue((output / "receipt.json").is_file())
            self.assertTrue((output / "receipt.json.sha256").is_file())
            self.assertTrue((output / "manifest.sha256").is_file())
            self.assertTrue((output / "artifacts/t2v-q8/frame-000.ppm").is_file())
            self.assertTrue((output / "transcripts/t2v-q8.paired.stdout.log").is_file())
            self.assertTrue((output / "transcripts/t2v-q8.dense-baseline.stdout.log").is_file())
            import hashlib

            for line in (output / "manifest.sha256").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(), digest)
            sealed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "terminal-complete")
            self.assertEqual(len(sealed["matrix"]), 6)
            with self.assertRaisesRegex(campaign.CampaignError, "already exists"):
                campaign.publish(output, source=SOURCE, model=MODEL, rows=rows, evidence_root=evidence_root)
            partial = Path(temporary) / "incomplete"
            with self.assertRaisesRegex(campaign.CampaignError, "incomplete"):
                campaign.publish(partial, source=SOURCE, model=MODEL, rows=rows[:-1], evidence_root=evidence_root)
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
