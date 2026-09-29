"""Weightless closure tests for the SC-20684 real-weight campaign launcher."""

from __future__ import annotations

import importlib.util
import copy
import functools
import hashlib
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "sc20684_krea_realtime_campaign.py"
ROOT = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location("sc20684_campaign", SCRIPT)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
GIB = 1024**3

SOURCE = {
    "repositoryHead": "a" * 40,
    "files": {path: "b" * 64 for path in campaign.SOURCE_FILES},
}
MODEL_FILES = {
    name: {"size": 1, "sha256": "c" * 64}
    for name in ("config.json", "dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
}
MODEL_INVENTORY = hashlib.sha256(
    "".join(
        f"{name}\0{identity['size']}\0{identity['sha256']}\n"
        for name, identity in MODEL_FILES.items()
    ).encode("utf-8")
).hexdigest()
MODEL = {
    "repository": campaign.MODEL_REPOSITORY,
    "revision": campaign.MODEL_REVISION,
    "variant": "q4",
    "configSha256": "c" * 64,
    "files": MODEL_FILES,
    "inventorySha256": MODEL_INVENTORY,
}


def source_budget(mode: str, tier: str) -> dict:
    per_token = 435_200 if tier == "q8" else 230_400
    previous = 0
    append = 0
    for frames in ((1, 3, 3) if mode == "i2v" else (3, 3, 1)):
        current = frames * 1_560
        staged = previous + current
        append = max(append, (previous + staged) * per_token + current * 819_200)
        previous = staged
    return {
        "modelLogicalBytes": 3,
        "terminalKvBytes": previous * per_token,
        "appendOldStagedAndNewDenseBytes": append,
        "decodeWorkingSetEstimateBytes": 64 * 28 * 480 * 832 + 6_500 * 28 * 256 * 256,
        "runtimeOverheadBytes": None,
        "wholeProcessPeakBoundBytes": None,
        "decodePlan": campaign.DECODE_PLAN_IDENTITY,
    }


def phase_memory(phases: tuple[str, ...]) -> list[dict]:
    return [
        {
            "phase": phase,
            "process": {"physFootprintBytes": 6 * GIB, "physFootprintPeakBytes": 8 * GIB},
            "mlxActiveBytes": 5 * GIB,
            "mlxCacheBytes": GIB // 2,
        }
        for phase in phases
    ]


def observation(mode: str, tier: str, run_id: str) -> dict:
    input_by_mode = {
        "t2v": {"kind": "text-only", "frameCount": 0, "vaeEncoding": "none", "v2vStrength": None},
        "i2v": {"kind": "deterministic-gradient-still", "frameCount": 1, "vaeEncoding": "WanVae.encode-mode", "v2vStrength": None},
        "v2v": {"kind": "deterministic-smooth-motion-clip", "frameCount": 25, "vaeEncoding": "WanVae.encode-sample", "v2vStrength": 0.6},
    }
    handle_identity = f"sc20684/krea-packed-affine-{tier}-d128-g64-v1"
    route_before = {
        "compiledHandleIdentity": handle_identity,
        "retainedHandleBytes": 128,
        "acceptedForwards": 0,
        "materializedScratchDispatches": 0,
        "boundedScratchBytes": 0,
        "denseWindowBytes": 0,
        "scoreMatrixBytes": 0,
        "dispatchGeometries": [],
    }
    route_at_cancel = {
        "compiledHandleIdentity": handle_identity,
        "retainedHandleBytes": 128,
        "acceptedForwards": 1,
        "materializedScratchDispatches": 1,
        "boundedScratchBytes": (3 * 8 * 128 + 8 * 8 + 3 * 8) * 4,
        "denseWindowBytes": 0,
        "scoreMatrixBytes": 0,
        "dispatchGeometries": [
            {"queryTokens": 8, "keyTokens": 16, "acceptedForwards": 1},
        ],
    }
    return {
        "schemaVersion": 6,
        "producer": "mlx-gen-krea-realtime/sc20684",
        "runId": run_id,
        "case": {"mode": mode, "cacheTier": tier},
        "source": SOURCE,
        "model": MODEL,
        "sourceBudget": source_budget(mode, tier),
        "phaseMemory": phase_memory((
            "weights-loaded", "conditioning-complete", "packed-generation-complete",
            "packed-decode-complete", "dense-generation-complete", "dense-decode-complete",
            "cancellation-complete", "release",
        )),
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
            "dispatchGeometries": [
                {"queryTokens": 8, "keyTokens": 16, "acceptedForwards": 5},
            ],
            "headDim": 128,
            "groupSize": 64,
            "queryTile": 8,
            "keyTile": 8,
            "dtypes": {
                "packedCodes": "uint32",
                "packedMetadata": "bfloat16",
                "queryAndCurrentKv": "bfloat16",
                "accumulator": "float32",
            },
            "mask": "block-causal",
            "width": 832,
            "height": 480,
            "frames": 25,
            "latentFrames": 7,
            "generatedLatentFrames": 6 if mode == "i2v" else 7,
        },
        "compiledHandle": {
            "identity": handle_identity,
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
            **{
                name: {
                    "physFootprintBytes": 6 * GIB,
                    "physFootprintPeakBytes": 8 * GIB,
                }
                for name in (
                    "processStart", "weightsLoaded", "packedTerminal", "candidateTerminal",
                    "verificationTerminal", "release",
                )
            },
            "mlx": {
                "weightsLoadedActiveBytes": 5 * GIB,
                "weightsLoadedCacheBytes": GIB // 2,
                "candidateTerminalActiveBytes": 6 * GIB,
                "candidateTerminalCacheBytes": GIB // 2,
                "verificationTerminalActiveBytes": 6 * GIB,
                "verificationTerminalCacheBytes": GIB // 2,
                "exactActivePeakBytes": 15 * GIB // 2,
                "sampledActivePeakBytes": 7 * GIB,
                "sampledCachePeakBytes": GIB,
                "sampledFootprintPeakBytes": 8 * GIB,
                "footprintPeakActiveBytes": 7 * GIB,
                "footprintPeakCacheBytes": GIB,
                "sampleCount": 10,
                "periodicSampleCount": 8,
                "samplingSpanMicros": 1000,
                "intervalMicros": 100,
                "maxGapMicros": 100,
                "releaseActiveBytes": 5 * GIB,
                "releaseCacheBytes": GIB // 2,
            },
            "releaseVerified": True,
        },
        "parity": {
            "status": "pass",
            "candidateTier": tier,
            "maxAbsError": 0.01,
            "tolerance": campaign.PARITY_MAX_ABS_ERROR_BY_TIER[tier],
        },
        "quality": {
            "status": "pass",
            "candidateTier": tier,
            "metric": "paired-rgb-and-temporal-delta",
            "maxAbsRgbU8": 2,
            "maxAbsRgbU8Tolerance": campaign.QUALITY_THRESHOLDS_BY_TIER[tier]["maxAbsRgbU8"],
            "meanAbsRgbU8": 0.5,
            "meanAbsRgbU8Tolerance": campaign.QUALITY_THRESHOLDS_BY_TIER[tier]["meanAbsRgbU8"],
            "packedMeanTemporalDelta": 4.0,
            "denseMeanTemporalDelta": 3.5,
            "temporalDeltaDrift": 0.5,
            "temporalDeltaDriftTolerance": campaign.QUALITY_THRESHOLDS_BY_TIER[tier]["temporalDeltaDrift"],
            "acknowledged": True,
        },
        "fallback": {"count": 0, "reason": None},
        "cancellation": {
            "status": "pass",
            "requests": 1,
            "trigger": "after-first-materialized-denoise-step",
            "progressStepsBeforeCancel": 1,
            "typedCancellationObserved": True,
            "dispatchToken": campaign._cancellation_dispatch_token(run_id, route_at_cancel),
            "routeBefore": route_before,
            "routeAtCancel": route_at_cancel,
            "expectedContextTokens": 2,
            "storedTokensAfterCancel": 2,
            "activeBytesBefore": 6 * GIB,
            "activeBytesAfterRelease": 6 * GIB,
            "cacheBytesBefore": GIB // 2,
            "cacheBytesAfterRelease": GIB // 2,
            "allocationObserved": True,
            "partialStateMutation": False,
            "scratchReleased": True,
        },
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
        "schemaVersion": 6,
        "producer": "mlx-gen-krea-realtime/sc20684-dense-baseline",
        "runId": run_id,
        "case": {"mode": mode, "cacheTier": tier},
        **{key: candidate[key] for key in ("source", "model", "input", "schedule", "toolchain", "sourceBudget")},
        "phaseMemory": phase_memory((
            "weights-loaded", "conditioning-complete", "dense-generation-complete",
            "dense-decode-complete", "release",
        )),
        "geometry": {**candidate["geometry"], "dispatchGeometries": []},
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
            **{
                name: {
                    "physFootprintBytes": 7 * GIB,
                    "physFootprintPeakBytes": 10 * GIB,
                }
                for name in ("processStart", "weightsLoaded", "generationTerminal", "release")
            },
            "mlx": {
                "weightsLoadedActiveBytes": 5 * GIB,
                "weightsLoadedCacheBytes": GIB // 2,
                "generationTerminalActiveBytes": 7 * GIB,
                "generationTerminalCacheBytes": GIB // 2,
                "exactActivePeakBytes": 19 * GIB // 2,
                "sampledActivePeakBytes": 9 * GIB,
                "sampledCachePeakBytes": GIB,
                "sampledFootprintPeakBytes": 10 * GIB,
                "footprintPeakActiveBytes": 9 * GIB,
                "footprintPeakCacheBytes": GIB,
                "sampleCount": 10,
                "periodicSampleCount": 8,
                "samplingSpanMicros": 1000,
                "intervalMicros": 100,
                "maxGapMicros": 100,
                "releaseActiveBytes": 5 * GIB,
                "releaseCacheBytes": GIB // 2,
            },
            "releaseVerified": True,
        },
        "output": {"status": "generated", "sha256": candidate["output"]["denseSha256"]},
    }


class KreaRealtimeCampaignTests(unittest.TestCase):
    def test_fixed_decode_plan_and_unknown_peak_are_required(self) -> None:
        row = observation("i2v", "q8", "run")
        self.assertEqual(row["sourceBudget"]["appendOldStagedAndNewDenseBytes"], 11_301_888_000)
        self.assertEqual(row["sourceBudget"]["decodeWorkingSetEstimateBytes"], 12_643_205_120)
        for field, value in (
            ("decodePlan", {**campaign.DECODE_PLAN_IDENTITY, "spatialTilePx": 320}),
            ("runtimeOverheadBytes", 0),
            ("terminalKvBytes", 0),
        ):
            altered = copy.deepcopy(row)
            altered["sourceBudget"][field] = value
            with self.subTest(field=field), self.assertRaises(campaign.CampaignError):
                campaign._validate_observation(
                    altered, expected_mode="i2v", expected_tier="q8", run_id="run",
                    expected_source=SOURCE, expected_snapshot=MODEL,
                )

    def test_phase_ledger_rejects_missing_or_reordered_phase(self) -> None:
        row = observation("t2v", "q4", "run")
        row["phaseMemory"].pop(3)
        with self.assertRaises(campaign.CampaignError):
            campaign._validate_observation(
                row, expected_mode="t2v", expected_tier="q4", run_id="run",
                expected_source=SOURCE, expected_snapshot=MODEL,
            )

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
            row = {
                "mode": mode,
                "cacheTier": tier,
                "launcherElapsedNs": 1,
                "artifactDirectory": f"artifacts/{mode}-{tier}",
                "processExitCodes": {"paired": 0, "dense-baseline": 0},
                "transcripts": {
                    role: {
                        stream: {"path": f"transcripts/{mode}-{tier}.{role}.{stream}.log", "sha256": "0" * 64, "bytes": 0}
                        for stream in ("stdout", "stderr")
                    }
                    for role in ("paired", "dense-baseline")
                },
                "comparison": {
                    "candidatePhysFootprintPeakBytes": 8 * GIB,
                    "baselinePhysFootprintPeakBytes": 10 * GIB,
                    "physFootprintReductionFraction": 0.2,
                    "candidateMlxFootprintPeakBytes": 8 * GIB,
                    "baselineMlxFootprintPeakBytes": 10 * GIB,
                    "mlxFootprintReductionFraction": 0.2,
                    "candidateRequestFirstFrameMs": 32.0,
                    "baselineRequestFirstFrameMs": 34.0,
                    "candidateMeanOutputFps": 1000.0 / 1.2,
                    "baselineMeanOutputFps": 1000.0 / (32.0 / 25.0),
                    "candidateSteadyDenoiseEquivalentFps": 50.0,
                    "baselineSteadyDenoiseEquivalentFps": 40.0,
                },
                "baseline": baseline,
                "observation": candidate,
            }
            row["decision"] = campaign.arm_decision(row, campaign.decision_policy())
            rows.append(row)
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

        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, encoding="utf-8", check=False)
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
        result = campaign.campaign_decision(rows, campaign.decision_policy())
        self.assertEqual(result["overall"]["decision"], "go")
        self.assertEqual(len(result["overall"]["eligibleGeometries"]), 6)
        self.assertEqual(result["tiers"]["q8"]["decision"], "go")
        self.assertEqual(result["tiers"]["q4"]["decision"], "go")
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

    def test_identity_drift_is_refused_and_dense_bytes_are_a_sealed_no_go(self) -> None:
        row = observation("t2v", "q8", "run")
        row["source"] = {**SOURCE, "repositoryHead": "e" * 40}
        with self.assertRaisesRegex(campaign.CampaignError, "source identity drift"):
            self.validate_from(row)
        row = observation("t2v", "q8", "run")
        row["bytes"]["denseWindow"] = 1
        validated = self.validate_from(row)
        matrix_row = self.complete_rows()[0]
        matrix_row["observation"] = validated
        decision = campaign.arm_decision(matrix_row, campaign.decision_policy())
        self.assertEqual(decision["decision"], "no-go")
        self.assertIn("zeroDenseWindowBytes", decision["failedCriteria"])

    def test_snapshot_identity_hashes_exact_inventory_and_detects_same_size_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / campaign.MODEL_REVISION / "q4"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 64}}),
                encoding="utf-8",
            )
            for name in ("dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "tokenizer.json"):
                (snapshot / name).write_bytes(name.encode("utf-8"))
            first = campaign.snapshot_identity(snapshot)
            self.assertEqual(first["configSha256"], first["files"]["config.json"]["sha256"])
            self.assertEqual(set(first["files"]), set(MODEL_FILES))
            original = (snapshot / "dit.safetensors").read_bytes()
            (snapshot / "dit.safetensors").write_bytes(b"x" * len(original))
            second = campaign.snapshot_identity(snapshot)
            self.assertEqual(
                first["files"]["dit.safetensors"]["size"],
                second["files"]["dit.safetensors"]["size"],
            )
            self.assertNotEqual(first["inventorySha256"], second["inventorySha256"])

            wrong_revision = Path(temporary) / ("0" * 40) / "q4"
            wrong_revision.parent.mkdir()
            snapshot.rename(wrong_revision)
            with self.assertRaisesRegex(campaign.CampaignError, "exact pinned revision"):
                campaign.snapshot_identity(wrong_revision)

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

    def test_per_dispatch_geometry_coverage_is_exact_and_counted(self) -> None:
        row = observation("t2v", "q8", "run")
        row["geometry"]["dispatchGeometries"] = [
            {"queryTokens": 8, "keyTokens": 16, "acceptedForwards": 3},
            {"queryTokens": 4, "keyTokens": 12, "acceptedForwards": 2},
        ]
        validated = self.validate_from(row)
        matrix_row = self.complete_rows()[0]
        matrix_row["observation"] = validated
        decision = campaign.arm_decision(matrix_row, campaign.decision_policy())
        self.assertEqual(
            decision["eligibleGeometry"]["geometry"]["dispatchGeometries"],
            row["geometry"]["dispatchGeometries"],
        )

        row["geometry"]["dispatchGeometries"][0]["acceptedForwards"] = 2
        with self.assertRaisesRegex(campaign.CampaignError, "coverage contradicts"):
            self.validate_from(row)

    def test_q8_and_q4_cannot_substitute_one_another(self) -> None:
        row = observation("v2v", "q4", "run")
        row["compiledHandle"]["identity"] = "sc20684/krea-packed-affine-q8-d128-g64-v1"
        with self.assertRaisesRegex(campaign.CampaignError, "launched cache tier"):
            self.validate_from(row, mode="v2v", tier="q4")
        row = observation("v2v", "q4", "run")
        row["quality"]["candidateTier"] = "q8"
        with self.assertRaisesRegex(campaign.CampaignError, "tier-substituted"):
            self.validate_from(row, mode="v2v", tier="q4")

    def test_provider_cannot_supply_its_own_parity_or_quality_tolerances(self) -> None:
        row = observation("t2v", "q8", "run")
        row["parity"]["maxAbsError"] = 1.0
        row["parity"]["tolerance"] = 2.0
        with self.assertRaisesRegex(campaign.CampaignError, "frozen decision policy"):
            self.validate_from(row)

        row = observation("v2v", "q4", "run")
        row["quality"]["meanAbsRgbU8"] = 4.0
        row["quality"]["meanAbsRgbU8Tolerance"] = 5.0
        with self.assertRaisesRegex(campaign.CampaignError, "frozen decision policy"):
            self.validate_from(row, mode="v2v", tier="q4")

    def test_release_must_return_to_loaded_model_and_terminal_boundaries(self) -> None:
        row = observation("t2v", "q8", "run")
        row["memory"]["mlx"]["releaseActiveBytes"] = 5 * GIB + 1
        row["memory"]["releaseVerified"] = False
        validated = self.validate_from(row)
        self.assertFalse(validated["memory"]["releaseVerified"])
        row["memory"]["releaseVerified"] = True
        with self.assertRaisesRegex(campaign.CampaignError, "terminal resource boundaries"):
            self.validate_from(row)

        candidate = self.validate()
        baseline = baseline_observation("t2v", "q8", "baseline", candidate)
        baseline["memory"]["mlx"]["releaseActiveBytes"] = 5 * GIB + 1
        baseline["memory"]["releaseVerified"] = True
        with self.assertRaisesRegex(campaign.CampaignError, "resource boundaries"):
            campaign._validate_baseline_observation(
                baseline,
                expected_mode="t2v",
                expected_tier="q8",
                run_id="baseline",
                candidate=candidate,
            )

    def test_cancellation_must_be_in_flight_allocated_and_cleaned(self) -> None:
        row = observation("t2v", "q8", "run")
        row["cancellation"]["trigger"] = "pre-cancelled"
        with self.assertRaisesRegex(campaign.CampaignError, "triggered in-flight"):
            self.validate_from(row)

        row = observation("t2v", "q8", "run")
        row["cancellation"]["routeAtCancel"] = dict(row["cancellation"]["routeBefore"])
        row["cancellation"]["dispatchToken"] = campaign._cancellation_dispatch_token(
            "run", row["cancellation"]["routeAtCancel"]
        )
        row["cancellation"]["allocationObserved"] = False
        row["cancellation"]["status"] = "fail"
        row["memory"]["releaseVerified"] = False
        validated = self.validate_from(row)
        self.assertEqual(validated["cancellation"]["status"], "fail")

        row["cancellation"]["allocationObserved"] = True
        with self.assertRaisesRegex(campaign.CampaignError, "packed-route receipt delta"):
            self.validate_from(row)

        row = observation("t2v", "q8", "run")
        row["cancellation"]["routeAtCancel"]["acceptedForwards"] = 2
        row["cancellation"]["routeAtCancel"]["materializedScratchDispatches"] = 2
        row["cancellation"]["routeAtCancel"]["dispatchGeometries"][0]["acceptedForwards"] = 2
        row["cancellation"]["dispatchToken"] = campaign._cancellation_dispatch_token(
            "run", row["cancellation"]["routeAtCancel"]
        )
        with self.assertRaisesRegex(campaign.CampaignError, "packed-route receipt delta"):
            self.validate_from(row)

        row = observation("t2v", "q8", "run")
        row["cancellation"]["dispatchToken"] = "f" * 64
        with self.assertRaisesRegex(campaign.CampaignError, "does not bind"):
            self.validate_from(row)

        row = observation("t2v", "q8", "run")
        row["cancellation"]["injectedAllocationBytes"] = 64 * 1024**2
        with self.assertRaisesRegex(campaign.CampaignError, "cancellation fields differ"):
            self.validate_from(row)

    def test_fallback_quality_and_release_failures_are_sealed_no_go_outcomes(self) -> None:
        matrix_row = self.complete_rows()[0]
        row = observation("t2v", "q8", "run")
        row["fallback"] = {"count": 1, "reason": "geometry"}
        matrix_row["observation"] = self.validate_from(row)
        decision = campaign.arm_decision(matrix_row, campaign.decision_policy())
        self.assertEqual(decision["decision"], "no-go")
        self.assertIn("noDenseFallback", decision["failedCriteria"])

        matrix_row = self.complete_rows()[1]
        row = observation("t2v", "q4", "run")
        row["quality"]["status"] = "fail"
        row["quality"]["meanAbsRgbU8"] = 4.0
        matrix_row["observation"] = self.validate_from(row, tier="q4")
        matrix_row["processExitCodes"]["paired"] = 101
        decision = campaign.arm_decision(matrix_row, campaign.decision_policy())
        self.assertEqual(decision["decision"], "no-go")
        self.assertIn("qualityPassed", decision["failedCriteria"])
        self.assertIn("pairedProcessExitZero", decision["failedCriteria"])

        matrix_row = self.complete_rows()[0]
        row = observation("t2v", "q8", "run")
        row["memory"]["mlx"]["releaseActiveBytes"] = 7 * GIB
        row["memory"]["releaseVerified"] = False
        matrix_row["observation"] = self.validate_from(row)
        decision = campaign.arm_decision(matrix_row, campaign.decision_policy())
        self.assertEqual(decision["decision"], "no-go")
        self.assertIn("candidateReleaseVerified", decision["failedCriteria"])

        row["memory"]["releaseVerified"] = True
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
            self.assertEqual(sealed["schemaVersion"], campaign.RECEIPT_SCHEMA_VERSION)
            self.assertEqual(sealed["status"], "terminal-go")
            self.assertEqual(sealed["decision"]["overall"]["decision"], "go")
            self.assertEqual(
                sealed["decision"]["policySha256"],
                campaign._canonical_sha256(campaign.decision_policy()),
            )
            self.assertEqual(len(sealed["matrix"]), 6)
            with self.assertRaisesRegex(campaign.CampaignError, "already exists"):
                campaign.publish(output, source=SOURCE, model=MODEL, rows=rows, evidence_root=evidence_root)
            partial = Path(temporary) / "incomplete"
            with self.assertRaisesRegex(campaign.CampaignError, "incomplete"):
                campaign.publish(partial, source=SOURCE, model=MODEL, rows=rows[:-1], evidence_root=evidence_root)
            self.assertFalse(partial.exists())

    def test_below_threshold_and_q4_quality_failure_publish_terminal_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = self.complete_rows()
            for row in rows:
                row["comparison"]["candidatePhysFootprintPeakBytes"] = (
                    row["comparison"]["baselinePhysFootprintPeakBytes"] - 128 * 1024**2
                )
                row["comparison"]["physFootprintReductionFraction"] = 0.0125
                row["observation"]["memory"]["candidateTerminal"]["physFootprintPeakBytes"] = (
                    row["comparison"]["candidatePhysFootprintPeakBytes"]
                )
                row["decision"] = campaign.arm_decision(row, campaign.decision_policy())
            q4 = next(row for row in rows if row["cacheTier"] == "q4")
            q4["observation"]["quality"]["status"] = "fail"
            q4["observation"]["quality"]["meanAbsRgbU8"] = 4.0
            q4["processExitCodes"]["paired"] = 101
            q4["decision"] = campaign.arm_decision(q4, campaign.decision_policy())

            evidence_root = Path(temporary) / "evidence-source"
            self.write_evidence(evidence_root, rows)
            output = Path(temporary) / "receipt"
            campaign.publish(
                output,
                source=SOURCE,
                model=MODEL,
                rows=rows,
                evidence_root=evidence_root,
            )
            sealed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], "terminal-no-go")
            self.assertEqual(sealed["decision"]["overall"]["eligibleGeometries"], [])
            self.assertIn(
                "physFootprintReductionMaterial",
                sealed["decision"]["arms"]["t2v/q8"]["failedCriteria"],
            )
            self.assertIn(
                "qualityPassed",
                sealed["decision"]["arms"]["t2v/q4"]["failedCriteria"],
            )

    def test_policy_explicitly_bounds_unswept_geometry_axes(self) -> None:
        policy = campaign.decision_policy()
        self.assertEqual(policy["materialMemory"]["minimumReductionBytes"], 256 * 1024**2)
        self.assertEqual(policy["materialMemory"]["minimumReductionFraction"], 0.05)
        self.assertEqual(policy["throughputNeutral"]["minimumMeanOutputFpsRatio"], 0.95)
        self.assertEqual(policy["accuracy"]["parityMaxAbsErrorByTier"], {"q8": 0.25, "q4": 0.75})
        self.assertEqual(policy["coverage"]["fixedProductGeometry"]["queryTile"], 8)
        self.assertEqual(policy["coverage"]["fixedProductGeometry"]["keyTile"], 8)
        self.assertIn("alternateGpuFamilies", policy["coverage"]["notSweptByThisSchedule"])
        self.assertEqual(
            policy["coverage"]["eligibilityBoundary"],
            "only exact measured arm geometries are eligible",
        )
        decision = campaign.arm_decision(self.complete_rows()[0], policy)
        self.assertEqual(decision["eligibleGeometry"]["geometry"]["queryTile"], 8)
        self.assertEqual(decision["eligibleGeometry"]["build"]["repositoryHead"], SOURCE["repositoryHead"])
        self.assertEqual(decision["eligibleGeometry"]["build"]["toolchain"]["rustc"], "rustc 1.90")

    def test_role_resume_binds_identity_and_rejects_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "observer"
            executable.write_bytes(b"prebuilt observer")
            policy = campaign.supervisor.SafetyPolicy(
                "darwin-mlx", 10, 100, 100, 1, 10**9, 10**5, 10**5,
                10**5, None, None, None, "f" * 64, b"{}\n",
            )
            resume = root / "resume"
            identity = campaign._prepare_media_resume(
                resume, source=json.loads(json.dumps(SOURCE)), model=MODEL,
                policy=policy, argv=[str(executable)], timeout=10,
            )
            moved_ref = json.loads(json.dumps(SOURCE))
            moved_ref["repositoryHead"] = "d" * 40
            self.assertEqual(identity, campaign._prepare_media_resume(
                resume, source=moved_ref, model=MODEL,
                policy=policy, argv=[str(executable)], timeout=10,
            ))
            transcripts = resume / "transcripts"
            transcripts.mkdir()
            stdout = transcripts / "t2v-q8.dense-baseline.stdout.log"
            stderr = transcripts / "t2v-q8.dense-baseline.stderr.log"
            stdout.write_bytes(b"validated observation")
            stderr.write_bytes(b"")
            record = {
                "runId": "run-1", "exitCode": 0,
                "transcripts": {
                    "stdout": campaign._file_identity(stdout, f"transcripts/{stdout.name}"),
                    "stderr": campaign._file_identity(stderr, f"transcripts/{stderr.name}"),
                },
                "observationSha256": "a" * 64,
                "supervision": {"pid": 123, "peakHostBytes": 1024, "hostFreeAtLaunch": 10**9, "ownedProcessGroupReaped": True},
            }
            campaign._save_resumed_role(resume, "i2v-q8.dense-baseline", identity, record)
            with self.assertRaisesRegex(campaign.CampaignError, "runtime-guarded admission"):
                campaign._load_resumed_role(resume, "i2v-q8.dense-baseline", identity)
            record["supervision"] = campaign._supervision_record(campaign.supervisor.RunResult(
                123, 0, 1024, None, 10**9, None, 1.0, (),
                campaign.supervisor.runtime_guarded_admission(policy),
            ))
            foreign = copy.deepcopy(record)
            foreign["supervision"]["admission"]["policySha256"] = "0" * 64
            campaign._save_resumed_role(resume, "v2v-q8.dense-baseline", identity, foreign)
            with self.assertRaisesRegex(campaign.CampaignError, "runtime-guarded admission"):
                campaign._load_resumed_role(resume, "v2v-q8.dense-baseline", identity)
            campaign._save_resumed_role(resume, "t2v-q8.dense-baseline", identity, record)
            self.assertEqual(campaign._load_resumed_role(
                resume, "t2v-q8.dense-baseline", identity,
            )["runId"], "run-1")
            stdout.write_bytes(b"tampered")
            with self.assertRaisesRegex(campaign.CampaignError, "changed|drift"):
                campaign._load_resumed_role(resume, "t2v-q8.dense-baseline", identity)
            executable.write_bytes(b"changed observer")
            with self.assertRaisesRegex(campaign.CampaignError, "resume identity changed"):
                campaign._prepare_media_resume(
                    resume, source=moved_ref, model=MODEL,
                    policy=policy, argv=[str(executable)], timeout=10,
                )

    def test_roles_are_admitted_by_runtime_guards_and_failures_are_unaccepted_records(self) -> None:
        policy = campaign.supervisor.SafetyPolicy(
            "darwin-mlx", 10, 20, 100, 1024, 10**9, 10**5, 10**5,
            10**5, None, None, None, "f" * 64, b"{}\n",
        )
        original = campaign.supervisor.run_guarded

        class Probe:
            def __init__(self, free: int, footprint: int) -> None:
                self.free, self.footprint = free, footprint

            def host_free(self) -> int:
                return self.free

            def tree_footprint(self, _owner: object) -> int:
                return self.footprint

        def run(root: Path, probe: Probe, script: str, argv: list[str] | None = None) -> Exception:
            command = shlex.join(argv or [sys.executable, "-c", script])
            with patch.object(campaign.supervisor, "run_guarded", functools.partial(original, probe=probe)):
                with self.assertRaises((campaign.CampaignError, campaign.supervisor.SupervisionError)) as caught:
                    campaign.run_matrix(command, root, SOURCE, MODEL, campaign.decision_policy(),
                                        10, root, policy, {"kind": "test"})
            return caught.exception

        def record(root: Path, attempt: int) -> dict:
            return json.loads((root / "logs" / f"t2v-q8.paired.{attempt}.unaccepted.json").read_bytes())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "spawned"
            spawn = f"open({str(marker)!r}, 'w').close()"
            # Admitted without any static peak bound: the product process is spawned.
            error = run(root / "admitted", Probe(10**12, 1024), spawn)
            self.assertIsInstance(error, campaign.CampaignError)
            self.assertTrue(marker.exists())
            self.assertTrue((root / "admitted/logs/t2v-q8.paired.0.stdout.log").is_file())
            # It exited cleanly without an observation: a failed, unaccepted role, never a receipt.
            sealed = record(root / "admitted", 0)
            self.assertEqual((sealed["accepted"], sealed["outcome"], sealed["reason"]),
                             (False, "failed", "invalid-evidence"))
            self.assertIsInstance(sealed["pid"], int)
            campaign.supervisor.validate_admission(sealed["admission"], policy_sha256=policy.sha256)
            self.assertFalse((root / "admitted/roles").exists())
            marker.unlink()

            missing = root / "missing"
            error = run(missing, Probe(10**12, 1024), "", argv=[str(root / "missing-observer")])
            self.assertEqual(error.reason, "spawn-failure")
            sealed = record(missing, 0)
            self.assertEqual((sealed["accepted"], sealed["outcome"], sealed["pid"]), (False, "failed", None))

            refused = root / "refused"
            short = policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes - 1
            for attempt in range(2):
                error = run(refused, Probe(short, 1024), spawn)
                self.assertEqual(error.reason, "preflight-memory")
                sealed = record(refused, attempt)
                self.assertEqual((sealed["accepted"], sealed["outcome"], sealed["pid"]), (False, "refused", None))
                self.assertEqual(sealed["coordinate"], "t2v-q8.paired")
                campaign.supervisor.validate_admission(sealed["admission"], policy_sha256=policy.sha256)
            # A gap in attempt indices must not reuse an index: the next attempt is max + 1.
            for path in (refused / "logs").glob("t2v-q8.paired.0.*"):
                path.unlink()
            self.assertEqual(run(refused, Probe(short, 1024), spawn).reason, "preflight-memory")
            self.assertEqual(record(refused, 2)["outcome"], "refused")
            self.assertFalse(marker.exists())
            self.assertFalse((refused / "roles").exists())

            aborted = root / "aborted"
            error = run(aborted, Probe(10**12, policy.child_footprint_cap_bytes + 1), "import time; time.sleep(5)")
            self.assertEqual(error.reason, "child-footprint")
            sealed = record(aborted, 0)
            self.assertEqual((sealed["accepted"], sealed["outcome"], sealed["pid"]), (False, "aborted", error.pid))
            self.assertIsNone(sealed["admission"]["wholeProcessPeakBoundBytes"])
            self.assertFalse((aborted / "roles").exists())

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
