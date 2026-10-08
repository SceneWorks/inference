"""CPU-only routing, immutable scope and trajectory completion refusal tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import tempfile
import unittest
from unittest import mock

import yaml
from scripts.ci import qwen21_current_failed_adapter as current
from scripts.ci.real_weights_workflow import inline_text

ROOT = Path(__file__).resolve().parents[2]
PHASE = {"QWEN_IMAGE_2_1_LORA_PHASE": "current-trajectory"}


class TrajectoryTests(unittest.TestCase):
    @staticmethod
    def write_admission(root: Path) -> dict:
        active, overhead, frozen, admitted, installed = (
            86_762_446_731, 25_526_824, 11_142_168_576, 6_597_947_509, 6_597_947_509)
        row = {"kind": "DIAGNOSTIC_ONLY", "refusal": None,
               "policy": "unchanged_active_envelope_plus_dynamic_free_cache_clamp_and_typed_scoped_grant",
               "activeEnvelopeBytes": active, "nonallocatorOverheadBytes": overhead,
               "frozenFreeCacheAllowanceBytes": frozen,
               "actualAllocatorCacheLimitBytes": 48_819_791_680,
               "previousCacheLimitBytes": 48_819_791_680,
               "admittedCacheLimitBytes": admitted, "effectiveCacheLimitBytes": installed,
               "cacheClampBytes": frozen - admitted,
               "reservedPhysicalCeilingBytes": active + overhead + admitted,
               "predictedMaxFullEnvelopeBytes": active + overhead + admitted,
               "installedFullEnvelopeBytes": active + overhead + installed,
               "activeAndHostReservesUnchanged": True, "cacheGrantReadBack": True,
               "typedNativeCacheGrantRequired": True, "cacheGrantRestoresOnDrop": True}
        (root / "numeric-physical-admission.json").write_text(json.dumps(row), encoding="utf-8")
        return row

    def test_actual_trajectory_source_closure_and_production_mutants(self):
        from scripts.tests.test_qwen21_current_failed_adapter import CurrentFailedAdapterTests
        with mock.patch.dict(os.environ, PHASE):
            case = CurrentFailedAdapterTests("test_actual_current_head_source_closure_and_mutations")
            case.setUp()
            case.test_actual_current_head_source_closure_and_mutations()

    def test_exact_phase_and_selector_bindings_and_old_config_unchanged(self):
        frozen = current.read_json(current.CONFIG)
        with mock.patch.dict(os.environ, {"QWEN_IMAGE_2_1_LORA_PHASE": "current-diagnostic"}):
            self.assertEqual(current.config(), frozen)
        with mock.patch.dict(os.environ, PHASE):
            expected = copy.deepcopy(frozen)
            expected["selector"] = current.TRAJECTORY_SELECTOR
            self.assertEqual(current.config(), expected)
            self.assertEqual(current.receipt_path(Path("evidence")).parts[-2], "current-trajectory")
        workflow = yaml.safe_load(inline_text())
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["qwen_image_2_1_lora_phase"]["default"], "full")
        job = workflow["jobs"]["mlx-qwen-image-2-1"]
        self.assertEqual(job["env"]["QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB"], "100")
        steps = {s.get("name"): s for s in job["steps"]}
        for name in ("Prove trained velocity survives adapter save and reload", "Materialize hash-pinned transferred adapters"):
            self.assertIn("!= 'current-trajectory'", steps[name]["if"])

    def test_exact_source_admission_and_forbidden_scope_mutants(self):
        source = ROOT / "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src"
        parent = (source / "conditioning_velocity_diagnostic.rs").read_text(encoding="utf-8")
        child = (source / "conditioning_velocity_trajectory.rs").read_text(encoding="utf-8")
        changed = ["crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/conditioning_velocity_trajectory.rs",
                   "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/conditioning_trajectory_math.rs",
                   "scripts/tests/test_qwen21_trajectory.py"]
        with mock.patch.dict(os.environ, PHASE):
            cfg = current.config()
            def git(_root, *args):
                if args == ("rev-parse", "HEAD"): return "a" * 40
                if args[0] == "rev-parse" and args[1].endswith("^{tree}"): return cfg["source"]["baseTree"]
                if args[0] == "status": return ""
                if args[0] == "diff": return "\n".join(changed)
                if args[0] == "show": return child if args[1].endswith("conditioning_velocity_trajectory.rs") else parent
                if args[0] == "rev-parse": return current.PRODUCTION_BLOBS[args[1].split(":", 1)[1]]
                raise AssertionError(args)
            with mock.patch.object(current, "git", side_effect=git):
                self.assertEqual(current.validate_source_closure(ROOT, "a" * 40, cfg), changed)
                for path in ("crates/media/mlx-gen/src/sampler.rs", "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/unrelated.rs"):
                    changed.append(path)
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, "allowlist"):
                        current.validate_source_closure(ROOT, "a" * 40, cfg)
                    changed.pop()
                for selector in (current.TRAJECTORY_SELECTOR.replace("::trajectory", "::wrong"),
                                 current.TRAJECTORY_SELECTOR + "_missing"):
                    mutant = copy.deepcopy(cfg); mutant["selector"] = selector
                    with self.subTest(selector=selector), self.assertRaises(ValueError):
                        current.validate_source_closure(ROOT, "a" * 40, mutant)
                with mock.patch.dict(os.environ, {"QWEN_IMAGE_2_1_LORA_PHASE": "current-diagnostic"}):
                    with self.assertRaisesRegex(ValueError, "exact named phase"):
                        current.validate_source_closure(ROOT, "a" * 40, cfg)

    def test_real_shell_route_and_zero_test_refusal_without_native_execution(self):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not Path(bash).exists(): self.skipTest("bash unavailable")
        script = ROOT / "scripts/ci/real-weights/mlx-qwen-image-2-1/run-the-qwen-image-2-1-lora-real-weight-gates.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root / "current-trajectory").mkdir()
            # Shell functions are boundary doubles: exercise the exact production route, never cargo/MLX.
            for count, expected in ((1, 0), (0, 1)):
                body = ('cargo() { printf "%s\\n" "$*" > "$QWEN_IMAGE_2_1_RENDER_OUT/cargo-args"; '
                        f'echo "test result: ok. {count} passed"; }}; '
                        'python3.12() { printf "%s\\n" "$*" > "$QWEN_IMAGE_2_1_RENDER_OUT/python-args"; }; '
                        f'source "{script.as_posix()}"')
                env = os.environ | PHASE | {"QWEN_IMAGE_2_1_RENDER_OUT": root.as_posix()}
                row = subprocess.run([bash, "-c", body], env=env, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(row.returncode, expected, row.stderr)
                self.assertIn(current.TRAJECTORY_SELECTOR + " -- --ignored --exact", (root / "cargo-args").read_text(encoding="utf-8"))
                self.assertEqual((root / "current-trajectory/selector.exit-code").read_text(encoding="utf-8").strip(), str(expected))
                self.assertIn("finish", (root / "python-args").read_text(encoding="utf-8"))
                self.assertFalse((root / "current-diagnostic").exists())

    def test_completion_rejects_count_cleanup_acceptance_and_identity_mutants(self):
        env = PHASE | {"GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        good = {"status": "DIAGNOSTIC_COMPLETED", "qualityAcceptance": None, "donorAccepted": False,
                "sourceCandidate": "a" * 40, "runId": 123, "runAttempt": 1,
                "forwardCount": 32, "trajectoryCount": 4, "endpointDecodeCount": 4,
                "steps": 8, "seed": 24163, "guidance": 1, "negativeBranch": False, "sampler": "Euler",
                "cleanup": {"nativeRetired": True, "watchdogJoined": True, "postDropReadback": True,
                            "previousMemoryLimitBytes": 10, "restoredMemoryLimitBytes": 10,
                            "previousCacheLimitBytes": 5, "restoredCacheLimitBytes": 5},
                "historicalEndpointIdentity": False,
                "trajectoryQualification": "SOURCE_EQUIVALENT_RECONSTRUCTION_ONLY_ENDPOINT_MISMATCH",
                "trajectories": [{"tier": tier, "adapted": adapted, "steps": [{}] * 8, "metrics": [{}] * 8,
                                   "endpoint": {"exactHistoricalPngMatch": False}}
                                  for tier, adapted in (("denseBF16", False), ("denseBF16", True), ("Q4", False), ("Q4", True))]}
        with mock.patch.dict(os.environ, env):
            current.validate_trajectory_receipt(good)
            for key, value in (("forwardCount", 31), ("endpointDecodeCount", 3),
                               ("qualityAcceptance", True), ("seed", 42), ("historicalEndpointIdentity", True)):
                mutant = copy.deepcopy(good); mutant[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): current.validate_trajectory_receipt(mutant)
            for key, value in (("postDropReadback", False), ("restoredCacheLimitBytes", 6)):
                mutant = copy.deepcopy(good); mutant["cleanup"][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "restoration"):
                    current.validate_trajectory_receipt(mutant)

    def test_physical_capture_digest_finiteness_inventory_and_endpoint_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = self.write_admission(root)
            def vector(name, shape):
                elements = __import__("math").prod(shape)
                payload = b"\0" * (elements * 4)
                path = root / (name + ".f32"); path.write_bytes(payload)
                return {"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                        "shape": shape, "dtype": "Float32", "elements": elements, "bytes": len(payload)}
            shape = [1, 2304, 64]
            sigmas = [1 - i / 8 for i in range(9)]
            row = {"inputs": {"sigmas": sigmas, "referenceOrder": ["source99", "palette"],
                              "fit": [[1024, 1024], [1024, 1024]],
                              "conditioningDenseVsQ4": {"meanAbsoluteDifference": 0,
                                                        "rootMeanSquareDifference": 0,
                                                        "maxAbsoluteDifference": 0},
                              "x0": vector("x0", shape), "noise": vector("noise", shape),
                              "conditioning": [vector(f"cond{i}", [1, 1]) for i in range(2)],
                              "references": [{"pixels": vector(f"pixels{i}", [1, 1]),
                                              "latents": vector(f"refs{i}", [1, 4096, 64])} for i in range(2)]},
                   "activePeakBytes": 1, "activeEnvelopeBytes": 2, "physicalPeakBytes": 3, "cpuCachePeakBytes": 4,
                   "trajectories": [], "comparisons": []}
            comparison_keys = ("denseQ4BaseTrajectoryDriftMse", "denseQ4AdaptedTrajectoryDriftMse",
                               "denseAdapterTrajectoryDifferenceMse", "q4AdapterTrajectoryDifferenceMse",
                               "denseTargetErrorGain", "q4TargetErrorGain")
            estimate_keys = ("denseDenoisedEstimateErrorGain", "q4DenoisedEstimateErrorGain")
            for step in range(9):
                row["comparisons"].append({"step": step,
                    **dict.fromkeys(comparison_keys, 0),
                    **dict.fromkeys(estimate_keys, None if step == 8 else 0)})
            for i in range(4):
                endpoint = root / f"endpoint{i}.png"; endpoint.write_bytes(b"identity mismatch fixture")
                row["trajectories"].append({"trajectory": i, "adapted": bool(i % 2),
                    "rawMasterVerifiedTensors": 672 if i % 2 else 0,
                    "steps": [{"step": s, "sigma": sigmas[s], "nextSigma": sigmas[s+1],
                               "x": vector(f"t{i}-s{s}-x", shape), "velocity": vector(f"t{i}-s{s}-v", shape)} for s in range(8)],
                    "metrics": [{"step": s, "nativeArithmeticPass": False, "targetPathError": 0,
                                 "denoisedEstimateError": 0, "updateToTargetProjection": 0,
                                 "updateNorm2": 0, "eulerUpdateResidualMax": 0} for s in range(8)],
                    "finalLatent": vector(f"t{i}-final", shape), "finalLatentTargetError": 0,
                    "endpoint": {"file": endpoint.name, "sha256": current.sha256_file(endpoint),
                                 "expectedSha256": current.TRAJECTORY_ENDPOINTS[i], "exactHistoricalPngMatch": False}})
            current.validate_trajectory_files(root, row)
            for key, value in (("admittedCacheLimitBytes", admission["frozenFreeCacheAllowanceBytes"]),
                               ("predictedMaxFullEnvelopeBytes", admission["reservedPhysicalCeilingBytes"] - 1),
                               ("cacheGrantReadBack", False),
                               ("typedNativeCacheGrantRequired", False),
                               ("cacheGrantRestoresOnDrop", False)):
                mutant = copy.deepcopy(admission); mutant[key] = value
                (root / "numeric-physical-admission.json").write_text(json.dumps(mutant), encoding="utf-8")
                with self.subTest(admission=key), self.assertRaisesRegex(ValueError, "cache|admission"):
                    current.validate_trajectory_files(root, row)
            (root / "numeric-physical-admission.json").write_text(json.dumps(admission), encoding="utf-8")
            # The original validator accepted the first missing-final mutation. Every
            # defined scalar must now be present/finite, with null only at final sigma zero.
            mutations = []
            for i in range(4):
                for value in ("MISSING", float("nan"), float("inf"), None, True):
                    mutant = copy.deepcopy(row)
                    if value == "MISSING": del mutant["trajectories"][i]["finalLatentTargetError"]
                    else: mutant["trajectories"][i]["finalLatentTargetError"] = value
                    mutations.append((f"final-error-{i}-{value}", mutant))
            for key in row["inputs"]["conditioningDenseVsQ4"]:
                for value in ("MISSING", float("nan"), float("inf"), None, True):
                    mutant = copy.deepcopy(row)
                    if value == "MISSING": del mutant["inputs"]["conditioningDenseVsQ4"][key]
                    else: mutant["inputs"]["conditioningDenseVsQ4"][key] = value
                    mutations.append((f"conditioning-{key}-{value}", mutant))
            for step in range(9):
                for key in comparison_keys + estimate_keys:
                    invalid = ("MISSING", float("nan"), float("inf"), True,
                               0 if step == 8 and key in estimate_keys else None)
                    for value in invalid:
                        mutant = copy.deepcopy(row)
                        if value == "MISSING": del mutant["comparisons"][step][key]
                        else: mutant["comparisons"][step][key] = value
                        mutations.append((f"comparison-{step}-{key}-{value}", mutant))
            for value in (None, [], row["comparisons"][:8], row["comparisons"] + [row["comparisons"][-1]],
                          list(reversed(row["comparisons"]))):
                mutant = copy.deepcopy(row); mutant["comparisons"] = value
                mutations.append(("comparison-inventory-or-order", mutant))
            mutant = copy.deepcopy(row); del mutant["comparisons"]
            mutations.append(("comparisons-missing", mutant))
            mutant = copy.deepcopy(row); mutant["comparisons"][1]["step"] = 0
            mutations.append(("comparison-duplicate-step", mutant))
            mutant = copy.deepcopy(row); mutant["comparisons"][0]["step"] = 0.0
            mutations.append(("comparison-noninteger-step", mutant))
            for name, mutant in mutations:
                with self.assertRaisesRegex(ValueError, "metric", msg=name):
                    current.validate_trajectory_files(root, mutant)
            vector_row = row["trajectories"][0]["steps"][0]["velocity"]
            path = root / vector_row["file"]
            original = path.read_bytes(); path.write_bytes(struct.pack("<f", float("nan")) + original[4:])
            with self.assertRaisesRegex(ValueError, "digest"): current.validate_trajectory_files(root, row)
            vector_row["sha256"] = current.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "nonfinite"): current.validate_trajectory_files(root, row)
            path.write_bytes(original); vector_row["sha256"] = current.sha256_file(path)
            row["trajectories"][0]["endpoint"]["exactHistoricalPngMatch"] = True
            with self.assertRaisesRegex(ValueError, "misqualified"): current.validate_trajectory_files(root, row)
