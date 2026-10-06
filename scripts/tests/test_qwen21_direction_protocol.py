"""CPU-only phase isolation and immutable failed-donor safety mutants."""

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.ci import qwen21_velocity_adapter as donor
from scripts.ci import qwen21_direction_receipts as evidence
from scripts.tests.test_qwen21_lora_phases import ROOT, shell_path
from scripts.tests import test_qwen21_lora_phases as phases, test_qwen21_q4_replay as wiring


def tensor_fixture(metadata):
    header, payload = {"__metadata__": metadata}, bytearray()
    for block in range(32):
        for path in ("attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
                     "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"):
            shapes = [[64, 64], [64, 16], [16, 64]]
            if path in ("img_mlp.gate_layer", "img_mlp.proj"):
                shapes = [[96, 64], [128, 16], [16, 64]]
            if path == "img_mlp.out":
                shapes = [[64, 96], [64, 16], [16, 128]]
            for suffix, shape in zip(("w1", "w2_a", "w2_b"), shapes):
                start = len(payload); payload.extend(b"\0" * (math.prod(shape) * 4))
                header[f"transformer_blocks.{block}.{path}.lokr_{suffix}"] = {
                    "dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


def training_fixture():
    metadata = {"family": "qwen-image-2-1", "baseModel": "qwen_image_2_1", "trainingMode": "edit",
                "networkType": "lokr", "rank": "16", "alpha": "16",
                "license": "Qwen Research License Agreement (research/evaluation only)",
                "modelspec.license": "Qwen Research License Agreement (research/evaluation only)"}
    return {"training": {"steps": 120, "stepsRun": 120, "losses": [0.1] * 120,
        "stepSamples": [{"step": i, "loss": 0.1} for i in range(1, 121)],
        "trainingStageTraceComplete": True, "adapterSha256": donor.ADAPTER_SHA, "adapterBytes": 6759417,
        "dataset": {"sha256": donor.DATASET}, "metadata": metadata, "rank": 16,
        "networkType": "Lokr", "learningRate": struct.unpack("<f", struct.pack("<f", 0.0001))[0],
        "resolution": 512, "editProtocol": {"trainingReferenceCount": 1, "evaluationReferenceCount": 2,
            "trainingCaption": donor.CAPTION, "trainingTargetEdge": 512, "trainingReferenceFittedEdge": 1024,
            "stepsRequested": 120, "evaluationKeySha256": "da3be3ca711ec3d0e89df8f1e91bc662365fea188fd513d0108782b6c726189f",
            "dataset": {"sha256": donor.DATASET}, "trainingDataRecipe": {"version": "balanced64-v1",
            "heldoutSource99UsedForTraining": False}}}}, metadata


class FailedDonorTests(unittest.TestCase):
    def test_manifest_is_exact_and_every_identity_mutant_fails(self):
        original = json.loads((ROOT / "scripts/ci/qwen21_velocity_adapter.json").read_text(encoding="utf-8"))
        donor.validate_manifest(original)
        mutants = []
        for key in original:
            bad = copy.deepcopy(original); bad[key] = "changed"; mutants.append(bad)
        for index in range(2):
            for key in original["adapters"][index]:
                bad = copy.deepcopy(original); bad["adapters"][index][key] = "changed"; mutants.append(bad)
        bad = copy.deepcopy(original); bad["acceptanceEvidence"] = 0; mutants.append(bad)
        for bad in mutants:
            with self.subTest(mutant=bad), self.assertRaises(ValueError):
                donor.validate_manifest(bad)

    def test_training_receipt_mutants_fail(self):
        receipt, metadata = training_fixture(); donor.validate_training(receipt, metadata)
        for key, value in [("stepsRun", 119), ("losses", [float("nan")] * 120),
                           ("trainingStageTraceComplete", False), ("dataset", {"sha256": "wrong"}),
                           ("adapterSha256", "wrong"), ("rank", 8), ("networkType", "Lora"),
                           ("learningRate", 0.001), ("resolution", 768)]:
            bad = copy.deepcopy(receipt); bad["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                donor.validate_training(bad, metadata)
        for key in ("family", "baseModel", "license", "modelspec.license", "trainingMode", "alpha"):
            bad = dict(metadata); bad[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                donor.validate_training(receipt, bad)

    def test_exact672_schema_finiteness_offsets_and_metadata(self):
        _, metadata = training_fixture()
        original = tensor_fixture(metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"; path.write_bytes(original)
            self.assertEqual(donor.read_adapter(path), metadata)
            size = struct.unpack("<Q", original[:8])[0]
            for mutation in ("missing", "axis", "overlap", "nan"):
                header = json.loads(original[8:8 + size]); payload = bytearray(original[8 + size:])
                first = next(k for k in header if k != "__metadata__")
                if mutation == "missing": del header[first]
                if mutation == "axis": header[first]["shape"] = [128, 32]
                if mutation == "overlap": header[first]["data_offsets"] = [4, 16388]
                if mutation == "nan": payload[:4] = struct.pack("<f", float("nan"))
                encoded = json.dumps(header).encode(); path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    donor.read_adapter(path)

    def test_draft_ownership_is_required_even_for_cached_bytes(self):
        for key, value in [("draft", False), ("id", 1), ("url", "https://api.github.com/repos/other/repo/releases/402708057")]:
            release = {"id": 402708057, "draft": True,
                       "url": "https://api.github.com/repos/SceneWorks/inference/releases/402708057", "assets": []}
            release[key] = value
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "redacted-test"}):
                path = Path(directory) / "adapters"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unpublished"):
                    donor.prepare(copy.deepcopy(donor.MANIFEST), path, "a" * 40,
                                  lambda request: io.BytesIO(json.dumps(release).encode()))
                self.assertFalse((path / "velocity-adapter-resolved.json").exists())
                self.assertFalse((path.parent / "DIAGNOSTIC_ONLY.json").exists())
                self.assertEqual(list(path.parent.glob("velocity-stage-*")), [])

    def test_complete_authenticated_prepare_and_failed_hash_never_publishes(self):
        receipt, metadata = training_fixture()
        adapter_bytes = tensor_fixture(metadata)
        adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
        receipt["training"]["adapterSha256"] = adapter_sha
        receipt_bytes = json.dumps(receipt).encode()
        entries = copy.deepcopy(donor.ENTRIES)
        for entry, raw in zip(entries, (adapter_bytes, receipt_bytes)):
            entry["size"] = len(raw); entry["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = copy.deepcopy(donor.MANIFEST); manifest["adapters"] = entries
        requests = []
        corrupt = False
        def fetch(request):
            requests.append(request)
            if request.full_url.endswith("/releases/402708057"):
                return io.BytesIO(json.dumps({"id": 402708057, "draft": True,
                    "url": "https://api.github.com/repos/SceneWorks/inference/releases/402708057",
                    "assets": [{"id": e["assetId"], "name": e["assetName"], "size": e["size"]} for e in entries]}).encode())
            return io.BytesIO(b"wrong" if corrupt else adapter_bytes if request.full_url.endswith("/611294721") else receipt_bytes)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "redacted-test"}), \
                patch.multiple(donor, MANIFEST=manifest, ENTRIES=entries, ADAPTER_SHA=adapter_sha, RECEIPT_SHA=entries[1]["sha256"]):
            destination = Path(directory) / "adapters"
            resolved = donor.prepare(manifest, destination, "a" * 40, fetch)
            data = json.loads(resolved.read_text(encoding="utf-8"))
            self.assertEqual(data["kind"], "DIAGNOSTIC_ONLY")
            self.assertIs(data["acceptanceEvidence"], False)
            self.assertEqual(len(data["adapters"]), 1)
            self.assertEqual(data["trainingProvenance"], donor.PROVENANCE)
            self.assertTrue(all(r.get_header("Authorization") == "Bearer redacted-test" for r in requests))
            self.assertEqual(len(requests), 3)
            corrupt = True
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                donor.prepare(manifest, destination, "a" * 40, fetch)
            self.assertFalse(resolved.exists())
            self.assertFalse((destination.parent / "DIAGNOSTIC_ONLY.json").exists())
            self.assertEqual(list(destination.parent.glob("velocity-stage-*")), [])


class DirectionPhaseTests(phases.PhaseTests):
    def run_direction(self, *, fail="", zero=False, script=None, fail_exit=101):
        # The fake Python records receipt verification while Cargo records actual routing.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.sh"
            source = (ROOT / self.script_name()).read_text(encoding="utf-8") if script is None else script
            path.write_text('python3.12() { echo "receipt-validator:$*"; if [[ "$*" == *"--style-only"* ]]; then [[ "${@: -1}" == 0 || "${@: -1}" == 101 ]]; fi; }; export -f python3.12\n' + source, encoding="utf-8")
            with patch.dict(os.environ, {"FAIL_EXIT": str(fail_exit)}):
                return self.run_phase("direction-protocol", fail=fail, zero=zero, script=shell_path(path))

    def script_name(self):
        return "scripts/ci/real-weights/mlx-qwen-image-2-1/run-the-qwen-image-2-1-lora-real-weight-gates.sh"

    def assert_direction_calls(self, names, text):
        self.assertEqual(names, ["diagnostic_reused_t2i_style_direction"])
        self.assertEqual(len(text.splitlines()), 2)
        self.assertIn('--lib conditioning_velocity_diagnostic::diagnostic_dense_q4_conditioning_velocity -- --ignored --exact --nocapture --test-threads 1', text)
        self.assertNotIn("probe=1", text)
        self.assertNotIn("trains_", text)
        self.assertNotIn("stacked_adapters", text)

    def test_direction_runs_only_serial_style_and_velocity_and_retains_both_failures(self):
        for fail, zero in [("", False), ("diagnostic_reused", False), ("diagnostic_dense", False)]:
            result, names, text = self.run_direction(fail=fail, zero=zero)
            self.assert_direction_calls(names, text)
            self.assertEqual(result.returncode == 0, not fail and not zero, result.stderr)
            self.assertIn("qwen21_direction_receipts", result.stdout)

    def test_crash_incomplete_or_zero_style_aborts_before_velocity(self):
        for code, zero in [(137, False), (1, False), (0, True)]:
            result, names, text = self.run_direction(fail="diagnostic_reused" if not zero else "", fail_exit=code, zero=zero)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(names, ["diagnostic_reused_t2i_style_direction"])
            self.assertNotIn("conditioning_velocity_diagnostic", text)

    def test_direction_selector_routing_mutants(self):
        source = (ROOT / self.script_name()).read_text(encoding="utf-8")
        for old, new in [("conditioning_velocity_diagnostic::diagnostic_dense_q4_conditioning_velocity", "conditioning_velocity_diagnostic::wrong_test"),
                         ("--lib \\", "--test integration \\")]:
            result, names, text = self.run_direction(script=source.replace(old, new))
            with self.subTest(mutation=old), self.assertRaises(AssertionError):
                self.assert_direction_calls(names, text)

    def test_workflow_group_permissions_and_materializer_env(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/real-weights.yml").read_text(encoding="utf-8"))
        self.assertIn("direction-protocol", workflow[True]["workflow_dispatch"]["inputs"]["qwen_image_2_1_lora_phase"]["options"])
        self.assertIs(workflow["concurrency"]["cancel-in-progress"], False)
        self.assertEqual(workflow["jobs"]["mlx-qwen-image-2-1"]["permissions"],
                         {"actions": "read", "contents": "read"})
        helper = wiring.WiringTests()
        result, calls, env = helper.run_script(ROOT / "scripts/ci/real-weights/mlx-qwen-image-2-1/materialize-hash-pinned-transferred-adapters.sh", "direction-protocol")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qwen21_adapter_imports.py", calls)
        self.assertIn("-m scripts.ci.qwen21_velocity_adapter", calls)
        self.assertNotIn("qwen21_diagnostic_adapter", calls)
        self.assertNotIn("qwen21_q4_replay", calls)
        self.assertIn("QWEN_IMAGE_2_1_IMPORT_MANIFEST=", env)
        self.assertIn("QWEN_IMAGE_2_1_VELOCITY_MANIFEST=", env)
        result, calls, _ = helper.run_script(ROOT / "scripts/ci/real-weights/mlx-qwen-image-2-1/build-and-record-mlx-library-identity.sh", "direction-protocol")
        self.assertEqual(result.returncode, 0, result.stderr)
        helper.assert_build_calls(calls)
        result, calls, _ = helper.run_script(ROOT / "scripts/ci/real-weights/mlx-qwen-image-2-1/prove-trained-velocity-survives-save-and-reload.sh", "direction-protocol")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("cargo test"), 1, "retain original12; no extra acceptance captures")


class ReceiptTests(unittest.TestCase):
    def fixture(self, root):
        source = "a" * 40
        style = root / "style-protocol"; style.mkdir()
        rows = []
        for tier in ("bf16", "q8", "q4"):
            request = {"prompt": "zxq style, a lighthouse on a rocky coast at dusk", "width": 768,
                       "height": 768, "steps": 8, "seed": 24163, "count": 1, "conditioningCount": 0}
            rows.append({"tier": tier, "mode": "t2i", "adapter": "mlx_t2i_1000_steps",
                "adapterSha256": evidence.STYLE_SHA, "requestProtocol": "original-1000-step-style",
                "request": request, "meanAbsDiff": 3, "paletteDistanceBase": 4,
                "paletteDistanceAdapted": 2, "paletteDistanceGain": 2})
            for suffix in ("style_base", "style_mlx_t2i_1000_steps"):
                (style / f"{tier}_{suffix}.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        style_receipt = {"purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False, "retrain": False,
                         "sourceCandidate": source, "trainingProvenance": evidence.STYLE_PROVENANCE,
                         "renderCount": 6, "renders": rows}
        for name in ("DIAGNOSTIC_ONLY", "style-direction"):
            (style / f"{name}.json").write_text(json.dumps(style_receipt), encoding="utf-8")
        controls = {"kind": "DIAGNOSTIC_ONLY", "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False, "sourceCandidate": source,
                    "renderCount": 6, "donorSha256": evidence.STYLE_SHA, "samplerJoined": True,
                    "safetyChecksComplete": True, "foregroundRetired": True,
                    "directionFailures": [], "stageResult": "passed"}
        sample = {"unixMillis": 1, "physFootprintBytes": 1, "physicalCeilingBytes": 100,
                  "pressureLevel": 1, "reclaimableBytes": 10}
        trace = (json.dumps(sample) + "\n").encode()
        (style / "physical-allocator-samples.jsonl").write_bytes(trace)
        controls.update({"githubRunId": os.environ.get("GITHUB_RUN_ID"), "sampleTrace": "physical-allocator-samples.jsonl",
                         "startedUnixMillis": 1,
                         "sampleTraceSha256": hashlib.sha256(trace).hexdigest(), "sampleCount": 1,
                         "finalSafetySample": {**sample, "unixMillis": 2}})
        (style / "stage-controls.json").write_text(json.dumps(controls), encoding="utf-8")
        manifest = root / "manifest.json"; manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")
        directory = root / "velocity-discriminator"; (directory / "velocities").mkdir(parents=True)
        raw = b"\0" * 589824
        vectors, states = [], []
        for state, (conditioning, dit) in enumerate((("denseBF16", "denseBF16"), ("Q4", "Q4"), ("denseBF16", "Q4"), ("Q4", "denseBF16"))):
            pairs = []
            for repeat in (0, 1):
                pairs.append({"repeat": repeat, "baseError": 1, "adaptedError": 2, "learnedGain": -1,
                              "projection": 0, "deltaNorm2": 1, "cpuIdentityResidual": 0, "cpuIdentityBound": 0.001})
                for adapted in (False, True):
                    file = f"velocities/state-{state}-repeat-{repeat}-{'adapted' if adapted else 'base'}.f32"
                    (directory / file).write_bytes(raw)
                    vectors.append({"state": state, "repeat": repeat, "adapted": adapted, "file": file,
                        "sha256": hashlib.sha256(raw).hexdigest(), "dtype": "Float32", "shape": [1, 2304, 64],
                        "elements": 147456, "bytes": 589824})
            states.append({"state": state, "conditioning": conditioning, "dit": dit, "pairs": pairs})
        velocity = {"kind": "DIAGNOSTIC_ONLY", "accepted": False, "acceptanceEvidence": False,
            "sourceCandidate": source, "sourceBase": donor.SOURCE, "discriminatorProtocolSha256": evidence.PROTOCOL,
            "trainingProvenance": donor.PROVENANCE, "inputManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "forwardCount": 16, "stateCount": 4, "repeatCount": 2, "adapterStrength": 1, "sigma": 0.5,
            "arithmeticBoundVerdict": "UNPROVEN_RELAXED_NAX_PRECISION", "cpuCachePeakBytes": 100,
            "vectors": vectors, "states": states}
        (directory / "receipt.json").write_text(json.dumps(velocity), encoding="utf-8")
        (root / "direction-velocity.exit-code").write_text("0\n", encoding="utf-8")
        return source, manifest, style_receipt, controls, velocity

    def test_negative_velocity_gain_is_observation_not_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, manifest, _, _, _ = self.fixture(root)
            result = evidence.validate(root, source, manifest, 0)
            self.assertIs(result["accepted"], False)
            self.assertEqual(result["velocityForwardCount"], 16)

    def test_velocity_provenance_counts_arithmetic_and_vector_mutants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, manifest, _, _, receipt = self.fixture(root)
            for key, value in [("sourceCandidate", "b" * 40), ("accepted", True), ("acceptanceEvidence", 0),
                               ("arithmeticBoundVerdict", "PASS"), ("forwardCount", 15), ("repeatCount", 1),
                               ("sigma", 0.7), ("adapterStrength", 2), ("cpuCachePeakBytes", 67108865)]:
                mutant = copy.deepcopy(receipt); mutant[key] = value
                (root / "velocity-discriminator/receipt.json").write_text(json.dumps(mutant), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evidence.validate(root, source, manifest, 0)
            for mutation in ("duplicate", "path", "shape", "hash"):
                mutant = copy.deepcopy(receipt)
                if mutation == "duplicate": mutant["vectors"][1] = mutant["vectors"][0]
                if mutation == "path": mutant["vectors"][0]["file"] = "../escape.f32"
                if mutation == "shape": mutant["vectors"][0]["shape"] = [1, 3, 16]
                if mutation == "hash": mutant["vectors"][0]["sha256"] = "0" * 64
                (root / "velocity-discriminator/receipt.json").write_text(json.dumps(mutant), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    evidence.validate(root, source, manifest, 0)

    def test_only_completed_style_direction_failure_may_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, _, receipt, controls, _ = self.fixture(root)
            row = receipt["renders"][2]; row["paletteDistanceGain"] = -1
            controls["stageResult"] = "direction_failed"
            controls["directionFailures"] = [{"tier": "q4", "criterion": "palette_gain", "actual": -1, "floor": 1}]
            (root / "style-protocol/style-direction.json").write_text(json.dumps(receipt), encoding="utf-8")
            path = root / "style-protocol/stage-controls.json"; path.write_text(json.dumps(controls), encoding="utf-8")
            self.assertEqual(evidence.validate_style(root, source, 101)["styleStageResult"], "direction_failed")
            for code in (0, 1, 137):
                with self.subTest(code=code), self.assertRaises(ValueError):
                    evidence.validate_style(root, source, code)
            for key in ("safetyChecksComplete", "foregroundRetired", "samplerJoined"):
                bad = dict(controls); bad[key] = False; path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evidence.validate_style(root, source, 101)

    def test_hash_correct_nonfinite_velocity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, manifest, _, _, receipt = self.fixture(root)
            vector = receipt["vectors"][0]
            path = root / "velocity-discriminator" / vector["file"]
            raw = struct.pack("<f", float("nan")) + path.read_bytes()[4:]
            path.write_bytes(raw); vector["sha256"] = hashlib.sha256(raw).hexdigest()
            (root / "velocity-discriminator/receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                evidence.validate(root, source, manifest, 0)

    def test_trace_run_identity_and_final_pressure_mutants_abort_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, _, _, controls, _ = self.fixture(root)
            path = root / "style-protocol/stage-controls.json"
            controls["githubRunId"] = "123"
            path.write_text(json.dumps(controls), encoding="utf-8")
            with patch.dict(os.environ, {"GITHUB_RUN_ID": "123"}):
                evidence.validate_style(root, source, 0)
            for key, value in [("sampleTraceSha256", "0" * 64), ("sampleCount", 0),
                               ("sampleTrace", "../foreign.jsonl"), ("githubRunId", "999")]:
                mutant = copy.deepcopy(controls); mutant[key] = value
                path.write_text(json.dumps(mutant), encoding="utf-8")
                with patch.dict(os.environ, {"GITHUB_RUN_ID": "123"}), self.subTest(key=key), self.assertRaises(ValueError):
                    evidence.validate_style(root, source, 0)
            mutant = copy.deepcopy(controls); mutant["finalSafetySample"]["pressureLevel"] = 2
            path.write_text(json.dumps(mutant), encoding="utf-8")
            with patch.dict(os.environ, {"GITHUB_RUN_ID": "123"}), self.assertRaisesRegex(ValueError, "safety"):
                evidence.validate_style(root, source, 0)


if __name__ == "__main__":
    unittest.main()
