"""CPU-only provenance, authenticated byte transfer and phase isolation for Q4 diagnosis."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.ci.qwen21_adapter_imports import materialize, validate_manifest
from scripts.ci.qwen21_q4_replay import prepare, validate_replay
from scripts.tests.test_qwen21_lora_phases import BASH, ROOT, shell_path

LANE = ROOT / "scripts/ci/real-weights/mlx-qwen-image-2-1"
MANIFEST = ROOT / "scripts/ci/qwen21_q4_replay.json"


class ReplayTests(unittest.TestCase):
    def manifest(self):
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_frozen_replay_manifest_and_every_identity_mutant(self):
        original = self.manifest()
        validate_replay(original)
        validate_manifest(original)
        mutants = []
        for key in original.keys() - {"adapters"}:
            bad = copy.deepcopy(original)
            bad[key] = True if key == "acceptanceEvidence" else "changed"
            mutants.append(bad)
        for index in range(2):
            for key in original["adapters"][index]:
                bad = copy.deepcopy(original)
                bad["adapters"][index][key] = "changed"
                mutants.append(bad)
        bad = copy.deepcopy(original); bad["adapters"].pop(); mutants.append(bad)
        bad = copy.deepcopy(original); bad["adapters"].reverse(); mutants.append(bad)
        for bad in mutants:
            with self.subTest(mutant=bad), self.assertRaises(ValueError):
                validate_replay(bad)
        # No metadata/download side effects for a changed donor or replay protocol.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "fixture"}):
            with self.assertRaises(ValueError):
                prepare(mutants[0], Path(directory), lambda _: self.fail("invalid provenance fetched"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_exact_replay_wrong_bytes_cannot_install_or_seal(self):
        manifest = self.manifest()
        calls = []
        def fetch(request):
            calls.append(request)
            if request.full_url.endswith("/releases/402708057"):
                return io.BytesIO(json.dumps({"assets": [
                    {"id": row["assetId"], "name": row["assetName"], "size": row["size"]}
                    for row in manifest["adapters"]]}).encode())
            return io.BytesIO(bytes(manifest["adapters"][0]["size"]))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "fixture"}):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare(manifest, Path(directory), fetch)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(r.get_header("Authorization") == "Bearer fixture" for r in calls))
        self.assertTrue(calls[1].full_url.endswith("/releases/assets/610462396"))

    def test_shared_png_transfer_checks_auth_ownership_hash_and_verified_reuse(self):
        payload = b"sanitized replay bytes"
        manifest = self.manifest()
        manifest["adapters"] = [manifest["adapters"][0]]
        row = manifest["adapters"][0]
        row["size"] = len(payload); row["sha256"] = hashlib.sha256(payload).hexdigest()
        requests = []
        def fetch(request):
            requests.append(request)
            if request.full_url.endswith("/releases/402708057"):
                return io.BytesIO(json.dumps({"assets": [{"id": row["assetId"],
                    "name": row["assetName"], "size": row["size"]}]}).encode())
            return io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GH_TOKEN": "fixture"}):
            root = Path(directory)
            materialize(manifest, root, fetch)
            materialize(manifest, root, fetch)
            self.assertEqual((root / row["file"]).read_bytes(), payload)
            self.assertEqual(len(requests), 3, "verified reuse still checks release ownership")
            self.assertTrue(all(r.get_header("Authorization") == "Bearer fixture" for r in requests))
            for assets in ([], [{"id": row["assetId"], "name": "other.png", "size": row["size"]}]):
                with self.assertRaisesRegex(ValueError, "does not own"):
                    materialize(manifest, root, lambda _: io.BytesIO(json.dumps({"assets": assets}).encode()))

    def test_bypassed_download_hash_mutant_is_rejected_by_the_byte_test(self):
        # Simulate an implementation that trusts the expected digest instead of hashing bytes.
        expected = self.manifest()["adapters"][0]["sha256"]
        with patch("scripts.ci.qwen21_adapter_imports.sha256", return_value=expected):
            with self.assertRaises(AssertionError):
                self.test_exact_replay_wrong_bytes_cannot_install_or_seal()

    def test_png_permission_does_not_relax_adapter_or_acceptance_manifests(self):
        original = self.manifest()
        for key, value in [("purpose", "ACCEPTANCE"), ("acceptanceEvidence", True)]:
            bad = copy.deepcopy(original); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(bad)
        for key, value in [("file", "../q4_edit_base.png"), ("kind", "lokr"),
                           ("name", "arbitrary_png"), ("assetName", "../other.png")]:
            bad = copy.deepcopy(original); bad["adapters"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(bad)


@unittest.skipUnless(BASH and Path(BASH).is_file(), "requires bash")
class WiringTests(unittest.TestCase):
    def run_script(self, script, phase, zero=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cargo", "python3.12"):
                path = root / name
                path.write_text('#!/bin/bash\n'
                    f'printf "{name} %s\\n" "$*" >> "$CALLS"\n'
                    'if [[ "$ZERO_TESTS" == 1 && "$*" == *fully_packed* ]]; then '
                    'echo "test result: ok. 0 passed"; else echo "test result: ok. 1 passed"; fi\n', encoding="utf-8")
                path.chmod(0o755)
            calls = root / "calls"; env = root / "env"
            result = subprocess.run([BASH, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
                'export PATH="$FAKE_BIN:$PATH"; source ' + shell_path(script)], cwd=ROOT,
                env={**os.environ, "FAKE_BIN": shell_path(root), "CALLS": shell_path(calls),
                     "GITHUB_ENV": shell_path(env), "QWEN_IMAGE_2_1_RENDER_OUT": shell_path(root),
                     "QWEN_IMAGE_2_1_LORA_PHASE": phase, "ZERO_TESTS": "1" if zero else "0"},
                capture_output=True, text=True, encoding="utf-8", check=False)
            return result, calls.read_text(encoding="utf-8"), env.read_text(encoding="utf-8") if env.exists() else ""

    def assert_build_calls(self, calls):
        self.assertIn('--test integration --no-run --message-format=json', calls)
        self.assertIn('--lib --no-run --message-format=json', calls)
        self.assertIn('--test-target lib --out', calls)

    def assert_fixture_calls(self, calls):
        self.assertEqual(calls.count("cargo test"), 2)
        old = 'training::tests::trained_factors_preserve_velocity_through_save_and_reload'
        packed = 'training::tests::fully_packed_trained_factors_preserve_direct_delta_through_export'
        self.assertLess(calls.index(old), calls.index(packed))
        self.assertIn('--lib ' + packed + ' -- --exact --nocapture --test-threads 1', calls)

    def test_numeric_build_and_fixture_only_with_retained_original_proof(self):
        build = LANE / "build-and-record-mlx-library-identity.sh"
        proof = LANE / "prove-trained-velocity-survives-save-and-reload.sh"
        for phase in ("probe", "diagnostic", "q4-numeric", "edit", "imports", "full"):
            result, calls, _ = self.run_script(build, phase)
            self.assertEqual(result.returncode, 0, result.stderr)
            if phase == "q4-numeric":
                self.assert_build_calls(calls)
            else:
                self.assertNotIn('--lib --no-run', calls)
            result, calls, _ = self.run_script(proof, phase)
            self.assertEqual(result.returncode, 0, result.stderr)
            if phase == "q4-numeric":
                self.assert_fixture_calls(calls)
            else:
                self.assertEqual(calls.count("cargo test"), 1)
        result, _, _ = self.run_script(proof, "q4-numeric", zero=True)
        self.assertNotEqual(result.returncode, 0, "a renamed packed fixture must fail closed")

    def test_build_and_fixture_routing_mutants(self):
        cases = [
            ("build-and-record-mlx-library-identity.sh", '--lib --no-run', '--test integration --no-run', self.assert_build_calls),
            ("prove-trained-velocity-survives-save-and-reload.sh", 'training::tests::fully_packed_trained_factors_preserve_direct_delta_through_export', 'training::tests::wrong_fixture', self.assert_fixture_calls),
        ]
        for script, old, new, detector in cases:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                source = (LANE / script).read_text(encoding="utf-8")
                self.assertIn(old, source)
                mutant = Path(directory) / "mutant.sh"
                mutant.write_text(source.replace(old, new), encoding="utf-8")
                _, calls, _ = self.run_script(mutant, "q4-numeric")
                with self.assertRaises((AssertionError, ValueError)):
                    detector(calls)

    def test_numeric_materialization_is_same_donor_plus_exact_replay_env(self):
        script = LANE / "materialize-hash-pinned-transferred-adapters.sh"
        for phase in ("diagnostic", "q4-numeric"):
            result, calls, env = self.run_script(script, phase)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('-m scripts.ci.qwen21_diagnostic_adapter', calls)
            self.assertIn('QWEN_IMAGE_2_1_DIAGNOSTIC_MANIFEST=', env)
            self.assertNotIn('qwen21_adapter_imports.py', calls)
            if phase == "q4-numeric":
                self.assertIn('-m scripts.ci.qwen21_q4_replay --manifest scripts/ci/qwen21_q4_replay.json', calls)
                self.assertIn('QWEN_IMAGE_2_1_Q4_REPLAY_DIR=', env)
            else:
                self.assertNotIn('qwen21_q4_replay', calls)
                self.assertNotIn('Q4_REPLAY_DIR', env)

    def test_workflow_retains_bounded_phase_permission_and_fixture_before_models(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/real-weights.yml").read_text(encoding="utf-8"))
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["qwen_image_2_1_lora_phase"]["options"],
                         ["probe", "diagnostic", "q4-numeric", "direction-protocol",
                          "current-diagnostic", "edit", "imports", "full"])
        job = workflow["jobs"]["mlx-qwen-image-2-1"]
        self.assertEqual(job["timeout-minutes"], 300)
        self.assertEqual(job["permissions"], {"actions": "read", "contents": "read"})
        names = [s.get("name") for s in job["steps"]]
        self.assertLess(names.index("Build the Qwen-Image 2.1 MLX test binary"), names.index("Prove trained velocity survives adapter save and reload"))
        self.assertLess(names.index("Prove trained velocity survives adapter save and reload"), names.index("Materialize and verify immutable snapshots"))
        transfer = next(s for s in job["steps"] if s.get("name") == "Materialize hash-pinned transferred adapters")
        self.assertEqual(transfer["if"], "inputs.qwen_image_2_1_lora_phase != 'probe' && inputs.qwen_image_2_1_lora_phase != 'current-diagnostic'")
        self.assertEqual(transfer["env"]["GH_TOKEN"], "${{ github.token }}")


if __name__ == "__main__":
    unittest.main()
