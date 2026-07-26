import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.reference.sa3_chunked_autoencoder_reference import (
    EXPECTED_ARTIFACT_BYTES,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_OUTPUTS_BYTES,
    EXPECTED_OUTPUTS_SHA256,
    InvalidReference,
    require_snapshot_files,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reference/sa3_chunked_autoencoder_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-chunked-reference"


class Sa3ChunkedAutoencoderReferenceTests(unittest.TestCase):
    def test_snapshot_authentication_hashes_loaded_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            revision = "a" * 40
            snapshot = Path(temporary) / revision
            snapshot.mkdir()
            (snapshot / "model_config.json").write_bytes(b"config")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            lock = {
                "files": {
                    name: {"sha256": hashlib.sha256(value).hexdigest()}
                    for name, value in (
                        ("model_config.json", b"config"),
                        ("model.safetensors", b"weights"),
                    )
                }
            }
            self.assertEqual(
                require_snapshot_files(snapshot, revision, "test", lock),
                {
                    "model_config.json": hashlib.sha256(b"config").hexdigest(),
                    "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
                },
            )
            (snapshot / "model.safetensors").write_bytes(b"mutated")
            with self.assertRaisesRegex(InvalidReference, "model.safetensors hash mismatch"):
                require_snapshot_files(snapshot, revision, "test", lock)

    def test_fixture_locks_frozen_plan_noise_and_seam_evidence(self):
        manifest_path = FIXTURE / "manifest.json"
        artifact = FIXTURE / "chunked-f32.safetensors"
        outputs = FIXTURE / "chunked-outputs-f16.safetensors"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            EXPECTED_MANIFEST_SHA256,
        )
        self.assertEqual(artifact.stat().st_size, EXPECTED_ARTIFACT_BYTES)
        self.assertEqual(
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )
        self.assertEqual(outputs.stat().st_size, EXPECTED_OUTPUTS_BYTES)
        self.assertEqual(
            hashlib.sha256(outputs.read_bytes()).hexdigest(),
            EXPECTED_OUTPUTS_SHA256,
        )
        self.assertEqual(
            manifest["upstream"]["commit"],
            "124e8a799f57a1f665495ecb72e547d0a62867f1",
        )
        self.assertEqual(
            manifest["snapshots"]["same_s"],
            {
                "revision": "fbeb3dcf53a326e5682f38e22e7f740202d44232",
                "configSha256": "c329dd0a6f61d0b3ea4f23930059a6c00437005692fed4310924eb286253303a",
                "modelSha256": "c19698ce3a0b462acb967ee495e9eb7945221f236968c50206cce8cf22b3d305",
            },
        )
        self.assertEqual(
            manifest["snapshots"]["same_l"],
            {
                "revision": "41acf79dd242877d6499a1108ca5dba5d5eecfc5",
                "configSha256": "9ca55d03d868b8f6769cd7d678430a6b70c328db24350601903a92001bd4f9f1",
                "modelSha256": "8cf93d5345c559307fc429bde896f594516fd6a1a23014c2f6055f78d8e13378",
            },
        )
        self.assertEqual(manifest["plan"]["starts"], [0, 96, 97])
        self.assertEqual(
            [entry["output"] for entry in manifest["plan"]["ownership"]],
            [[0, 112], [112, 113], [113, 225]],
        )
        self.assertEqual(set(manifest["cases"]), {"same_s", "same_l"})
        self.assertEqual(manifest["cases"]["same_s"]["encodeNoise"], [])
        self.assertEqual(
            [entry["stream"] for entry in manifest["cases"]["same_l"]["encodeNoise"]],
            [0, 1, 2],
        )
        for case in manifest["cases"].values():
            self.assertEqual(
                [entry["stream"] for entry in case["decodeNoise"]],
                [100, 101, 102, 103, 104, 105],
            )
            self.assertEqual(len(case["spectralBoundariesZeroNoise"]), 2)
        p0 = json.loads(
            (ROOT / "docs/migration/sa3-reference/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        full_configs = [
            snapshot["consumedConfig"]
            for snapshot in p0["snapshots"]
            if snapshot["kind"] == "dit"
        ]
        self.assertEqual(len(full_configs), 6)
        self.assertTrue(all(config["pretransform"]["chunked"] for config in full_configs))

    def test_resource_evidence_locks_exact_head_hardware_and_reductions(self):
        evidence_path = FIXTURE / "resource-evidence.json"
        self.assertEqual(
            hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "32a6cbcf52d06d0e0dbb737459fc2438fbaaab9b10fc9c3bf40d8f364b4bf63d",
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["story"], "sc-14540")
        self.assertEqual(
            evidence["measuredCommit"],
            "424e90166386d02bf95bdaa57e958e839693418a",
        )
        self.assertEqual(evidence["metalWorkflow"]["runId"], 30194850972)
        self.assertEqual(evidence["metalWorkflow"]["result"], "success")
        self.assertEqual(evidence["metal"]["hardware"]["chip"], "Apple M5 Max")
        self.assertEqual(evidence["geometry"]["latentLength"], 1292)
        for case in ("same_s", "same_l"):
            for operation in ("encode", "decode"):
                measurement = evidence["metal"][case][operation]
                self.assertLess(
                    measurement["chunkedPeakDeviceBytes"],
                    measurement["directPeakDeviceBytes"],
                )
                self.assertLess(measurement["wallTimeRatio"], 2.0)

    def test_verifier_accepts_repository_pins_and_rejects_plan_mutation(self):
        subprocess.run(
            [sys.executable, SCRIPT, "--verify", "--output", FIXTURE], check=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            manifest = json.loads(
                (FIXTURE / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["plan"]["starts"][-1] = 98
            (target / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for name in ("chunked-f32.safetensors", "chunked-outputs-f16.safetensors"):
                (target / name).symlink_to(FIXTURE / name)
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("manifest repository pin mismatch", failed.stderr)

    def test_verifier_rejects_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "manifest.json").symlink_to(FIXTURE / "manifest.json")
            artifact = target / "chunked-f32.safetensors"
            shutil.copyfile(FIXTURE / "chunked-f32.safetensors", artifact)
            (target / "chunked-outputs-f16.safetensors").symlink_to(
                FIXTURE / "chunked-outputs-f16.safetensors"
            )
            with artifact.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("F32 artifact manifest hash mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
