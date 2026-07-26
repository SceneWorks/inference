import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reference"))

from scripts.reference.sa3_small_music_provider_reference import (
    EXPECTED_ARTIFACT_BYTES,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TENSORS,
    inspect_safetensors,
)


SCRIPT = ROOT / "scripts/reference/sa3_small_music_provider_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-small-music-provider-reference"


class Sa3SmallMusicProviderReferenceTests(unittest.TestCase):
    def test_fixture_locks_provider_identity_geometry_and_tensors(self):
        manifest_path = FIXTURE / "manifest.json"
        artifact = FIXTURE / "provider-output.safetensors"
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
        self.assertEqual(inspect_safetensors(artifact), EXPECTED_TENSORS)
        self.assertEqual(manifest["story"], "sc-14543")
        self.assertEqual(
            manifest["snapshot"]["revision"],
            "0fef1392cd842149a2b6d445e181c97608faac06",
        )
        self.assertEqual(manifest["output"]["frames"], 1_323_000)
        self.assertEqual(len(manifest["noise"]["draws"]), 17)

    def test_verifier_accepts_repository_fixture(self):
        subprocess.run(
            [sys.executable, SCRIPT, "--verify", "--output", FIXTURE], check=True
        )

    def test_verifier_rejects_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            manifest = json.loads(
                (FIXTURE / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["request"]["steps"] = 7
            (target / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (target / "provider-output.safetensors").symlink_to(
                FIXTURE / "provider-output.safetensors"
            )
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("provider manifest repository pin mismatch", failed.stderr)

    def test_verifier_rejects_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "manifest.json").symlink_to(FIXTURE / "manifest.json")
            artifact = target / "provider-output.safetensors"
            shutil.copyfile(FIXTURE / "provider-output.safetensors", artifact)
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
            self.assertIn("provider artifact repository pin mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
