import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.reference import sa3_sampler_guidance_reference as reference


class SamplerGuidanceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.source = cls.root / "docs/migration/sa3-sampler-reference"

    def test_committed_reference_verifies(self):
        reference.verify(self.source)

    def test_artifact_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(self.source, target, dirs_exist_ok=True)
            artifact = target / reference.ARTIFACT
            data = bytearray(artifact.read_bytes())
            data[-1] ^= 1
            artifact.write_bytes(data)
            with self.assertRaises(reference.InvalidReference):
                reference.verify(target)

    def test_coupled_artifact_manifest_and_tensor_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(self.source, target, dirs_exist_ok=True)
            artifact = target / reference.ARTIFACT
            data = bytearray(artifact.read_bytes())
            data[-1] ^= 1
            artifact.write_bytes(data)
            path = target / "guidance-manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["artifactSha256"] = reference.sha256_file(artifact)
            manifest["tensors"] = reference._records(artifact)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(reference.InvalidReference):
                reference.verify(target)

    def test_coupled_snapshot_lock_and_guidance_model_hash_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(self.source, target / "reference")
            snapshot_lock = target / "snapshot-files.json"
            shutil.copy2(reference.SNAPSHOT_LOCK_PATH, snapshot_lock)
            lock = json.loads(snapshot_lock.read_text(encoding="utf-8"))
            mutated = "0" * 64
            lock["snapshots"]["small-music"]["files"]["model.safetensors"]["sha256"] = mutated
            snapshot_lock.write_text(json.dumps(lock), encoding="utf-8")
            path = target / "reference/guidance-manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["snapshots"]["small-music"]["modelSha256"] = mutated
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                mock.patch.object(reference, "SNAPSHOT_LOCK_PATH", snapshot_lock),
                self.assertRaisesRegex(
                    reference.InvalidReference, "snapshot file lock payload mismatch"
                ),
            ):
                reference.verify(target / "reference")

    def test_provenance_and_control_mutations_are_rejected(self):
        for mutation in (
            "commit",
            "source",
            "runtime",
            "repository",
            "revision",
            "config",
            "model",
            "objective",
            "p0",
            "schedule",
            "padding",
            "negative",
            "control",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                shutil.copytree(self.source, target, dirs_exist_ok=True)
                path = target / "guidance-manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "commit":
                    manifest["upstreamCommit"] = "0" * 40
                elif mutation == "source":
                    manifest["sourceSha256"] = {}
                elif mutation == "runtime":
                    manifest["runtime"]["torch"] = "0"
                elif mutation == "repository":
                    manifest["snapshots"]["small-music"]["repository"] = "wrong/repository"
                elif mutation == "revision":
                    manifest["snapshots"]["small-music"]["revision"] = "0" * 40
                elif mutation == "config":
                    manifest["snapshots"]["small-music"]["modelConfigSha256"] = "0" * 64
                elif mutation == "model":
                    manifest["snapshots"]["small-music"]["modelSha256"] = "0" * 64
                elif mutation == "objective":
                    manifest["snapshots"]["small-music"]["objective"] = "v"
                elif mutation == "p0":
                    manifest["snapshots"]["small-music"]["p0Sha256"] = "0" * 64
                elif mutation == "schedule":
                    manifest["inputs"]["schedule"] = [1.0, 0.0]
                elif mutation == "padding":
                    manifest["inputs"]["paddingValid"] = 16
                elif mutation == "negative":
                    manifest["inputs"]["negativeConditioning"] = "positive"
                else:
                    manifest["inputs"]["variants"]["vanilla"]["cfg_scale"] = 1.0
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(reference.InvalidReference):
                    reference.verify(target)


if __name__ == "__main__":
    unittest.main()
