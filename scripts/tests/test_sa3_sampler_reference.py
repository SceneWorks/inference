import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.reference import sa3_sampler_reference as reference


class SamplerReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.source = cls.root / "docs/migration/sa3-sampler-reference"

    def test_committed_reference_verifies(self):
        reference.verify(self.source)

    def test_payload_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(self.source, target, dirs_exist_ok=True)
            payload = json.loads((target / "sampler.json").read_text(encoding="utf-8"))
            payload["schedules"]["partialStrength"][1] += 0.125
            (target / "sampler.json").write_bytes(reference.canonical(payload))
            with self.assertRaises(reference.InvalidReference):
                reference.verify(target)

    def test_coupled_payload_and_manifest_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(self.source, target, dirs_exist_ok=True)
            artifact = target / "sampler.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["contract"]["field"] = "v(x,t)=0"
            artifact.write_bytes(reference.canonical(payload))
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifactSha256"] = reference.sha256(artifact)
            manifest_path.write_bytes(reference.canonical(manifest))
            with self.assertRaises(reference.InvalidReference):
                reference.verify(target)

    def test_manifest_and_contract_mutations_are_rejected(self):
        mutations = [
            ("story", "sc-wrong"),
            ("upstreamCommit", "0" * 40),
            ("sourceSha256", {}),
            ("artifact", "missing.json"),
            ("artifactSha256", "0" * 64),
        ]
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                shutil.copytree(self.source, target, dirs_exist_ok=True)
                manifest_path = target / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest[key] = value
                manifest_path.write_bytes(reference.canonical(manifest))
                with self.assertRaises((reference.InvalidReference, FileNotFoundError, IsADirectoryError)):
                    reference.verify(target)

    def test_p0_manifest_artifact_snapshot_and_config_mutations_are_rejected(self):
        mutations = [
            ("manifestSha256", "0" * 64),
            ("artifactSha256", "0" * 64),
            ("snapshotRevision", "0" * 40),
            ("modelConfigSha256", "0" * 64),
        ]
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                shutil.copytree(self.source, target, dirs_exist_ok=True)
                manifest_path = target / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if key == "manifestSha256":
                    manifest["p0Reference"][key] = value
                else:
                    manifest["p0Reference"]["cases"]["small-music"][key] = value
                manifest_path.write_bytes(reference.canonical(manifest))
                with self.assertRaises(reference.InvalidReference):
                    reference.verify(target)

    def test_generator_is_reproducible_against_frozen_checkout(self):
        upstream = Path("/private/tmp/sa3-upstream-review")
        runtime = Path("/private/tmp/stable-audio-3-sc14535/.venv/bin/python")
        if not upstream.is_dir() or not runtime.is_file():
            self.skipTest("frozen Stable Audio 3 checkout/runtime unavailable")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            subprocess.run(
                [
                    str(runtime),
                    str(self.root / "scripts/reference/sa3_sampler_reference.py"),
                    "--generate",
                    "--upstream",
                    str(upstream),
                    "--output",
                    str(target),
                ],
                check=True,
            )
            self.assertEqual(
                (target / "sampler.json").read_bytes(),
                (self.source / "sampler.json").read_bytes(),
            )
            self.assertEqual(
                (target / "manifest.json").read_bytes(),
                (self.source / "manifest.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
