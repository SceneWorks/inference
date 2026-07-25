import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "reference" / "sa3_text_reference.py"
SPEC = importlib.util.spec_from_file_location("sa3_text_reference", MODULE_PATH)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)
FIXTURE = ROOT / "docs" / "migration" / "sa3-text-reference"


class StableAudio3TextReferenceTest(unittest.TestCase):
    def test_committed_oracle_is_provenance_and_payload_locked(self):
        REFERENCE.verify(FIXTURE)
        manifest = json.loads(
            (FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [item["rawTokenCount"] for item in manifest["prompts"]], [3, 17, 394]
        )
        self.assertEqual(
            [item["validTokenCount"] for item in manifest["prompts"]], [3, 17, 256]
        )
        self.assertEqual(
            [item["truncated"] for item in manifest["prompts"]],
            [False, False, True],
        )
        self.assertEqual(
            manifest["snapshot"]["textFiles"]["model.safetensors"],
            {
                "bytes": 1_183_022_944,
                "sha256": REFERENCE.TEXT_MODEL_SHA256,
            },
        )
        probe = manifest["truncationProbe"]
        self.assertEqual(probe, REFERENCE.EXPECTED_TRUNCATION_PROBE)
        self.assertTrue(
            set(probe["retainedBoundaryTokenIds"]).isdisjoint(
                probe["discardedTailTokenIds"]
            )
        )

    def test_provenance_and_artifact_mutations_fail_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["transformers"] = "5.8.1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "runtime mismatch"
            ):
                REFERENCE.verify(copy)

        for section, field, value, message in [
            ("upstream", "commit", "0" * 40, "upstream provenance mismatch"),
            ("snapshot", "revision", "0" * 40, "snapshot provenance mismatch"),
        ]:
            with self.subTest(section=section, field=field):
                with tempfile.TemporaryDirectory() as temp:
                    copy = Path(temp) / "oracle"
                    shutil.copytree(FIXTURE, copy)
                    manifest_path = copy / "manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest[section][field] = value
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        REFERENCE.InvalidReference, message
                    ):
                        REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["snapshot"]["textFiles"]["model.safetensors"]["sha256"] = (
                "0" * 64
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "snapshot provenance mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prompts"][2]["validTokenCount"] = 255
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "prompt metadata mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["truncationProbe"]["discardedTailTokenIds"] = manifest[
                "truncationProbe"
            ]["retainedBoundaryTokenIds"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "truncation probe mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tensorArtifact"]["bytes"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "artifact contract mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cpuTensorArtifact"]["computePolicy"] = "BF16 compute"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "CPU tensor artifact contract mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            artifact = copy / "text.safetensors"
            with artifact.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\0")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "artifact identity mismatch"
            ):
                REFERENCE.verify(copy)

        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "oracle"
            shutil.copytree(FIXTURE, copy)
            artifact = copy / "text-cpu-f32.safetensors"
            with artifact.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\0")
            with self.assertRaisesRegex(
                REFERENCE.InvalidReference, "CPU tensor artifact identity mismatch"
            ):
                REFERENCE.verify(copy)


if __name__ == "__main__":
    unittest.main()
