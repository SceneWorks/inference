"""CPU-only adversarial checks for the hosted immutable reference transfer."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_precision_reference_transfer as TRANSFER


class ReferenceTransferTests(unittest.TestCase):
    def fixture(self, directory: Path):
        content = b"fixed CPU reference"
        previous = json.dumps({"engine_sha": TRANSFER.SOURCE_ENGINE_SHA,
                               "sha256": TRANSFER.digest(content), "runner": "nax-macos",
                               "source": "pinned YuE2 VAE upstream CPU fixture"}).encode()
        files = {"vae_real_reference.safetensors": content,
                 "reference-provenance.json": previous, "NONCOMMERCIAL.txt": b"license"}
        archive = directory / "source.zip"
        with zipfile.ZipFile(archive, "w") as target:
            for name, value in files.items():
                target.writestr(name, value)
        run = {"id": TRANSFER.SOURCE_RUN_ID, "run_attempt": 1,
               "head_sha": TRANSFER.SOURCE_ENGINE_SHA, "conclusion": "success",
               "event": "workflow_dispatch", "name": "YuE2 targeted stage-precision proof",
               "repository": {"id": 1299380446, "full_name": "SceneWorks/inference"}}
        artifact = {"id": TRANSFER.SOURCE_ARTIFACT_ID, "name": "yue2-precision-reference",
                    "digest": f"sha256:{TRANSFER.digest(archive.read_bytes())}", "expired": False,
                    "workflow_run": {"id": TRANSFER.SOURCE_RUN_ID,
                                     "head_sha": TRANSFER.SOURCE_ENGINE_SHA,
                                     "repository_id": 1299380446}}
        run_file, artifact_file = directory / "run.json", directory / "artifact.json"
        run_file.write_text(json.dumps(run), encoding="utf-8")
        artifact_file.write_text(json.dumps(artifact), encoding="utf-8")
        pinned = {name: (len(value), TRANSFER.digest(value)) for name, value in files.items()}
        return archive, run_file, artifact_file, run, artifact, pinned

    def test_exact_transport_bytes_and_provenance_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, run_file, artifact_file, run, artifact, pinned = self.fixture(root)
            constants = {"SOURCE_ZIP_SHA256": TRANSFER.digest(archive.read_bytes()),
                         "REFERENCE_SHA256": pinned["vae_real_reference.safetensors"][1],
                         "SOURCE_METADATA_SHA256": pinned["reference-provenance.json"][1],
                         "LICENSE_SHA256": pinned["NONCOMMERCIAL.txt"][1], "FILES": pinned}
            with patch.multiple(TRANSFER, **constants):
                output = root / "copied"
                result = TRANSFER.transfer(archive, run_file, artifact_file,
                                           "a" * 40, "b" * 40, "123", "1", output)
                self.assertEqual((output / "vae_real_reference.safetensors").read_bytes(),
                                 b"fixed CPU reference")
                self.assertEqual((output / "NONCOMMERCIAL.txt").read_bytes(), b"license")
                self.assertEqual(result["source_artifact_zip_sha256"], constants["SOURCE_ZIP_SHA256"])
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    TRANSFER.transfer(archive, run_file, artifact_file,
                                      "a" * 40, "b" * 40, "123", "1", output)
                for field, value, reason in (("head_sha", "c" * 40, "run identity"),
                                              ("conclusion", "failure", "run identity")):
                    bad = dict(run, **{field: value})
                    run_file.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, reason):
                        TRANSFER.transfer(archive, run_file, artifact_file,
                                          "a" * 40, "b" * 40, "124", "1", root / "bad")
                run_file.write_text(json.dumps(run), encoding="utf-8")
                for field, value in (("digest", "sha256:" + "0" * 64), ("expired", True)):
                    bad = dict(artifact, **{field: value})
                    artifact_file.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "artifact transport"):
                        TRANSFER.transfer(archive, run_file, artifact_file,
                                          "a" * 40, "b" * 40, "124", "1", root / "bad")
                artifact_file.write_text(json.dumps(artifact), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "run identity"):
                    TRANSFER.transfer(archive, run_file, artifact_file,
                                      "a" * 40, "b" * 40, "123", "2", root / "bad")
                with zipfile.ZipFile(archive, "a") as changed:
                    changed.writestr("../extra", b"unsafe")
                with self.assertRaisesRegex(ValueError, "ZIP digest"):
                    TRANSFER.transfer(archive, run_file, artifact_file,
                                      "a" * 40, "b" * 40, "124", "1", root / "bad")
                changed_digest = TRANSFER.digest(archive.read_bytes())
                artifact_file.write_text(json.dumps({**artifact, "digest": f"sha256:{changed_digest}"}),
                                         encoding="utf-8")
                with patch.object(TRANSFER, "SOURCE_ZIP_SHA256", changed_digest), \
                     self.assertRaisesRegex(ValueError, "archive inventory"):
                    TRANSFER.transfer(archive, run_file, artifact_file,
                                      "a" * 40, "b" * 40, "124", "1", root / "bad")


if __name__ == "__main__":
    unittest.main()
