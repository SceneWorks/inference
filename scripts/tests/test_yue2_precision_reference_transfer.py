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

    def test_pinned_relay_preserves_original_teacher_and_refuses_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_zip, source_run, source_artifact, _, _, pinned = self.fixture(root)
            source_constants = {
                "SOURCE_ZIP_SHA256": TRANSFER.digest(source_zip.read_bytes()),
                "REFERENCE_SHA256": pinned["vae_real_reference.safetensors"][1],
                "SOURCE_METADATA_SHA256": pinned["reference-provenance.json"][1],
                "LICENSE_SHA256": pinned["NONCOMMERCIAL.txt"][1], "FILES": pinned,
            }
            with patch.multiple(TRANSFER, **source_constants):
                first = root / "first-transfer"
                TRANSFER.transfer(source_zip, source_run, source_artifact,
                                  TRANSFER.RELAY_ENGINE_SHA, TRANSFER.RELAY_CONTROL_SHA,
                                  str(TRANSFER.RELAY_RUN_ID), "1", first)
                relay_zip = root / "relay.zip"
                with zipfile.ZipFile(relay_zip, "w") as target:
                    for name in TRANSFER.RELAY_FILES:
                        target.write(first / name, name)
                relay_files = {name: (len((first / name).read_bytes()),
                                      TRANSFER.digest((first / name).read_bytes()))
                               for name in TRANSFER.RELAY_FILES}
                relay_constants = {"RELAY_ZIP_SHA256": TRANSFER.digest(relay_zip.read_bytes()),
                                   "RELAY_METADATA_SHA256": relay_files["reference-provenance.json"][1],
                                   "RELAY_FILES": relay_files}
                relay_run = {"id": TRANSFER.RELAY_RUN_ID, "run_attempt": 1,
                             "head_sha": TRANSFER.RELAY_CONTROL_SHA, "conclusion": "success",
                             "event": "workflow_dispatch", "name": "YuE2 targeted stage-precision proof",
                             "repository": {"id": 1299380446, "full_name": "SceneWorks/inference"}}
                relay_artifact = {"id": TRANSFER.RELAY_ARTIFACT_ID,
                                  "name": "yue2-precision-reference",
                                  "digest": "sha256:" + relay_constants["RELAY_ZIP_SHA256"],
                                  "expired": False,
                                  "workflow_run": {"id": TRANSFER.RELAY_RUN_ID,
                                                   "head_sha": TRANSFER.RELAY_CONTROL_SHA,
                                                   "repository_id": 1299380446}}
                relay_run_file, relay_artifact_file = root / "relay-run.json", root / "relay-artifact.json"
                relay_run_file.write_text(json.dumps(relay_run), encoding="utf-8")
                relay_artifact_file.write_text(json.dumps(relay_artifact), encoding="utf-8")
                with patch.multiple(TRANSFER, **relay_constants):
                    result = TRANSFER.transfer(relay_zip, relay_run_file, relay_artifact_file,
                                               "c" * 40, "d" * 40, "125", "1",
                                               root / "second-transfer", "relay")
                    self.assertEqual(result["source_artifact_zip_sha256"],
                                     source_constants["SOURCE_ZIP_SHA256"])
                    self.assertEqual(result["relay_artifact_zip_sha256"],
                                     relay_constants["RELAY_ZIP_SHA256"])
                    self.assertEqual((root / "second-transfer/vae_real_reference.safetensors").read_bytes(),
                                     b"fixed CPU reference")
                    for changed_run in ({**relay_run, "head_sha": "e" * 40},
                                        {**relay_run, "conclusion": "failure"}):
                        relay_run_file.write_text(json.dumps(changed_run), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "run identity"):
                            TRANSFER.transfer(relay_zip, relay_run_file, relay_artifact_file,
                                              "c" * 40, "d" * 40, "126", "1", root / "bad", "relay")
                    relay_run_file.write_text(json.dumps(relay_run), encoding="utf-8")
                    relay_artifact_file.write_text(json.dumps({**relay_artifact, "expired": True}),
                                                   encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "artifact transport"):
                        TRANSFER.transfer(relay_zip, relay_run_file, relay_artifact_file,
                                          "c" * 40, "d" * 40, "126", "1", root / "bad", "relay")
                    relay_artifact_file.write_text(json.dumps(relay_artifact), encoding="utf-8")
                    with zipfile.ZipFile(relay_zip, "a") as changed:
                        changed.writestr("extra", b"unexpected")
                    with self.assertRaisesRegex(ValueError, "ZIP digest"):
                        TRANSFER.transfer(relay_zip, relay_run_file, relay_artifact_file,
                                          "c" * 40, "d" * 40, "126", "1", root / "bad", "relay")
                wrong_chain = dict(json.loads((first / "reference-provenance.json").read_text(encoding="utf-8")))
                wrong_chain["source_artifact_id"] = 1
                mutated = root / "mutated-relay.zip"
                with zipfile.ZipFile(mutated, "w") as target:
                    for name in TRANSFER.RELAY_FILES:
                        payload = ((json.dumps(wrong_chain) + "\n").encode()
                                   if name == "reference-provenance.json" else (first / name).read_bytes())
                        target.writestr(name, payload)
                with zipfile.ZipFile(mutated) as archive:
                    mutated_files = {name: (len(archive.read(name)), TRANSFER.digest(archive.read(name)))
                                     for name in TRANSFER.RELAY_FILES}
                mutated_zip_sha = TRANSFER.digest(mutated.read_bytes())
                relay_artifact_file.write_text(json.dumps({**relay_artifact,
                    "digest": "sha256:" + mutated_zip_sha}), encoding="utf-8")
                with patch.multiple(TRANSFER, RELAY_ZIP_SHA256=mutated_zip_sha,
                                    RELAY_METADATA_SHA256=mutated_files["reference-provenance.json"][1],
                                    RELAY_FILES=mutated_files):
                    with self.assertRaisesRegex(ValueError, "relay provenance"):
                        TRANSFER.transfer(mutated, relay_run_file, relay_artifact_file,
                                          "c" * 40, "d" * 40, "126", "1", root / "bad", "relay")


if __name__ == "__main__":
    unittest.main()
