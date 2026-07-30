import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.reference.sa3_same_l_reference import (
    EXPECTED_ARTIFACT_BYTES,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_EXTENDED_ARTIFACT_BYTES,
    EXPECTED_EXTENDED_ARTIFACT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_OUTPUTS_ARTIFACT_BYTES,
    EXPECTED_OUTPUTS_ARTIFACT_SHA256,
    EXPECTED_RESOURCE_EVIDENCE_BYTES,
    EXPECTED_RESOURCE_EVIDENCE_SHA256,
    safetensors_prefix_digest,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reference/sa3_same_l_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-same-l-reference"


class Sa3SameLReferenceTests(unittest.TestCase):
    def test_fixture_locks_provenance_architecture_and_all_boundary_layers(self):
        manifest_path = FIXTURE / "manifest.json"
        artifact_path = FIXTURE / "same-l.safetensors"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            EXPECTED_MANIFEST_SHA256,
        )
        self.assertEqual(artifact_path.stat().st_size, EXPECTED_ARTIFACT_BYTES)
        self.assertEqual(
            hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )
        extended_path = FIXTURE / "same-l-extended.safetensors"
        outputs_path = FIXTURE / "same-l-outputs-f16.safetensors"
        self.assertEqual(extended_path.stat().st_size, EXPECTED_EXTENDED_ARTIFACT_BYTES)
        self.assertEqual(
            hashlib.sha256(extended_path.read_bytes()).hexdigest(),
            EXPECTED_EXTENDED_ARTIFACT_SHA256,
        )
        self.assertEqual(outputs_path.stat().st_size, EXPECTED_OUTPUTS_ARTIFACT_BYTES)
        self.assertEqual(
            hashlib.sha256(outputs_path.read_bytes()).hexdigest(),
            EXPECTED_OUTPUTS_ARTIFACT_SHA256,
        )
        resource_path = FIXTURE / "resource-evidence.json"
        self.assertEqual(
            resource_path.stat().st_size, EXPECTED_RESOURCE_EVIDENCE_BYTES
        )
        self.assertEqual(
            hashlib.sha256(resource_path.read_bytes()).hexdigest(),
            EXPECTED_RESOURCE_EVIDENCE_SHA256,
        )
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        self.assertEqual(resource["story"], "sc-14539")
        self.assertEqual(resource["backend"], "Cpu")
        self.assertEqual(resource["dtype"], "F32")
        self.assertEqual(
            set(resource["runs"]), {"literal380Seconds", "exactMaximum"}
        )
        self.assertEqual(
            resource["runs"]["literal380Seconds"]["inputSamples"], 16_758_000
        )
        self.assertEqual(
            resource["runs"]["exactMaximum"]["inputSamples"], 16_777_216
        )
        for run in resource["runs"].values():
            self.assertEqual(run["packedLength"], run["latentLength"] * 17)
            self.assertEqual(
                run["paddedSamples"], run["latentLength"] * 4096
            )
            self.assertEqual(len(run["pcmSha256"]), 64)
        self.assertEqual(manifest["story"], "sc-14539")
        self.assertEqual(
            manifest["upstream"]["commit"],
            "124e8a799f57a1f665495ecb72e547d0a62867f1",
        )
        self.assertEqual(
            manifest["runtime"],
            {
                "python": "3.12.13",
                "torch": "2.7.1",
                "torchaudio": "2.7.1",
                "transformers": "5.8.0",
            },
        )
        self.assertEqual(
            manifest["architecture"],
            {
                "depth": 12,
                "dim": 1536,
                "heads": 24,
                "queryTile": 1024,
                "sinusoidalBlocks": list(range(5, 12)),
                "stride": 16,
                "subchunk": 17,
                "window": [17, 17],
            },
        )
        self.assertEqual(
            manifest["embeddedIdentity"],
            {
                "configSha256": "e61e27487e452e8a83d4e6277476b4d14666b14a8d5b41a405b1693b3f2bb2bf",
                "payload": {
                    "bytes": 3_408_509_828,
                    "prefix": "pretransform.model.",
                    "sha256": "a91db184266e0a1874ebde54e53ec6c6ac25d27d6c712968ab22418e6b32b405",
                    "tensors": 472,
                },
            },
        )
        self.assertEqual(
            set(manifest["cases"]),
            {
                "standalone.short",
                "embedded.short",
                "embedded.ten_seconds",
                "embedded.long_120_seconds",
                "standalone.ten_seconds",
                "standalone.long_120_seconds",
                "standalone.stride7",
            },
        )
        tensors = manifest["artifact"]["tensors"]
        self.assertEqual(len(tensors), 354)
        extended_tensors = manifest["extendedArtifact"]["tensors"]
        self.assertEqual(len(extended_tensors), 278)
        tensors = {**tensors, **extended_tensors}
        for case in (
            "standalone.short",
            "embedded.short",
            "standalone.ten_seconds",
            "standalone.long_120_seconds",
            "embedded.ten_seconds",
            "embedded.long_120_seconds",
            "standalone.stride7",
        ):
            for direction in ("encoder", "decoder"):
                for layer in range(12):
                    prefix = f"{case}.{direction}.block_{layer}.slice_"
                    self.assertTrue(
                        any(name.startswith(prefix) for name in tensors),
                        f"missing {prefix}",
                    )
        outputs = manifest["outputsArtifact"]["tensors"]
        self.assertEqual(len(outputs), 14)
        self.assertTrue(all(record["dtype"] == "float16" for record in outputs.values()))
        for case in manifest["cases"]:
            self.assertIn(f"{case}.latents", outputs)
            self.assertIn(f"{case}.decoded", outputs)

    def test_verifier_accepts_only_the_repository_pins(self):
        subprocess.run(
            [sys.executable, SCRIPT, "--verify", "--output", FIXTURE],
            check=True,
        )
        original = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            mutated = json.loads(json.dumps(original))
            mutated["architecture"]["queryTile"] = 2048
            (target / "manifest.json").write_text(
                json.dumps(mutated), encoding="utf-8"
            )
            (target / "same-l.safetensors").symlink_to(
                FIXTURE / "same-l.safetensors"
            )
            (target / "same-l-extended.safetensors").symlink_to(
                FIXTURE / "same-l-extended.safetensors"
            )
            (target / "same-l-outputs-f16.safetensors").symlink_to(
                FIXTURE / "same-l-outputs-f16.safetensors"
            )
            (target / "resource-evidence.json").symlink_to(
                FIXTURE / "resource-evidence.json"
            )
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
            artifact = target / "same-l.safetensors"
            shutil.copyfile(FIXTURE / "same-l.safetensors", artifact)
            (target / "same-l-extended.safetensors").symlink_to(
                FIXTURE / "same-l-extended.safetensors"
            )
            (target / "same-l-outputs-f16.safetensors").symlink_to(
                FIXTURE / "same-l-outputs-f16.safetensors"
            )
            (target / "resource-evidence.json").symlink_to(
                FIXTURE / "resource-evidence.json"
            )
            with artifact.open("r+b") as handle:
                handle.seek(-1, 2)
                byte = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([byte[0] ^ 1]))
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("artifact repository pin mismatch", failed.stderr)

    def test_verifier_rejects_resource_evidence_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for name in (
                "manifest.json",
                "same-l.safetensors",
                "same-l-extended.safetensors",
                "same-l-outputs-f16.safetensors",
            ):
                (target / name).symlink_to(FIXTURE / name)
            resource = json.loads(
                (FIXTURE / "resource-evidence.json").read_text(encoding="utf-8")
            )
            resource["runs"]["exactMaximum"]["peakRssBytes"] += 1
            (target / "resource-evidence.json").write_text(
                json.dumps(resource, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("resource evidence repository pin mismatch", failed.stderr)

    def test_streamed_prefix_digest_ignores_other_namespaces_and_detects_payload_drift(self):
        def write(path, selected_payload):
            entries = {
                "dit.weight": ("F32", [1], b"xxxx"),
                "pretransform.model.a": ("F32", [1], selected_payload),
                "pretransform.model.b": ("I64", [1], b"12345678"),
            }
            offset = 0
            header = {}
            payload = bytearray()
            for name, (dtype, shape, value) in entries.items():
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + len(value)],
                }
                payload.extend(value)
                offset += len(value)
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            write(first, b"abcd")
            write(second, b"abcd")
            expected = safetensors_prefix_digest(first, "pretransform.model.")
            self.assertEqual(
                expected,
                safetensors_prefix_digest(second, "pretransform.model."),
            )
            write(second, b"abce")
            self.assertNotEqual(
                expected["sha256"],
                safetensors_prefix_digest(second, "pretransform.model.")["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
