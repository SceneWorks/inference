import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reference/sa3_same_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-same-s-reference"


class Sa3SameReferenceTests(unittest.TestCase):
    def fixture_copy(self, target, manifest, mutable_artifact=None):
        (target / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (target / "two-stage-config.json").symlink_to(
            FIXTURE / "two-stage-config.json"
        )
        for artifact in manifest["artifacts"]:
            name = artifact["file"]
            destination = target / name
            if name == mutable_artifact:
                destination.write_bytes((FIXTURE / name).read_bytes())
            else:
                destination.symlink_to(FIXTURE / name)

    def assert_verifier_rejects(self, manifest, message):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            self.fixture_copy(target, manifest)
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(message, failed.stderr + failed.stdout)

    def test_fixture_locks_architecture_provenance_noise_and_music(self):
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["story"], "sc-14538")
        self.assertEqual(
            manifest["upstream"]["commit"],
            "124e8a799f57a1f665495ecb72e547d0a62867f1",
        )
        self.assertEqual(
            manifest["environment"],
            {"machine": "arm64", "system": "Darwin", "torchDevice": "cpu"},
        )
        self.assertEqual(
            manifest["architecture"],
            {
                "alignmentSamples": 8192,
                "attentionTokens": 34,
                "blocks": 6,
                "latentDim": 256,
                "midpointShiftTokens": 17,
                "midpointSplit": [3, 3],
                "patchSize": 256,
                "ropePositions": [0, 33],
                "stride": 16,
                "subchunkTokens": 17,
            },
        )
        self.assertEqual(
            manifest["snapshots"]["standalone"]["prefix"],
            "",
        )
        self.assertEqual(
            manifest["snapshots"]["embedded"]["prefix"],
            "pretransform.model.",
        )
        self.assertEqual(manifest["music"]["sampleRate"], 44_100)
        self.assertEqual(manifest["music"]["channels"], 2)
        self.assertEqual(manifest["music"]["samples"], 441_000)
        self.assertEqual(manifest["music"]["license"], "Public Domain Mark 1.0")
        self.assertGreaterEqual(
            manifest["music"]["quality"]["snrDb"],
            manifest["music"]["bounds"]["snrDbMinimum"],
        )
        self.assertLessEqual(
            manifest["music"]["quality"]["mrstft"],
            manifest["music"]["bounds"]["mrstftMaximum"],
        )

        expected = {
            "oracle.safetensors": [
                "encoder.folded_input",
                "encoder.expanded_tokens",
                "encoder.block_0",
                "encoder.block_5",
                "encoder.selected_segments",
                "encoder.output",
                "decoder.folded_input",
                "decoder.expanded_tokens",
                "decoder.block_0",
                "decoder.block_5",
                "decoder.selected_segments",
                "decoder.output",
                "regularization_noise",
                "decoder_mask_noise",
                "decoded",
                "stride8.encoder.folded_input",
                "stride8.encoder.expanded_tokens",
                "stride8.encoder.block_0",
                "stride8.encoder.block_5",
                "stride8.encoder.selected_segments",
                "stride8.encoder.output",
                "stride8.decoder.folded_input",
                "stride8.decoder.expanded_tokens",
                "stride8.decoder.block_0",
                "stride8.decoder.block_5",
                "stride8.decoder.selected_segments",
                "stride8.decoder.output",
                "stride8.latents",
                "stride8.regularization_noise",
                "stride8.decoder_mask_noise",
                "stride8.decoded",
                "stride8.perturbation_unit",
                "stride8.perturbation_1e-6.decoded",
                "stride8.perturbation_3e-6.decoded",
            ],
            "music-roundtrip.safetensors": [
                "audio",
                "latents",
                "regularization_noise",
                "decoder_mask_noise",
                "decoded_padded",
            ],
            "two-stage.safetensors": [
                "audio",
                "order24.encoder0.folded_input",
                "order24.encoder0.expanded_tokens",
                "order24.encoder0.block_0",
                "order24.encoder0.selected_segments",
                "order24.encoder0.output",
                "order24.encoder1.folded_input",
                "order24.encoder1.expanded_tokens",
                "order24.encoder1.block_0",
                "order24.encoder1.selected_segments",
                "order24.encoder1.output",
                "order24.latents",
                "order24.decoder0.folded_input",
                "order24.decoder1.folded_input",
                "order24.decoded",
                "order42.encoder0.folded_input",
                "order42.encoder1.folded_input",
                "order42.decoder0.folded_input",
                "order42.decoder1.folded_input",
                "order42.latents",
                "order42.decoded",
                "weights.bottleneck.noise_scaling_factor",
            ],
        }
        self.assertEqual(
            {artifact["file"]: len(artifact["tensors"]) for artifact in manifest["artifacts"]},
            {
                "oracle.safetensors": 80,
                "music-roundtrip.safetensors": 5,
                "two-stage.safetensors": 161,
            },
        )
        for artifact in manifest["artifacts"]:
            path = FIXTURE / artifact["file"]
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
            )
            for name in expected[artifact["file"]]:
                self.assertIn(name, artifact["tensors"])

    def test_verifier_accepts_fixture(self):
        subprocess.run(
            [sys.executable, SCRIPT, "--verify", "--output", FIXTURE],
            check=True,
        )

    def test_verifier_rejects_payload_header_and_inventory_corruption(self):
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        mutations = [
            (
                lambda data: data.__setitem__(-1, data[-1] ^ 1),
                "tensor record mismatch",
            ),
            (
                lambda data: data.__setitem__(
                    slice(*self.find_bytes(data, b"sc-14538")),
                    b"sc-99999",
                ),
                "safetensors metadata mismatch",
            ),
            (
                lambda data: data.__setitem__(
                    slice(*self.find_bytes(data, b'"audio"')),
                    b'"audix"',
                ),
                "tensor inventory mismatch",
            ),
            (
                lambda data: data.__setitem__(
                    slice(*self.find_bytes(data, b'"shape":[1,2,16384]')),
                    b'"shape":[1,3,16384]',
                ),
                "tensor offset mismatch",
            ),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                target = Path(temp)
                self.fixture_copy(target, manifest, "oracle.safetensors")
                path = target / "oracle.safetensors"
                data = bytearray(path.read_bytes())
                mutate(data)
                path.write_bytes(data)
                failed = subprocess.run(
                    [sys.executable, SCRIPT, "--verify", "--output", target],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(message, failed.stderr + failed.stdout)

    @staticmethod
    def find_bytes(data, needle):
        start = data.index(needle)
        return start, start + len(needle)

    def test_verifier_rejects_every_evidence_class_mutation(self):
        original = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        mutations = [
            (
                lambda manifest: manifest.__setitem__("unexpected", True),
                "top-level inventory mismatch",
            ),
            (
                lambda manifest: manifest.__setitem__("story", "sc-99999"),
                "identity mismatch",
            ),
            (
                lambda manifest: manifest["upstream"].__setitem__(
                    "commit", "0" * 40
                ),
                "upstream provenance mismatch",
            ),
            (
                lambda manifest: manifest["upstream"]["files"].__setitem__(
                    "autoencoders.py", "0" * 64
                ),
                "upstream provenance mismatch",
            ),
            (
                lambda manifest: manifest["runtime"].__setitem__("torch", "0.0.0"),
                "runtime provenance mismatch",
            ),
            (
                lambda manifest: manifest["environment"].__setitem__(
                    "machine", "x86_64"
                ),
                "generation environment mismatch",
            ),
            (
                lambda manifest: manifest["snapshots"]["standalone"].__setitem__(
                    "revision", "0" * 40
                ),
                "snapshot provenance mismatch",
            ),
            (
                lambda manifest: manifest["snapshots"]["embedded"].__setitem__(
                    "modelSha256", "0" * 64
                ),
                "snapshot provenance mismatch",
            ),
            (
                lambda manifest: manifest["architecture"].__setitem__("stride", 8),
                "architecture provenance mismatch",
            ),
            (
                lambda manifest: manifest["syntheticTwoStage"].__setitem__(
                    "configSha256", "0" * 64
                ),
                "synthetic provenance mismatch",
            ),
            (
                lambda manifest: manifest["syntheticTwoStage"].__setitem__(
                    "overrideOrders", [[2, 4]]
                ),
                "synthetic provenance mismatch",
            ),
            (
                lambda manifest: manifest["backendSensitivity"]["scales"][
                    "1e-6"
                ].__setitem__("cosine", 1.0),
                "backend sensitivity provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"].__setitem__(
                    "sourceSha256", "0" * 64
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"].__setitem__(
                    "sourceUrl", "https://example.invalid/audio.ogg"
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"].__setitem__(
                    "license", "unknown"
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"].__setitem__(
                    "offsetSamples", 0
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"]["quality"].__setitem__(
                    "snrEquation", "wrong"
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"]["quality"].__setitem__(
                    "resolutions", []
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["music"]["bounds"].__setitem__(
                    "mrstftMaximum", 999.0
                ),
                "music provenance mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0].__setitem__(
                    "sha256", "0" * 64
                ),
                "frozen record mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0].__setitem__(
                    "bytes", 1
                ),
                "frozen record mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0]["tensors"].pop("audio"),
                "tensor inventory mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0]["tensors"]["audio"].__setitem__(
                    "dtype", "float64"
                ),
                "tensor record mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0]["tensors"]["audio"].__setitem__(
                    "shape", [1, 1, 16_384]
                ),
                "tensor record mismatch",
            ),
            (
                lambda manifest: manifest["artifacts"][0]["tensors"]["audio"].__setitem__(
                    "sha256", "0" * 64
                ),
                "tensor record mismatch",
            ),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                manifest = copy.deepcopy(original)
                mutate(manifest)
                self.assert_verifier_rejects(manifest, message)


if __name__ == "__main__":
    unittest.main()
