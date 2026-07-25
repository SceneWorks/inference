import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reference/sa3_primitives_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-primitives-reference"


class Sa3PrimitiveReferenceTests(unittest.TestCase):
    def test_fixture_is_independent_and_locked(self):
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["story"], "sc-14536")
        self.assertEqual(
            manifest["upstreamCommit"],
            "124e8a799f57a1f665495ecb72e547d0a62867f1",
        )
        self.assertEqual(
            manifest["environment"],
            {
                "python": "3.12.13",
                "torch": "2.7.1",
                "torchaudio": "2.7.1",
                "transformers": "5.8.0",
            },
        )
        snapshot_lock = json.loads(
            (ROOT / "docs/migration/sa3-reference/snapshot-files.json").read_text(
                encoding="utf-8"
            )
        )
        for oracle_key, lock_key in {
            "small": "small-music",
            "medium": "medium",
            "sameS": "same-s",
            "sameL": "same-l",
        }.items():
            locked = snapshot_lock["snapshots"][lock_key]
            self.assertEqual(
                manifest["sources"][oracle_key]["revision"], locked["revision"]
            )
            self.assertEqual(
                manifest["sources"][oracle_key]["modelSha256"],
                locked["files"]["model.safetensors"]["sha256"],
            )
        artifact = FIXTURE / manifest["artifact"]["file"]
        self.assertEqual(artifact.stat().st_size, manifest["artifact"]["bytes"])
        self.assertEqual(
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            manifest["artifact"]["sha256"],
        )
        tensors = manifest["artifact"]["tensors"]
        for name in [
            "branch_qk_ln.q_norm.weight",
            "branch_qk_ln.q_norm.bias",
            "branch_qk_ln.k_norm.weight",
            "branch_qk_ln.k_norm.bias",
            "branch_rms.gamma",
            "branch_rms_output",
        ]:
            self.assertIn(name, tensors)
        self.assertNotIn("branch_qk_ln.q_norm.gamma", tensors)
        self.assertNotIn("branch_qk_ln.q_norm.beta", tensors)
        # This story intentionally does not mutate the sc-14534 eight-artifact manifest.
        old = json.loads(
            (ROOT / "docs/migration/sa3-reference/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(old["artifacts"]), 8)

    def test_verifier_accepts_fixture_and_rejects_corruption(self):
        subprocess.run(
            [sys.executable, SCRIPT, "--verify", "--output", FIXTURE],
            check=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            (target / "manifest.json").write_bytes(
                (FIXTURE / "manifest.json").read_bytes()
            )
            artifact_name = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8")
            )[
                "artifact"
            ]["file"]
            data = bytearray((FIXTURE / artifact_name).read_bytes())
            data[-1] ^= 1
            (target / artifact_name).write_bytes(data)
            failed = subprocess.run(
                [sys.executable, SCRIPT, "--verify", "--output", target],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("hash mismatch", failed.stderr + failed.stdout)

    def test_verifier_rejects_provenance_and_environment_mutations(self):
        original = json.loads(
            (FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        mutations = [
            ("story", lambda m: m.__setitem__("story", "sc-99999"), "story mismatch"),
            (
                "source revision",
                lambda m: m["sources"]["small"].__setitem__("revision", "0" * 40),
                "source provenance mismatch",
            ),
            (
                "source hash",
                lambda m: m["sources"]["sameL"].__setitem__(
                    "modelSha256", "0" * 64
                ),
                "source provenance mismatch",
            ),
            (
                "environment",
                lambda m: m["environment"].__setitem__("torch", "0.0.0"),
                "environment mismatch",
            ),
            (
                "upstream lock",
                lambda m: m["upstreamLock"].__setitem__("sha256", "0" * 64),
                "upstream lock mismatch",
            ),
        ]
        for label, mutate, error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                target = Path(temp)
                manifest = copy.deepcopy(original)
                mutate(manifest)
                (target / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                failed = subprocess.run(
                    [sys.executable, SCRIPT, "--verify", "--output", target],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(error, failed.stderr + failed.stdout)


if __name__ == "__main__":
    unittest.main()
