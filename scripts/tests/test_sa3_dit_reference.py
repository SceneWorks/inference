import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reference/sa3_dit_reference.py"
FIXTURE = ROOT / "docs/migration/sa3-dit-reference"


class Sa3DitReferenceTests(unittest.TestCase):
    def run_verify(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, SCRIPT, "verify", "--output", root],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def fixture_copy(self, target: Path) -> dict:
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(FIXTURE / manifest["artifact"]["file"], target)
        return manifest

    def test_committed_oracle_is_compact_provenance_locked_and_verifiable(self):
        verified = self.run_verify(FIXTURE)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["story"], "sc-14541")
        self.assertEqual(
            manifest["upstream"]["commit"],
            "124e8a799f57a1f665495ecb72e547d0a62867f1",
        )
        self.assertEqual(len(manifest["upstream"]["sourceSha256"]), 5)
        self.assertEqual(len(manifest["artifact"]["tensors"]), 20)
        artifact = FIXTURE / manifest["artifact"]["file"]
        self.assertLess(artifact.stat().st_size, 4 * 1024 * 1024)
        self.assertEqual(
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            manifest["artifact"]["sha256"],
        )
        self.assertEqual(
            manifest["inputs"]["localOrder"],
            ["inpaint_mask", "inpaint_masked_input"],
        )
        self.assertIsNone(manifest["inputs"]["crossMask"])
        self.assertEqual(manifest["inputs"]["paddingSemantics"], "zero-v-only")

    def test_verifier_rejects_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            manifest = self.fixture_copy(target)
            artifact = target / manifest["artifact"]["file"]
            data = bytearray(artifact.read_bytes())
            data[-1] ^= 1
            artifact.write_bytes(data)
            failed = self.run_verify(target)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("artifact hash mismatch", failed.stderr + failed.stdout)

    def test_verifier_rejects_every_provenance_contract_mutation(self):
        original = json.loads(
            (FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        mutations = [
            ("story", lambda m: m.__setitem__("story", "sc-0"), "schema/story"),
            (
                "commit",
                lambda m: m["upstream"].__setitem__("commit", "0" * 40),
                "upstream revision",
            ),
            (
                "source",
                lambda m: m["upstream"]["sourceSha256"].__setitem__(
                    "stable_audio_3/models/dit.py", "0" * 64
                ),
                "source provenance",
            ),
            (
                "snapshot",
                lambda m: m["snapshot"].__setitem__("modelSha256", "0" * 64),
                "snapshot provenance",
            ),
            (
                "runtime",
                lambda m: m["runtime"].__setitem__("torch", "0"),
                "runtime mismatch",
            ),
            (
                "duration",
                lambda m: m["inputs"].__setitem__("secondsTotal", 1.0),
                "input contract",
            ),
            (
                "local order",
                lambda m: m["inputs"].__setitem__(
                    "localOrder", ["inpaint_masked_input", "inpaint_mask"]
                ),
                "input contract",
            ),
            (
                "cross mask",
                lambda m: m["inputs"].__setitem__("crossMask", True),
                "input contract",
            ),
            (
                "padding semantics",
                lambda m: m["inputs"].__setitem__(
                    "paddingSemantics", "masked-softmax"
                ),
                "input contract",
            ),
            (
                "tensor payload",
                lambda m: m["artifact"]["tensors"]["layer0_output"].__setitem__(
                    "sha256", "0" * 64
                ),
                "tensor inventory/payload",
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
                shutil.copy2(
                    FIXTURE / original["artifact"]["file"],
                    target / original["artifact"]["file"],
                )
                failed = self.run_verify(target)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(error, failed.stderr + failed.stdout)


if __name__ == "__main__":
    unittest.main()
