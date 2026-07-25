import json
import tempfile
import unittest
from pathlib import Path

from scripts.reference.sa3_reference import (
    COMMON_FILES,
    SNAPSHOTS,
    T5_FILES,
    InvalidReference,
    resolve_snapshots,
    verify_artifacts,
)


class StableAudio3ReferenceTests(unittest.TestCase):
    def make_snapshots(self, root: Path) -> dict[str, str]:
        environ = {}
        for spec in SNAPSHOTS:
            snapshot = root / spec.revision
            snapshot.mkdir()
            for name in COMMON_FILES:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            if spec.kind == "dit":
                for name in T5_FILES:
                    path = snapshot / "t5gemma-b-b-ul2" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            environ[spec.env] = str(snapshot)
        return environ

    def test_requires_all_explicit_snapshot_environment_variables(self) -> None:
        with self.assertRaisesRegex(InvalidReference, SNAPSHOTS[0].env):
            resolve_snapshots({})

    def test_accepts_all_pinned_complete_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environ = self.make_snapshots(Path(temporary))
            resolved = resolve_snapshots(environ)
            self.assertEqual(set(resolved), {spec.key for spec in SNAPSHOTS})

    def test_rejects_revision_drift_and_missing_t5_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environ = self.make_snapshots(root)
            first = SNAPSHOTS[0]
            wrong = root / ("a" * 40)
            Path(environ[first.env]).rename(wrong)
            environ[first.env] = str(wrong)
            with self.assertRaisesRegex(InvalidReference, "revision mismatch"):
                resolve_snapshots(environ)

            wrong.rename(root / first.revision)
            environ[first.env] = str(root / first.revision)
            (root / first.revision / "t5gemma-b-b-ul2/tokenizer.json").unlink()
            with self.assertRaisesRegex(InvalidReference, "tokenizer.json"):
                resolve_snapshots(environ)

    def test_artifact_verifier_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            artifact = output / "probe.safetensors"
            artifact.write_bytes(b"probe")
            import hashlib

            digest = hashlib.sha256(b"probe").hexdigest()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "upstream": {
                            "commit": "124e8a799f57a1f665495ecb72e547d0a62867f1"
                        },
                        "artifacts": {
                            "probe": {"file": artifact.name, "sha256": digest}
                        },
                    }
                ),
                encoding="utf-8",
            )
            verify_artifacts(output)
            artifact.write_bytes(b"mutated")
            with self.assertRaisesRegex(InvalidReference, "hash mismatch"):
                verify_artifacts(output)


if __name__ == "__main__":
    unittest.main()
