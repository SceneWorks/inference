import tempfile
import unittest
from pathlib import Path

from scripts.release.verify_model_snapshot import MARKER, verify_snapshot


MODEL = {
    "key": "test-model",
    "revision": "a" * 40,
    "expected_files": ["config.json", "weights/model.safetensors"],
}


class ModelSnapshotTests(unittest.TestCase):
    def make_snapshot(self, root: Path, name: str) -> Path:
        snapshot = root / name
        (snapshot / "weights").mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "weights/model.safetensors").write_bytes(b"fixture")
        return snapshot

    def test_accepts_standard_hf_revision_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            verify_snapshot(MODEL, self.make_snapshot(Path(temporary), MODEL["revision"]))

    def test_accepts_materialized_snapshot_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), "materialized")
            (snapshot / MARKER).write_text(MODEL["revision"] + "\n", encoding="utf-8")
            verify_snapshot(MODEL, snapshot)

    def test_rejects_revision_drift_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), "wrong")
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                verify_snapshot(MODEL, snapshot)
            (snapshot / MARKER).write_text(MODEL["revision"], encoding="utf-8")
            (snapshot / "config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing: config.json"):
                verify_snapshot(MODEL, snapshot)


if __name__ == "__main__":
    unittest.main()
