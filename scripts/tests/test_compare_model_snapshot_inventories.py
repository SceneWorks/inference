import json
import tempfile
import unittest
from pathlib import Path

from scripts.release.compare_model_snapshot_inventories import compare


MODEL = "starvector-1b-im2svg"


def inventory() -> dict:
    return {
        "schema_version": 1,
        "model": MODEL,
        "repository": "starvector/starvector-1b-im2svg",
        "revision": "a" * 40,
        "inventory_sha256": "b" * 64,
        "files": [{"path": "model.safetensors", "kind": "file", "size": 1, "sha256": "c" * 64}],
    }


class CompareModelSnapshotInventoriesTests(unittest.TestCase):
    def write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_accepts_equal_content_identity_when_file_kind_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_value = inventory()
            actual_value = inventory()
            actual_value["files"][0]["kind"] = "symlink"
            expected = self.write(root, "expected.json", expected_value)
            actual = self.write(root, "actual.json", actual_value)
            self.assertEqual(compare(expected, actual, MODEL), "b" * 64)

    def test_rejects_digest_or_identity_drift(self) -> None:
        mutations = {
            "digest": lambda value: value.update(inventory_sha256="d" * 64),
            "repository": lambda value: value.update(repository="other/repository"),
            "revision": lambda value: value.update(revision="e" * 40),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = self.write(root, "expected.json", inventory())
                actual_value = inventory()
                mutate(actual_value)
                actual = self.write(root, "actual.json", actual_value)
                with self.assertRaisesRegex(RuntimeError, "differs across native hosts"):
                    compare(expected, actual, MODEL)

    def test_rejects_symlink_or_unexpected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self.write(root, "target.json", inventory())
            linked = root / "linked.json"
            linked.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "non-empty regular file"):
                compare(target, linked, MODEL)
            malformed = inventory()
            malformed["extra"] = True
            actual = self.write(root, "actual.json", malformed)
            with self.assertRaisesRegex(RuntimeError, "unexpected schema"):
                compare(target, actual, MODEL)


if __name__ == "__main__":
    unittest.main()
