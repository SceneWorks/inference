import tempfile
import unittest
from pathlib import Path

from scripts.release.ensure_model_snapshot_file import (
    canonical_snapshot_path,
    ensure_expected_file,
)


REVISION = "a" * 40
MODEL = {
    "key": "test-q4",
    "repository": "SceneWorks/test-model",
    "revision": REVISION,
    "requires_auth": False,
    "download_files": ["q4/*", "LICENSE.pdf"],
    "expected_files": ["LICENSE.pdf", "q4/config.json"],
}


class EnsureModelSnapshotFileTests(unittest.TestCase):
    def make_snapshot(self, root: Path) -> Path:
        snapshot = canonical_snapshot_path(root, MODEL)
        (snapshot / "q4").mkdir(parents=True)
        (snapshot / "q4/config.json").write_text("{}", encoding="utf-8")
        return snapshot

    def test_repairs_only_the_manifest_required_file_at_the_exact_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshot = self.make_snapshot(cache_root)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                target = snapshot / kwargs["filename"]
                target.write_bytes(b"pinned licence")
                return str(target)

            target = ensure_expected_file(MODEL, cache_root, "LICENSE.pdf", download)
            self.assertEqual(target.read_bytes(), b"pinned licence")
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": MODEL["repository"],
                        "filename": "LICENSE.pdf",
                        "revision": REVISION,
                        "cache_dir": str(cache_root),
                        "token": False,
                        "force_download": True,
                    }
                ],
            )

    def test_rejects_unlisted_and_disallowed_files_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            self.make_snapshot(cache_root)
            calls = []
            for name in ("README.md", "../LICENSE.pdf"):
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    ensure_expected_file(
                        MODEL,
                        cache_root,
                        name,
                        lambda **kwargs: calls.append(kwargs),
                    )
            self.assertEqual(calls, [])

    def test_rejects_mutable_revision_and_noncanonical_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            calls = []
            for mutation, message in (
                ({"revision": "main"}, "immutable 40-hex"),
                ({"repository": "SceneWorks/test/model"}, "canonical owner/name"),
            ):
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    RuntimeError, message
                ):
                    ensure_expected_file(
                        {**MODEL, **mutation},
                        cache_root,
                        "LICENSE.pdf",
                        lambda **kwargs: calls.append(kwargs),
                    )
            self.assertEqual(calls, [])

    def test_refuses_to_repair_when_another_required_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshot = self.make_snapshot(cache_root)
            (snapshot / "q4/config.json").unlink()
            calls = []
            with self.assertRaisesRegex(RuntimeError, "q4/config.json"):
                ensure_expected_file(
                    MODEL,
                    cache_root,
                    "LICENSE.pdf",
                    lambda **kwargs: calls.append(kwargs),
                )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
