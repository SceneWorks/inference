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
DONOR_MODEL = {
    **MODEL,
    "key": "test-wan-donor",
    "download_files": ["q8/vae.safetensors"],
    "expected_files": ["q8/vae.safetensors"],
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

    def test_materializes_an_absent_standalone_exact_file_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshot = canonical_snapshot_path(cache_root, DONOR_MODEL)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                target = snapshot / kwargs["filename"]
                target.parent.mkdir(parents=True)
                target.write_bytes(b"wan z16 donor")
                return str(target)

            target = ensure_expected_file(
                DONOR_MODEL,
                cache_root,
                "q8/vae.safetensors",
                download,
            )
            self.assertEqual(target.read_bytes(), b"wan z16 donor")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repo_id"], DONOR_MODEL["repository"])
            self.assertEqual(calls[0]["filename"], "q8/vae.safetensors")
            self.assertEqual(calls[0]["revision"], REVISION)
            self.assertEqual(calls[0]["cache_dir"], str(cache_root))
            self.assertTrue(calls[0]["force_download"])

    def test_standalone_projection_runs_full_verification_after_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshot = canonical_snapshot_path(cache_root, DONOR_MODEL)

            def incomplete_download(**kwargs):
                target = snapshot / kwargs["filename"]
                target.parent.mkdir(parents=True)
                return str(target)

            with self.assertRaisesRegex(RuntimeError, "q8/vae.safetensors"):
                ensure_expected_file(
                    DONOR_MODEL,
                    cache_root,
                    "q8/vae.safetensors",
                    incomplete_download,
                )

    def test_absent_projection_with_other_required_files_stays_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            with self.assertRaisesRegex(RuntimeError, "snapshot directory does not exist"):
                ensure_expected_file(
                    MODEL,
                    Path(temporary),
                    "LICENSE.pdf",
                    lambda **kwargs: calls.append(kwargs),
                )
            self.assertEqual(calls, [])

    def test_absent_standalone_projection_refuses_a_snapshot_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshot = canonical_snapshot_path(cache_root, DONOR_MODEL)
            snapshot.parent.mkdir(parents=True)
            snapshot.symlink_to(cache_root / "missing-target", target_is_directory=True)
            calls = []
            with self.assertRaisesRegex(RuntimeError, "unsafe snapshot directory symlink"):
                ensure_expected_file(
                    DONOR_MODEL,
                    cache_root,
                    "q8/vae.safetensors",
                    lambda **kwargs: calls.append(kwargs),
                )
            self.assertEqual(calls, [])

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
