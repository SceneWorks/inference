import json
import tempfile
import unittest
from pathlib import Path

from scripts.release.ensure_model_snapshot import ensure_snapshot
from scripts.release.verify_model_snapshot import MARKER, snapshot_inventory, verify_snapshot


MODEL = {
    "key": "test-model",
    "repository": "example/test-model",
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

    def test_inventory_binds_every_dereferenced_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), MODEL["revision"])
            first = snapshot_inventory(MODEL, snapshot)
            self.assertEqual(first["revision"], MODEL["revision"])
            self.assertEqual(
                {item["path"] for item in first["files"]},
                {"config.json", "weights/model.safetensors"},
            )
            (snapshot / "weights/model.safetensors").write_bytes(b"mutated")
            second = snapshot_inventory(MODEL, snapshot)
            self.assertNotEqual(first["inventory_sha256"], second["inventory_sha256"])

    def test_inventory_hashes_symlink_target_content_and_rejects_broken_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self.make_snapshot(root, MODEL["revision"])
            target = root / "blob"
            target.write_bytes(b"one")
            linked = snapshot / "extra.safetensors"
            linked.write_bytes(b"one")
            file_backed = snapshot_inventory(MODEL, snapshot)
            linked.unlink()
            linked.symlink_to(target)
            first = snapshot_inventory(MODEL, snapshot)
            self.assertEqual(file_backed["inventory_sha256"], first["inventory_sha256"])
            target.write_bytes(b"two")
            second = snapshot_inventory(MODEL, snapshot)
            self.assertNotEqual(first["inventory_sha256"], second["inventory_sha256"])
            target.unlink()
            with self.assertRaisesRegex(RuntimeError, "cannot inventory model file"):
                snapshot_inventory(MODEL, snapshot)

    def test_inventory_excludes_revision_marker_and_local_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), MODEL["revision"])
            before = snapshot_inventory(MODEL, snapshot)
            (snapshot / MARKER).write_text(MODEL["revision"], encoding="utf-8")
            cache = snapshot / ".cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "download.json").write_text("transient", encoding="utf-8")
            after = snapshot_inventory(MODEL, snapshot)
            self.assertEqual(before["inventory_sha256"], after["inventory_sha256"])

    def test_rejects_revision_drift_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), "wrong")
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                verify_snapshot(MODEL, snapshot)
            (snapshot / MARKER).write_text(MODEL["revision"], encoding="utf-8")
            (snapshot / "config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing: config.json"):
                verify_snapshot(MODEL, snapshot)

    def test_weight_index_requires_every_referenced_shard(self) -> None:
        indexed = {**MODEL, "expected_files": ["weights/model.safetensors.index.json"]}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            (snapshot / "weights").mkdir(parents=True)
            (snapshot / "weights/model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "references missing shards"):
                verify_snapshot(indexed, snapshot)
            (snapshot / "weights/model-00001-of-00001.safetensors").write_bytes(b"fixture")
            verify_snapshot(indexed, snapshot)

    def test_rejects_invalid_weight_index(self) -> None:
        indexed = {**MODEL, "expected_files": ["weights/model.safetensors.index.json"]}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            (snapshot / "weights").mkdir(parents=True)
            (snapshot / "weights/model.safetensors.index.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid weight index"):
                verify_snapshot(indexed, snapshot)

    def test_ensure_reuses_a_valid_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            snapshot = self.make_snapshot(Path(temporary), MODEL["revision"])
            self.assertFalse(
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            )
            self.assertEqual(calls, [])

    def test_ensure_materializes_and_marks_a_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.make_snapshot(Path(temporary), "materialized")

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(
                (snapshot / MARKER).read_text(encoding="utf-8"),
                MODEL["revision"] + "\n",
            )
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": MODEL["repository"],
                        "revision": MODEL["revision"],
                        "local_dir": str(snapshot),
                        "token": False,
                    }
                ],
            )

    def test_ensure_limits_download_to_download_files_allow_list(self) -> None:
        model = {**MODEL, "download_files": ["config.json", "weights/model.safetensors"]}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.make_snapshot(Path(temporary), "materialized")

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0]["allow_patterns"],
                ["config.json", "weights/model.safetensors"],
            )
            # The base kwargs are unchanged by the allow-list.
            self.assertEqual(calls[0]["repo_id"], model["repository"])
            self.assertEqual(calls[0]["revision"], model["revision"])
            self.assertIs(calls[0]["token"], False)

    def test_ensure_uses_configured_credential_for_gated_model(self) -> None:
        model = {**MODEL, "requires_auth": True}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.make_snapshot(Path(temporary), "materialized")

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertIs(calls[0]["token"], True)

    def test_ensure_omits_allow_patterns_without_download_files(self) -> None:
        # The default (no `download_files`) must NOT pass `allow_patterns` — the whole repo is fetched.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.make_snapshot(Path(temporary), "materialized")

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertNotIn("allow_patterns", calls[0])

    def test_ensure_repairs_an_incomplete_matching_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            snapshot.mkdir()

            def download(**kwargs) -> None:
                self.make_snapshot(Path(temporary), MODEL["revision"])

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            verify_snapshot(MODEL, snapshot)

    def test_ensure_does_not_overwrite_revision_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            snapshot = self.make_snapshot(Path(temporary), "wrong")
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
