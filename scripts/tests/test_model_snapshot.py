import json
import tempfile
import unittest
from pathlib import Path

from scripts.release.ensure_model_snapshot import ensure_snapshot, hf_cache_location
from scripts.release.verify_model_snapshot import (
    MARKER,
    snapshot_inventory,
    snapshot_path,
    verify_snapshot,
)


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


class SnapshotPathExpansionTests(unittest.TestCase):
    """`--snapshot` must accept the `~/`-relative form the runner variables are moving to.

    The two macOS boxes do not share a home, so an absolute `/Users/michael/...` variable is only
    correct on whichever one claims a job -- run 32025783719 died on the other Mac trying to
    `mkdir '/Users/michael'`. `resolve_snapshot_paths.py` expands the value inside a job; this is
    the same expansion for every other way these scripts are invoked.
    """

    def test_expands_a_leading_tilde_against_the_current_home(self) -> None:
        self.assertEqual(
            snapshot_path("~/.cache/huggingface/hub"),
            Path.home() / ".cache/huggingface/hub",
        )

    def test_leaves_an_absolute_path_untouched(self) -> None:
        self.assertEqual(snapshot_path("/opt/models/pinned"), Path("/opt/models/pinned"))

    def test_does_not_expand_an_interior_tilde(self) -> None:
        self.assertEqual(snapshot_path("/opt/~/models"), Path("/opt/~/models"))


class CacheResidentSnapshotTests(unittest.TestCase):
    """A cache-resident snapshot is materialized THROUGH the cache, never into it via `local_dir`.

    Run 32025783719's Windows CUDA lane pointed `local_dir` at
    `E:\\huggingface\\hub\\models--MiniMaxAI--MiniMax-H3\\snapshots\\939557dc...`, so
    `huggingface_hub` resolved each blob into that very directory and then copied it onto itself:
    `shutil.SameFileError` on `audio_vae/config.json`. The cache-correct fetch passes `cache_dir`
    (the hub root) and NO `local_dir`, which writes blob, ref and snapshot symlink together and
    lands the result at the same path the lane asked for.
    """

    def cache_snapshot(self, root: Path) -> Path:
        return root / "hub" / "models--example--test-model" / "snapshots" / MODEL["revision"]

    def make_cache_snapshot(self, root: Path) -> Path:
        snapshot = self.cache_snapshot(root)
        snapshot.mkdir(parents=True)
        return snapshot

    def fill(self, snapshot: Path) -> None:
        (snapshot / "weights").mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "weights/model.safetensors").write_bytes(b"fixture")

    def test_splits_the_hub_layout_into_root_repo_and_revision(self) -> None:
        self.assertEqual(
            hf_cache_location(Path("/c/hub/models--org--name/snapshots/deadbeef")),
            (Path("/c/hub"), "models--org--name", "deadbeef"),
        )
        # `HF_HUB_CACHE` need not be called `hub`; the models--*/snapshots pair is the marker.
        self.assertEqual(
            hf_cache_location(Path("/mnt/w/models--org--name/snapshots/deadbeef")),
            (Path("/mnt/w"), "models--org--name", "deadbeef"),
        )
        self.assertIsNone(hf_cache_location(Path("/opt/models/minimax-h3/snapshots/deadbeef")))
        self.assertIsNone(hf_cache_location(Path("/c/hub/models--org--name/blobs/deadbeef")))
        # The cache's own `snapshots/` directory names no revision, so it is not a snapshot.
        self.assertIsNone(hf_cache_location(Path("/c/hub/models--org--name/snapshots")))
        # Anchored at the tail: a path BELOW a snapshot directory is not one.
        self.assertIsNone(
            hf_cache_location(Path("/c/hub/models--org--name/snapshots/deadbeef/vae"))
        )

    def test_complete_cache_resident_snapshot_is_accepted_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            self.fill(snapshot)
            calls = []
            self.assertFalse(
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            )
            self.assertEqual(calls, [])
            # Nothing is written into the cache -- not even the materialization marker.
            self.assertFalse((snapshot / MARKER).exists())

    def test_incomplete_cache_resident_snapshot_heals_through_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.fill(snapshot)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": MODEL["repository"],
                        "revision": MODEL["revision"],
                        "cache_dir": str(Path(temporary) / "hub"),
                        "token": False,
                    }
                ],
            )
            self.assertNotIn("local_dir", calls[0])
            verify_snapshot(MODEL, snapshot)
            # The cache names the revision in the directory itself; nothing is stamped into it.
            self.assertFalse((snapshot / MARKER).exists())

    def test_absent_cache_resident_snapshot_heals_through_the_cache(self) -> None:
        # The fresh-runner state: the revision directory does not exist yet. `local_dir` would
        # still resolve the blob into this path and then copy it onto itself.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.cache_snapshot(Path(temporary))
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                snapshot.mkdir(parents=True)
                self.fill(snapshot)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["cache_dir"], str(Path(temporary) / "hub"))
            self.assertNotIn("local_dir", calls[0])
            verify_snapshot(MODEL, snapshot)

    def test_cache_heal_carries_the_manifest_allow_list_and_credential(self) -> None:
        # The allow-list is scoped by the MODEL, so a cache-resident target must not silently
        # become a whole-repo fetch. Same for the gated-checkpoint credential opt-in.
        model = {
            **MODEL,
            "download_files": ["config.json", "weights/model.safetensors"],
            "requires_auth": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.cache_snapshot(Path(temporary))
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                snapshot.mkdir(parents=True)
                self.fill(snapshot)

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertEqual(calls[0]["allow_patterns"], model["download_files"])
            self.assertIs(calls[0]["token"], True)

    def test_cache_path_naming_the_wrong_repository_refuses_before_fetching(self) -> None:
        # `cache_dir` sends the fetch to `models--example--test-model/` under this same root, so
        # healing would fill a SIBLING of the requested path -- multi-GB, then a failed verify.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = (
                Path(temporary)
                / "hub"
                / "models--other--model"
                / "snapshots"
                / MODEL["revision"]
            )
            snapshot.mkdir(parents=True)
            calls = []
            with self.assertRaises(RuntimeError) as raised:
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            message = str(raised.exception)
            self.assertEqual(calls, [])
            self.assertIn("wrong repository/revision", message)
            self.assertIn(f"models--example--test-model/snapshots/{MODEL['revision']}", message)
            self.assertIn(f"models--other--model/snapshots/{MODEL['revision']}", message)

    def test_absent_cache_path_naming_the_wrong_revision_refuses_before_fetching(self) -> None:
        # The directory does not exist, so `ensure_snapshot`'s marker/name revision check is
        # skipped entirely -- this precondition is the only thing standing in front of the fetch.
        with tempfile.TemporaryDirectory() as temporary:
            drifted = "b" * 40
            snapshot = (
                Path(temporary) / "hub" / "models--example--test-model" / "snapshots" / drifted
            )
            calls = []
            with self.assertRaises(RuntimeError) as raised:
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            message = str(raised.exception)
            self.assertEqual(calls, [])
            self.assertFalse(snapshot.exists())
            self.assertIn("wrong repository/revision", message)
            self.assertIn(f"snapshots/{MODEL['revision']}", message)
            self.assertIn(f"snapshots/{drifted}", message)

    def test_cache_heal_that_does_not_satisfy_the_pin_still_refuses(self) -> None:
        # The backstop: the cache-correct fetch ran and the snapshot is STILL short, so there is
        # no "run it by hand" repair left to suggest.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                # An under-publishing revision: config.json lands, the weights never do.
                (snapshot / "config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(RuntimeError) as raised:
                ensure_snapshot(MODEL, snapshot, download)
            message = str(raised.exception)
            self.assertEqual(len(calls), 1)
            self.assertIn("still does not satisfy the pin after", message)
            self.assertIn("missing: weights/model.safetensors", message)
            self.assertIn(MODEL["revision"], message)

    def test_cache_heal_reports_a_failing_download_against_the_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))

            def download(**kwargs) -> None:
                raise OSError("no route to host")

            with self.assertRaises(RuntimeError) as raised:
                ensure_snapshot(MODEL, snapshot, download)
            message = str(raised.exception)
            self.assertIn("cache-correct fetch into", message)
            self.assertIn(str(Path(temporary) / "hub"), message)
            self.assertIn("no route to host", message)

    def test_plain_materialize_directory_still_uses_local_dir(self) -> None:
        # The control: an identical shortfall OUTSIDE any hub layout keeps the `local_dir` path,
        # the revision marker and the post-download verification.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialize" / MODEL["revision"]
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.fill(snapshot)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["local_dir"], str(snapshot))
            self.assertNotIn("cache_dir", calls[0])
            self.assertEqual(
                (snapshot / MARKER).read_text(encoding="utf-8"), MODEL["revision"] + "\n"
            )
            verify_snapshot(MODEL, snapshot)


if __name__ == "__main__":
    unittest.main()
