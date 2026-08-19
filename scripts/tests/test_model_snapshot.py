import json
import tempfile
import unittest
from pathlib import Path

from scripts.release.ensure_model_snapshot import (
    _managed_staging_paths,
    ensure_snapshot,
    hf_cache_location,
)
from scripts.release.verify_model_snapshot import (
    MARKER,
    MATERIALIZATION_INCOMPLETE,
    MATERIALIZATION_RECEIPT,
    completed_materialization_receipt,
    snapshot_inventory,
    snapshot_path,
    verify_materialization_provenance,
    verify_snapshot,
)


MODEL = {
    "key": "test-model",
    "repository": "example/test-model",
    "revision": "a" * 40,
    "expected_files": ["config.json", "weights/model.safetensors"],
}

MIRRORED_MODEL = {
    **MODEL,
    "materialization_repository": "public/test-model-mirror",
    "materialization_revision": "b" * 40,
    "materialization_expected_files": ["config.json", "weights/model.safetensors"],
}


class ModelSnapshotTests(unittest.TestCase):
    def make_snapshot(self, root: Path, name: str) -> Path:
        snapshot = root / name
        (snapshot / "weights").mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "weights/model.safetensors").write_bytes(b"fixture")
        return snapshot

    def complete_download(self, kwargs: dict) -> str:
        staging = Path(kwargs["local_dir"])
        self.make_snapshot(staging.parent, staging.name)
        return str(staging)

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

    def test_inventory_excludes_provenance_markers_and_local_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), MODEL["revision"])
            before = snapshot_inventory(MODEL, snapshot)
            (snapshot / MARKER).write_text(MODEL["revision"], encoding="utf-8")
            (snapshot / MATERIALIZATION_RECEIPT).write_text(
                json.dumps(completed_materialization_receipt(MIRRORED_MODEL, snapshot)),
                encoding="utf-8",
            )
            cache = snapshot / ".cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "download.json").write_text("transient", encoding="utf-8")
            after = snapshot_inventory(MIRRORED_MODEL, snapshot)
            self.assertEqual(before["inventory_sha256"], after["inventory_sha256"])

    def test_materialization_fields_are_paired_and_prefix_is_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_snapshot(Path(temporary), MODEL["revision"])
            for model, message in (
                ({**MODEL, "materialization_repository": "mirror/repo"}, "together"),
                ({**MIRRORED_MODEL, "materialization_revision": "not-a-sha"}, "40-hex"),
                ({**MIRRORED_MODEL, "materialization_path_prefix": "../bf16"}, "normalized"),
                ({**MIRRORED_MODEL, "materialization_requires_auth": "false"}, "boolean"),
                ({**MIRRORED_MODEL, "download_files": ["../weights"]}, "relative patterns"),
                (
                    {**MIRRORED_MODEL, "materialization_expected_files": ["weights/*"]},
                    "sorted, unique",
                ),
                ({**MODEL, "materialization_path_prefix": "bf16"}, "without a source"),
                ({**MODEL, "materialization_requires_auth": True}, "without a source"),
                (
                    {**MODEL, "materialization_expected_files": ["config.json"]},
                    "without a source",
                ),
            ):
                with self.subTest(model=model), self.assertRaisesRegex(RuntimeError, message):
                    verify_snapshot(model, snapshot)

    def test_materialization_receipt_is_exact_and_inventory_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "materialized"
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

            before = snapshot_inventory(
                MODEL,
                self.make_snapshot(root, MODEL["revision"]),
            )["inventory_sha256"]
            self.assertTrue(ensure_snapshot(MIRRORED_MODEL, snapshot, download))
            verify_materialization_provenance(MIRRORED_MODEL, snapshot, required=True)
            self.assertEqual(
                snapshot_inventory(MIRRORED_MODEL, snapshot)["inventory_sha256"], before
            )
            self.assertEqual(calls[0]["repo_id"], MIRRORED_MODEL["materialization_repository"])
            self.assertEqual(calls[0]["revision"], MIRRORED_MODEL["materialization_revision"])
            self.assertIs(calls[0]["token"], False)

            receipt = snapshot / MATERIALIZATION_RECEIPT
            document = json.loads(receipt.read_text(encoding="utf-8"))
            document["materialization_revision"] = "c" * 40
            receipt.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "receipt does not match"):
                verify_snapshot(MIRRORED_MODEL, snapshot)

    def test_required_provenance_accepts_only_receipt_or_pristine_canonical_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "models--example--test-model" / "snapshots"
            canonical = self.make_snapshot(cache, MODEL["revision"])
            verify_materialization_provenance(MIRRORED_MODEL, canonical, required=True)
            (canonical / MARKER).write_text(MODEL["revision"], encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "requires either"):
                verify_materialization_provenance(MIRRORED_MODEL, canonical, required=True)

            untracked = self.make_snapshot(root, MODEL["revision"])
            with self.assertRaisesRegex(RuntimeError, "requires either"):
                verify_materialization_provenance(MIRRORED_MODEL, untracked, required=True)

    def test_strict_provenance_self_heals_a_legacy_marker_only_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self.make_snapshot(root, "legacy")
            (snapshot / MARKER).write_text(MODEL["revision"] + "\n", encoding="utf-8")
            calls = []

            # Non-strict Windows consumers retain their already-provisioned canonical cache.
            self.assertFalse(
                ensure_snapshot(MIRRORED_MODEL, snapshot, lambda **kwargs: calls.append(kwargs))
            )
            self.assertEqual(calls, [])

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

            self.assertTrue(
                ensure_snapshot(
                    MIRRORED_MODEL,
                    snapshot,
                    download,
                    require_materialization_provenance=True,
                )
            )
            self.assertEqual(len(calls), 1)
            verify_materialization_provenance(MIRRORED_MODEL, snapshot, required=True)

    def test_hub_failure_cannot_re_receipt_an_existing_persistent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self.make_snapshot(root, "legacy")
            (snapshot / MARKER).write_text(MODEL["revision"] + "\n", encoding="utf-8")

            def inaccessible_hub(**kwargs) -> None:
                staging = Path(kwargs["local_dir"])
                self.assertEqual(list(staging.iterdir()), [])
                raise RuntimeError("mirror repo-info unavailable")

            with self.assertRaisesRegex(RuntimeError, "mirror repo-info unavailable"):
                ensure_snapshot(
                    MIRRORED_MODEL,
                    snapshot,
                    inaccessible_hub,
                    require_materialization_provenance=True,
                )
            self.assertFalse((snapshot / MATERIALIZATION_RECEIPT).exists())
            self.assertTrue((snapshot / MATERIALIZATION_INCOMPLETE).is_file())
            with self.assertRaisesRegex(RuntimeError, "incomplete model materialization"):
                verify_snapshot(MIRRORED_MODEL, snapshot)

    def test_invalid_staged_success_never_destroys_a_legacy_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            for case in ("incomplete", "symlink", "underselected"):
                with self.subTest(case=case):
                    model = (
                        {
                            **MIRRORED_MODEL,
                            "materialization_expected_files": [
                                "config.json",
                                "selected-metadata.json",
                                "weights/model.safetensors",
                            ],
                        }
                        if case == "underselected"
                        else MIRRORED_MODEL
                    )
                    snapshot = self.make_snapshot(root, f"legacy-{case}")
                    (snapshot / MARKER).write_text(
                        MODEL["revision"] + "\n", encoding="utf-8"
                    )
                    before_config = (snapshot / "config.json").read_bytes()
                    before_weights = (snapshot / "weights/model.safetensors").read_bytes()
                    outside = root / f"outside-{case}.safetensors"
                    outside.write_bytes(b"untrusted")

                    def invalid_download(**kwargs) -> str:
                        staging = Path(kwargs["local_dir"])
                        (staging / "config.json").write_text(
                            '{"replacement":true}', encoding="utf-8"
                        )
                        if case == "symlink":
                            (staging / "weights").mkdir()
                            (staging / "weights/model.safetensors").symlink_to(outside)
                        elif case == "underselected":
                            (staging / "weights").mkdir()
                            (staging / "weights/model.safetensors").write_bytes(b"replacement")
                        return str(staging)

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "snapshot is incomplete|materialization source tree contains a symlink|"
                        "exact projected file set",
                    ):
                        ensure_snapshot(
                            model,
                            snapshot,
                            invalid_download,
                            require_materialization_provenance=True,
                        )
                    self.assertEqual((snapshot / "config.json").read_bytes(), before_config)
                    self.assertEqual(
                        (snapshot / "weights/model.safetensors").read_bytes(), before_weights
                    )
                    self.assertFalse((snapshot / MATERIALIZATION_RECEIPT).exists())
                    self.assertTrue((snapshot / MATERIALIZATION_INCOMPLETE).is_file())

    def test_retry_reclaims_only_its_deterministic_hard_interruption_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            staging, claim, document = _managed_staging_paths(snapshot)
            staging.mkdir()
            (staging / "partial.safetensors").write_bytes(b"orphan")
            claim.write_text(json.dumps(document), encoding="utf-8")
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                actual_staging = Path(kwargs["local_dir"])
                self.assertEqual(actual_staging, staging)
                self.assertEqual(list(actual_staging.iterdir()), [])
                return self.complete_download(kwargs)

            self.assertTrue(ensure_snapshot(MIRRORED_MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertFalse(staging.exists())
            self.assertFalse(claim.exists())
            verify_materialization_provenance(MIRRORED_MODEL, snapshot, required=True)

    def test_receipt_inventory_rejects_tampered_symlinks_and_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "materialized"
            self.assertTrue(
                ensure_snapshot(
                    MIRRORED_MODEL,
                    snapshot,
                    lambda **kwargs: self.complete_download(kwargs),
                )
            )
            outside = root / "outside.safetensors"
            outside.write_bytes(b"external")
            weights = snapshot / "weights/model.safetensors"
            weights.unlink()
            weights.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
                verify_snapshot(MIRRORED_MODEL, snapshot)

            weights.unlink()
            weights.write_bytes(b"tamper!")
            with self.assertRaisesRegex(RuntimeError, "receipt inventory"):
                verify_snapshot(MIRRORED_MODEL, snapshot)

            weights.write_bytes(b"fixture")
            (snapshot / "stale.safetensors").write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "exact projected file set"):
                verify_snapshot(MIRRORED_MODEL, snapshot)

    def test_materializer_never_follows_snapshot_or_provenance_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            target = outside / "target"
            target.write_text("do not overwrite", encoding="utf-8")

            for unsafe_name in (MARKER, MATERIALIZATION_RECEIPT):
                with self.subTest(unsafe_name=unsafe_name):
                    snapshot = self.make_snapshot(root, f"unsafe-{unsafe_name}")
                    if unsafe_name == MATERIALIZATION_RECEIPT:
                        (snapshot / MARKER).write_text(
                            MODEL["revision"] + "\n", encoding="utf-8"
                        )
                    (snapshot / unsafe_name).symlink_to(target)
                    with self.assertRaisesRegex(RuntimeError, "unsafe"):
                        ensure_snapshot(
                            MIRRORED_MODEL,
                            snapshot,
                            lambda **kwargs: self.complete_download(kwargs),
                            require_materialization_provenance=True,
                        )
                    self.assertEqual(target.read_text(encoding="utf-8"), "do not overwrite")

            snapshot_link = root / "snapshot-link"
            snapshot_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                ensure_snapshot(
                    MIRRORED_MODEL,
                    snapshot_link,
                    lambda **kwargs: self.complete_download(kwargs),
                    require_materialization_provenance=True,
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "do not overwrite")

            receipt_backed = root / "receipt-backed"
            self.assertTrue(
                ensure_snapshot(
                    MIRRORED_MODEL,
                    receipt_backed,
                    lambda **kwargs: self.complete_download(kwargs),
                )
            )
            receipt_alias = root / "receipt-alias"
            receipt_alias.symlink_to(receipt_backed, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "unsafe snapshot directory symlink"):
                verify_snapshot(
                    MIRRORED_MODEL,
                    receipt_alias,
                    require_materialization_provenance=True,
                )
            download_calls = []
            with self.assertRaisesRegex(RuntimeError, "unsafe snapshot directory symlink"):
                ensure_snapshot(
                    MIRRORED_MODEL,
                    receipt_alias,
                    lambda **kwargs: download_calls.append(kwargs),
                    require_materialization_provenance=True,
                )
            self.assertEqual(download_calls, [])

            canonical_parent = root / "models--example--test-model" / "snapshots"
            canonical = self.make_snapshot(canonical_parent, MODEL["revision"])
            canonical_alias = root / "canonical-alias"
            canonical_alias.symlink_to(canonical, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "unsafe snapshot directory symlink"):
                verify_snapshot(
                    MIRRORED_MODEL,
                    canonical_alias,
                    require_materialization_provenance=True,
                )

    def test_materialization_prefix_is_downloaded_then_projected_to_canonical_layout(self) -> None:
        model = {**MIRRORED_MODEL, "materialization_path_prefix": "bf16"}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                staging = Path(kwargs["local_dir"])
                mirrored = staging / "bf16"
                (mirrored / "weights").mkdir(parents=True)
                (mirrored / "config.json").write_text("{}", encoding="utf-8")
                (mirrored / "weights/model.safetensors").write_bytes(b"fixture")
                return str(staging)

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertEqual(calls[0]["allow_patterns"], ["bf16/**"])
            self.assertFalse((snapshot / "bf16").exists())
            self.assertTrue((snapshot / "config.json").is_file())
            self.assertTrue((snapshot / "weights/model.safetensors").is_file())
            verify_materialization_provenance(model, snapshot, required=True)

    def test_materialization_prefix_scopes_an_existing_download_allow_list(self) -> None:
        model = {
            **MIRRORED_MODEL,
            "materialization_path_prefix": "dense/bf16",
            "download_files": ["config.json", "weights/model.safetensors"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                staging = Path(kwargs["local_dir"])
                mirrored = staging / "dense/bf16"
                (mirrored / "weights").mkdir(parents=True)
                (mirrored / "config.json").write_text("{}", encoding="utf-8")
                (mirrored / "weights/model.safetensors").write_bytes(b"fixture")
                return str(staging)

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertEqual(
                calls[0]["allow_patterns"],
                ["dense/bf16/config.json", "dense/bf16/weights/model.safetensors"],
            )
            self.assertFalse((snapshot / "dense").exists())

    def test_interrupted_alternate_download_never_publishes_a_completion_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "materialized"
            model = {
                **MIRRORED_MODEL,
                "download_files": [
                    "config.json",
                    "weights/model.safetensors",
                    "selected-metadata.json",
                ],
                "materialization_expected_files": [
                    "config.json",
                    "selected-metadata.json",
                    "weights/model.safetensors",
                ],
            }

            def interrupt(**kwargs) -> None:
                staging = Path(kwargs["local_dir"])
                self.make_snapshot(staging.parent, staging.name)
                raise RuntimeError("transfer interrupted")

            with self.assertRaisesRegex(RuntimeError, "transfer interrupted"):
                ensure_snapshot(model, snapshot, interrupt)
            self.assertTrue((snapshot / MARKER).is_file())
            self.assertFalse((snapshot / MATERIALIZATION_RECEIPT).exists())
            self.assertTrue((snapshot / MATERIALIZATION_INCOMPLETE).is_file())
            with self.assertRaisesRegex(RuntimeError, "incomplete model materialization"):
                verify_snapshot(model, snapshot)

            def repair(**kwargs) -> str:
                staging = Path(kwargs["local_dir"])
                self.make_snapshot(staging.parent, staging.name)
                (staging / "selected-metadata.json").write_text("{}", encoding="utf-8")
                return str(staging)

            self.assertTrue(ensure_snapshot(model, snapshot, repair))
            verify_snapshot(model, snapshot)
            verify_materialization_provenance(model, snapshot, required=True)
            self.assertFalse((snapshot / MATERIALIZATION_INCOMPLETE).exists())

    def test_materialization_replaces_stale_destination_symlinks_and_extra_payload(self) -> None:
        model = {**MIRRORED_MODEL, "materialization_path_prefix": "bf16"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = (
                root
                / "models--example--test-model"
                / "snapshots"
                / MODEL["revision"]
            )
            outside = root / "outside"
            outside.mkdir()
            (snapshot / "q4").mkdir(parents=True)
            (snapshot / "q4/stale.safetensors").write_bytes(b"stale")

            def download(**kwargs) -> str:
                staging = Path(kwargs["local_dir"])
                source = staging / "bf16/weights"
                source.mkdir(parents=True)
                (staging / "bf16/config.json").write_text("{}", encoding="utf-8")
                (source / "model.safetensors").write_bytes(b"fixture")
                (snapshot / "weights").symlink_to(outside, target_is_directory=True)
                return str(staging)

            self.assertTrue(
                ensure_snapshot(
                    model,
                    snapshot,
                    download,
                    require_materialization_provenance=True,
                )
            )
            self.assertEqual(list(outside.iterdir()), [])
            self.assertFalse((snapshot / "q4").exists())
            self.assertFalse((snapshot / "weights").is_symlink())
            self.assertTrue((snapshot / "weights/model.safetensors").is_file())

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

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(
                (snapshot / MARKER).read_text(encoding="utf-8"),
                MODEL["revision"] + "\n",
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repo_id"], MODEL["repository"])
            self.assertEqual(calls[0]["revision"], MODEL["revision"])
            self.assertIs(calls[0]["token"], False)
            self.assertNotEqual(Path(calls[0]["local_dir"]), snapshot)

    def test_ensure_limits_download_to_download_files_allow_list(self) -> None:
        model = {**MODEL, "download_files": ["config.json", "weights/model.safetensors"]}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

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

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

            self.assertTrue(ensure_snapshot(model, snapshot, download))
            self.assertIs(calls[0]["token"], True)

    def test_ensure_omits_allow_patterns_without_download_files(self) -> None:
        # The default (no `download_files`) must NOT pass `allow_patterns` — the whole repo is fetched.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialized"
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                return self.complete_download(kwargs)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertNotIn("allow_patterns", calls[0])

    def test_ensure_repairs_an_incomplete_matching_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL["revision"]
            snapshot.mkdir()

            def download(**kwargs) -> str:
                return self.complete_download(kwargs)

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

    def test_cache_heal_retires_a_stale_incomplete_marker_and_then_stays_cached(self) -> None:
        """The state an interrupted main-era staging run leaves in a canonical cache directory.

        `verify_materialization_provenance` refuses on `MATERIALIZATION_INCOMPLETE`
        unconditionally, so a marker the cache heal did not remove survived its own fetch: every
        run re-materialized the whole repo and then failed the re-verify, reporting a
        provenance refusal through a message about `expected_files`. The second `ensure_snapshot`
        below is the half that would have kept failing forever.
        """
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            self.fill(snapshot)
            (snapshot / MATERIALIZATION_INCOMPLETE).write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "canonical_repository": MODEL["repository"],
                        "canonical_revision": MODEL["revision"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.fill(snapshot)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertFalse((snapshot / MATERIALIZATION_INCOMPLETE).exists())
            verify_snapshot(MODEL, snapshot)
            # The recovery is the point: the next run costs nothing at all.
            self.assertFalse(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)

    def test_cache_heal_retires_a_receipt_a_retired_alternate_source_left(self) -> None:
        """The same shape one row-edit away: a model that used to declare an alternate source.

        Its receipt stays in the cache directory, `verify_materialization_provenance` calls an
        unexpected receipt a failure for a model that declares no source, and the plain-directory
        path already unlinks one before its own download. Only the cache heal did not.
        """
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            self.fill(snapshot)
            (snapshot / MATERIALIZATION_RECEIPT).write_text(
                json.dumps(completed_materialization_receipt(MIRRORED_MODEL, snapshot)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected alternate-source"):
                verify_snapshot(MODEL, snapshot)

            self.assertTrue(ensure_snapshot(MODEL, snapshot, lambda **kwargs: None))
            self.assertFalse((snapshot / MATERIALIZATION_RECEIPT).exists())
            verify_snapshot(MODEL, snapshot)

    def test_cache_heal_refuses_an_unsafe_incomplete_marker_instead_of_unlinking_it(self) -> None:
        """Retiring provenance is a targeted unlink of a regular file, never an rm of whatever is
        at that name. A directory or a symlink there is not a marker this run wrote."""
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            self.fill(snapshot)
            (snapshot / MATERIALIZATION_INCOMPLETE).mkdir()
            with self.assertRaisesRegex(RuntimeError, "unsafe .* entry"):
                ensure_snapshot(MODEL, snapshot, lambda **kwargs: None)
            self.assertTrue((snapshot / MATERIALIZATION_INCOMPLETE).is_dir())

    def test_cache_heal_threads_the_strict_provenance_flag_into_its_re_verify(self) -> None:
        """The heal's own `verify_snapshot` must carry the flag, not silently drop it.

        Two-sided, because a heal that ignored the flag would also pass the pristine half. The
        strict gate accepts a canonical `snapshots/<revision>` directory only while it carries no
        `.sceneworks-model-revision` marker — a marker means this script wrote the directory, and
        then it owes a receipt. No shipped lane pairs the flag with a canonical model (all six
        strict invocations name alternate-source rows), so this pins the plumbing.
        """
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            calls = []

            def download(**kwargs) -> None:
                calls.append(kwargs)
                self.fill(snapshot)

            self.assertTrue(
                ensure_snapshot(
                    MODEL, snapshot, download, require_materialization_provenance=True
                )
            )
            self.assertEqual(len(calls), 1)

            (snapshot / MARKER).write_text(MODEL["revision"] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "requires either"):
                ensure_snapshot(
                    MODEL, snapshot, download, require_materialization_provenance=True
                )
            # The same directory, the same heal, without the flag: accepted.
            self.assertFalse(ensure_snapshot(MODEL, snapshot, download))

    def test_a_receipt_model_at_a_cache_path_stages_under_strict_provenance(self) -> None:
        """The combined path: alternate source, cache-shaped destination, strict flag, and a stale
        incomplete marker on top.

        An alternate-source model is heal-ELIGIBLE by location and still must not take the cache
        heal — its bytes come from a different repository and revision than the directory names, so
        a `cache_dir` fetch would land them in a sibling. It takes staging instead, which is what
        publishes the receipt the strict gate then demands.
        """
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.make_cache_snapshot(Path(temporary))
            (snapshot / MATERIALIZATION_INCOMPLETE).write_text("{}\n", encoding="utf-8")
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                staging = Path(kwargs["local_dir"])
                self.fill(staging)
                return str(staging)

            self.assertTrue(
                ensure_snapshot(
                    MIRRORED_MODEL,
                    snapshot,
                    download,
                    require_materialization_provenance=True,
                )
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repo_id"], MIRRORED_MODEL["materialization_repository"])
            self.assertEqual(calls[0]["revision"], MIRRORED_MODEL["materialization_revision"])
            self.assertNotIn("cache_dir", calls[0])
            self.assertFalse((snapshot / MATERIALIZATION_INCOMPLETE).exists())
            self.assertTrue((snapshot / MATERIALIZATION_RECEIPT).is_file())
            verify_snapshot(MIRRORED_MODEL, snapshot, require_materialization_provenance=True)
            # And the receipt is what carries it: strip it and the strict gate refuses again,
            # because a directory this script wrote is not a pristine canonical cache.
            (snapshot / MATERIALIZATION_RECEIPT).unlink()
            with self.assertRaisesRegex(RuntimeError, "requires either"):
                verify_snapshot(
                    MIRRORED_MODEL, snapshot, require_materialization_provenance=True
                )

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
        # The control: an identical shortfall OUTSIDE any hub layout keeps the `local_dir` path
        # (never `cache_dir`), the revision marker and the post-download verification. Since the
        # materialization-receipt flow landed, `local_dir` is the managed staging sibling rather
        # than the snapshot itself; the verified tree is projected into the snapshot afterwards.
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "materialize" / MODEL["revision"]
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            calls = []

            def download(**kwargs) -> str:
                calls.append(kwargs)
                self.fill(Path(kwargs["local_dir"]))
                return kwargs["local_dir"]

            self.assertTrue(ensure_snapshot(MODEL, snapshot, download))
            self.assertEqual(len(calls), 1)
            self.assertNotIn("cache_dir", calls[0])
            staging, _claim, _document = _managed_staging_paths(snapshot)
            self.assertEqual(calls[0]["local_dir"], str(staging))
            self.assertEqual(
                (snapshot / MARKER).read_text(encoding="utf-8"), MODEL["revision"] + "\n"
            )
            verify_snapshot(MODEL, snapshot)


if __name__ == "__main__":
    unittest.main()
