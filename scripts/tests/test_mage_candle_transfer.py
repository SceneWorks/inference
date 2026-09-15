from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "verify_mage_candle_transfer.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("verify_mage_candle_transfer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MageCandleTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "oracles"
        self.output.mkdir()
        self.snapshots = []
        for index in range(4):
            snapshot = self.root / f"snapshot-{index}"
            snapshot.mkdir()
            (snapshot / self.module.REVISION_MARKER).write_text(
                str(index + 1) * 40 + "\n", encoding="utf-8"
            )
            self.snapshots.append(snapshot)
        for name in self.module.FILES:
            (self.output / name).write_bytes(name.encode())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_transfer_manifest_round_trip_and_mutations(self) -> None:
        self.module.verify(self.output, *self.snapshots, True)
        self.module.verify(self.output, *self.snapshots, False)

        manifest_path = self.output / self.module.MANIFEST
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        for mutation in ("hash", "population", "revision", "bool-size"):
            document = json.loads(json.dumps(original))
            if mutation == "hash":
                document["files"][0]["sha256"] = "0" * 64
            elif mutation == "population":
                document["files"].pop()
            elif mutation == "revision":
                document["editSnapshotRevision"] = "f" * 40
            else:
                document["files"][0]["bytes"] = True
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaises(
                self.module.InvalidTransfer
            ):
                self.module.verify(self.output, *self.snapshots, False)

    def test_rejects_missing_or_mutated_transferred_file(self) -> None:
        self.module.verify(self.output, *self.snapshots, True)
        (self.output / self.module.FILES[0]).write_bytes(b"mutated")
        with self.assertRaises(self.module.InvalidTransfer):
            self.module.verify(self.output, *self.snapshots, False)

    def test_migrates_only_the_edit_variant_manifest_record(self) -> None:
        self.module.verify(self.output, *self.snapshots, True)
        with self.assertRaises(self.module.InvalidTransfer):
            self.module.migrate_edit_variant_manifest_hash_only(
                self.output, *self.snapshots
            )
        target = self.output / "mage_edit_variants_manifest.json"
        target.write_bytes(b"strictly-migrated-edit-variant-manifest")

        self.module.migrate_edit_variant_manifest_hash_only(
            self.output, *self.snapshots
        )
        self.module.verify(self.output, *self.snapshots, False)

    def test_migration_rejects_any_second_stale_record_or_revision(self) -> None:
        self.module.verify(self.output, *self.snapshots, True)
        (self.output / "mage_edit_variants_manifest.json").write_bytes(b"migrated")
        manifest_path = self.output / self.module.MANIFEST
        original = manifest_path.read_text(encoding="utf-8")

        (self.output / self.module.FILES[0]).write_bytes(b"also-stale")
        with self.assertRaises(self.module.InvalidTransfer):
            self.module.migrate_edit_variant_manifest_hash_only(
                self.output, *self.snapshots
            )
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), original)

        (self.output / self.module.FILES[0]).write_bytes(self.module.FILES[0].encode())
        document = json.loads(original)
        document["generationSnapshotRevision"] = "f" * 40
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(self.module.InvalidTransfer):
            self.module.migrate_edit_variant_manifest_hash_only(
                self.output, *self.snapshots
            )

    def test_migration_rejects_a_linked_transfer_manifest(self) -> None:
        self.module.verify(self.output, *self.snapshots, True)
        (self.output / "mage_edit_variants_manifest.json").write_bytes(b"migrated")
        manifest_path = self.output / self.module.MANIFEST
        external = self.root / "external-manifest.json"
        manifest_path.replace(external)
        os.link(external, manifest_path)
        with self.assertRaises(self.module.InvalidTransfer):
            self.module.migrate_edit_variant_manifest_hash_only(
                self.output, *self.snapshots
            )
