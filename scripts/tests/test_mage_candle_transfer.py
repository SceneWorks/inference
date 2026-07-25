from __future__ import annotations

import importlib.util
import json
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
