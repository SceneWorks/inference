from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "provision_mage_edit_variants.py"
)


def load_script():
    numpy = types.ModuleType("numpy")
    numpy.int64 = object()
    numpy.ndarray = object
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: None
    spec = importlib.util.spec_from_file_location("provision_mage_edit_variants", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"numpy": numpy, "safetensors": safetensors}
    ):
        spec.loader.exec_module(module)
    return module


class MageEditVariantOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "oracles"
        self.output.mkdir()
        self.revisions = {
            "edit": "1" * 40,
            "edit-base": "2" * 40,
            "edit-turbo": "3" * 40,
        }
        self.snapshots = {}
        for label, revision in self.revisions.items():
            snapshot = self.root / label
            snapshot.mkdir()
            (snapshot / self.module.REVISION_MARKER).write_text(
                revision + "\n", encoding="utf-8"
            )
            self.snapshots[label] = snapshot
        for _label, filename, _steps, _cfg in self.module.CASES:
            (self.output / filename).write_bytes(filename.encode())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, verify_only: bool = True) -> list[str]:
        args = [
            str(SCRIPT),
            "--edit",
            str(self.snapshots["edit"]),
            "--edit-base",
            str(self.snapshots["edit-base"]),
            "--edit-turbo",
            str(self.snapshots["edit-turbo"]),
            "--output",
            str(self.output),
        ]
        if verify_only:
            args.append("--verify-only")
        return args

    def manifest(self) -> dict:
        records = []
        for label, filename, steps, cfg in self.module.CASES:
            path = self.output / filename
            records.append(
                {
                    "variant": label,
                    "snapshotRevision": self.revisions[label],
                    "file": filename,
                    "bytes": path.stat().st_size,
                    "sha256": f"hash:{filename}",
                    "cfg": cfg,
                    "steps": steps,
                }
            )
        return {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored reference",
            "device": "cpu",
            "files": records,
        }

    def test_verify_only_checks_every_variant_and_never_regenerates(self) -> None:
        (self.output / "mage_edit_variants_manifest.json").write_text(
            json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8"
        )
        with (
            mock.patch.object(sys, "argv", self.args()),
            mock.patch.object(self.module, "sha256", side_effect=lambda path: f"hash:{path.name}"),
            mock.patch.object(self.module, "validate") as validate,
            mock.patch.object(
                self.module.subprocess,
                "run",
                side_effect=AssertionError("verify-only must not regenerate"),
            ),
        ):
            self.assertEqual(self.module.main(), 0)
        self.assertEqual(
            validate.call_args_list,
            [
                mock.call(
                    self.output / filename,
                    self.revisions[label],
                    steps,
                    cfg,
                )
                for label, filename, steps, cfg in self.module.CASES
            ],
        )

    def test_verify_only_rejects_revision_geometry_cfg_or_hash_manifest_drift(self) -> None:
        manifest = self.manifest()
        manifest["files"][1]["sha256"] = "stale"
        (self.output / "mage_edit_variants_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (
            mock.patch.object(sys, "argv", self.args()),
            mock.patch.object(self.module, "sha256", side_effect=lambda path: f"hash:{path.name}"),
            mock.patch.object(self.module, "validate"),
            self.assertRaisesRegex(RuntimeError, "manifest .* stale"),
        ):
            self.module.main()


if __name__ == "__main__":
    unittest.main()
