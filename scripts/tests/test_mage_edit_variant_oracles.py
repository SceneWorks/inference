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

    def record(self, steps: int = 30, cfg: float = 5.0) -> dict:
        sigmas, timesteps = self.module.expected_schedule(steps)
        return {
            "metadata": {
                "device": "cpu",
                "edit_revision": self.revisions["edit"],
                "prompt": self.module.PROMPT,
                "edit_instruction": self.module.EDIT_INSTRUCTION,
            },
            "tensors": {
                key: {"dtype": dtype, "shape": shape}
                for key, (dtype, shape) in self.module.edit_schema(steps, cfg).items()
            },
            "values": {
                "geometry": [256, 256, 4, steps],
                "seed": [42],
                "cfg": [cfg],
                "gs_key": [20260720],
                "drop_idx": [34, 64],
                "static_shift": [6.0],
                "target_tokens": [256],
                "img_shapes": self.module.expected_img_shapes(cfg),
                f"sigmas_{steps}": sigmas,
                f"timesteps_{steps}": timesteps,
            },
            "mutationChecks": {
                "trajectoryChanges": True,
                "targetDiffersReference": True,
                "finalChangesInitial": True,
                "imageDiscriminating": True,
                "referenceDiscriminating": True,
                "imageDiffersReference": True,
            },
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

    def test_exact_schema_values_and_discrimination_reject_producer_mutations(self) -> None:
        self.module.validate_record(self.record(), self.revisions["edit"], 30, 5.0)
        self.module.validate_record(self.record(4, 1.0), self.revisions["edit"], 4, 1.0)
        mutations = []
        extra = self.record()
        extra["tensors"]["unexpected"] = {"dtype": "float32", "shape": [1]}
        mutations.append(extra)
        wrong_dtype = self.record()
        wrong_dtype["tensors"]["final_tokens"]["dtype"] = "float16"
        mutations.append(wrong_dtype)
        wrong_shape = self.record()
        wrong_shape["tensors"]["seq_step0"]["shape"] = [1, 768, 128]
        mutations.append(wrong_shape)
        wrong_channel = self.record()
        wrong_channel["tensors"]["image_u8"]["shape"] = [256, 256, 4]
        mutations.append(wrong_channel)
        wrong_img_shapes = self.record()
        wrong_img_shapes["values"]["img_shapes"] = [[1, 16, 16]] * 3 + [[0, 16, 16]]
        mutations.append(wrong_img_shapes)
        wrong_scalar = self.record()
        wrong_scalar["values"]["seed"] = [43]
        mutations.append(wrong_scalar)
        wrong_schedule = self.record()
        wrong_schedule["values"]["timesteps_30"][2], wrong_schedule["values"]["timesteps_30"][3] = (
            wrong_schedule["values"]["timesteps_30"][3],
            wrong_schedule["values"]["timesteps_30"][2],
        )
        mutations.append(wrong_schedule)
        for check in self.record()["mutationChecks"]:
            failed = self.record()
            failed["mutationChecks"][check] = False
            mutations.append(failed)
        for mutated in mutations:
            with self.assertRaises(RuntimeError):
                self.module.validate_record(mutated, self.revisions["edit"], 30, 5.0)


if __name__ == "__main__":
    unittest.main()
