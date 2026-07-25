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
    / "verify_mage_candle_oracles.py"
)


def load_script():
    numpy = types.ModuleType("numpy")
    numpy.array_equal = lambda left, right: left == right
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: None
    spec = importlib.util.spec_from_file_location("verify_mage_candle_oracles", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"numpy": numpy, "safetensors": safetensors}
    ):
        spec.loader.exec_module(module)
    return module


class MageCandleOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "oracles"
        self.output.mkdir()
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.revision = "a" * 40
        (self.snapshot / self.module.REVISION_MARKER).write_text(
            self.revision + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def tensor(dtype: str, shape: list[int]) -> dict:
        return {"dtype": dtype, "shape": shape}

    def common(self, filename: str) -> dict:
        return {
            "file": filename,
            "bytes": 123,
            "sha256": f"hash:{filename}",
            "metadata": {
                "device": "cpu",
                "gen_revision": self.revision,
                "prompt": self.module.PROMPT,
            },
            "tensors": {
                "geometry": self.tensor("int32", [4]),
                "seed": self.tensor("int64", [1]),
                "cfg": self.tensor("float32", [1]),
                "gs_key": self.tensor("int64", [1]),
                "drop_idx": self.tensor("int32", [2]),
                "static_shift": self.tensor("float32", [1]),
            },
            "values": {
                "geometry": self.module.GEOMETRY,
                "seed": [42],
                "cfg": [5.0],
                "gs_key": [20260720],
                "drop_idx": [34, 64],
                "static_shift": [6.0],
            },
            "mutationChecks": {
                "ditChangesInput": None,
                "trajectoryChanges": None,
                "finalChangesInitial": None,
                "imageDiscriminating": None,
            },
        }

    def dit(self) -> dict:
        record = self.common(self.module.FILES[0])
        record["tensors"].update(
            {
                "dit_in.img": self.tensor("float32", [1, 8192, 128]),
                "dit_in.txt": self.tensor("float32", [1, 26, 2560]),
                "dit_in.timesteps": self.tensor("float32", [2]),
                "dit_in.img_cu_seqlens": self.tensor("int64", [3]),
                "dit_in.txt_cu_seqlens": self.tensor("int64", [3]),
                "dit_out": self.tensor("float32", [1, 8192, 128]),
                "img_shapes": self.tensor("int32", [2, 3]),
            }
        )
        record["values"].update(
            {
                "dit_in.timesteps": [1.0, 1.0],
                "dit_in.img_cu_seqlens": [0, 4096, 8192],
                "dit_in.txt_cu_seqlens": [0, 20, 26],
                "img_shapes": self.module.EXPECTED_IMG_SHAPES,
            }
        )
        record["mutationChecks"]["ditChangesInput"] = True
        return record

    def e2e(self) -> dict:
        record = self.common(self.module.FILES[1])
        record["tensors"].update(
            {
                "final_tokens": self.tensor("float32", [1, 4096, 128]),
                "final_latent": self.tensor("float32", [1, 128, 64, 64]),
                "image_u8": self.tensor("uint8", [1024, 1024, 3]),
                "traj_step0": self.tensor("float32", [1, 8192, 128]),
                "traj_step1": self.tensor("float32", [1, 8192, 128]),
                "img_shapes": self.tensor("int32", [2, 3]),
            }
        )
        record["values"]["img_shapes"] = self.module.EXPECTED_IMG_SHAPES
        for steps in (20, 4, 30):
            record["tensors"][f"sigmas_{steps}"] = self.tensor(
                "float32", [steps + 1]
            )
            record["tensors"][f"timesteps_{steps}"] = self.tensor(
                "float32", [steps]
            )
            sigmas, timesteps = self.module.expected_schedule(steps)
            record["values"][f"sigmas_{steps}"] = sigmas
            record["values"][f"timesteps_{steps}"] = timesteps
        record["mutationChecks"]["trajectoryChanges"] = True
        record["mutationChecks"]["finalChangesInitial"] = True
        record["mutationChecks"]["imageDiscriminating"] = True
        return record

    def test_accepts_exact_schema_values_mutations_and_hash_manifest(self) -> None:
        records = [self.dit(), self.e2e()]
        expected = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "geometry": self.module.GEOMETRY,
            "files": records,
        }
        (self.output / self.module.MANIFEST).write_text(
            json.dumps(expected), encoding="utf-8"
        )
        with mock.patch.object(self.module, "inspect", side_effect=records):
            self.module.verify(self.output, self.snapshot, False)

    def test_rejects_absent_corrupt_and_stale_manifest(self) -> None:
        with self.assertRaisesRegex(self.module.InvalidOracle, "missing"):
            self.module.inspect(self.output / self.module.FILES[0])
        corrupt = self.output / self.module.FILES[0]
        corrupt.write_bytes(b"not safetensors")
        with (
            mock.patch.object(
                self.module,
                "safe_open",
                side_effect=ValueError("corrupt header"),
            ),
            self.assertRaisesRegex(self.module.InvalidOracle, "cannot inspect"),
        ):
            self.module.inspect(corrupt)
        (self.output / self.module.MANIFEST).write_text("{}", encoding="utf-8")
        with (
            mock.patch.object(self.module, "inspect", side_effect=[self.dit(), self.e2e()]),
            self.assertRaisesRegex(self.module.InvalidOracle, "manifest .* stale"),
        ):
            self.module.verify(self.output, self.snapshot, False)

    def test_rejects_schema_value_mutation_and_hash_drift(self) -> None:
        missing = self.dit()
        del missing["tensors"]["dit_out"]
        with self.assertRaisesRegex(self.module.InvalidOracle, "population"):
            self.module.validate_dit(missing, self.revision)
        wrong_geometry = self.e2e()
        wrong_geometry["values"]["geometry"] = [512, 512, 20, 4]
        with self.assertRaisesRegex(self.module.InvalidOracle, "geometry.*stale"):
            self.module.validate_e2e(wrong_geometry, self.revision)
        no_trajectory = self.e2e()
        no_trajectory["mutationChecks"]["trajectoryChanges"] = False
        with self.assertRaisesRegex(self.module.InvalidOracle, "step0 equals step1"):
            self.module.validate_e2e(no_trajectory, self.revision)
        extra = self.e2e()
        extra["tensors"]["unexpected"] = self.tensor("float32", [1])
        with self.assertRaisesRegex(self.module.InvalidOracle, "population"):
            self.module.validate_e2e(extra, self.revision)
        wrong_dtype = self.dit()
        wrong_dtype["tensors"]["dit_in.img"]["dtype"] = "float16"
        with self.assertRaisesRegex(self.module.InvalidOracle, "dtype"):
            self.module.validate_dit(wrong_dtype, self.revision)
        wrong_channel = self.e2e()
        wrong_channel["tensors"]["image_u8"]["shape"] = [1024, 1024, 4]
        with self.assertRaisesRegex(self.module.InvalidOracle, "shape"):
            self.module.validate_e2e(wrong_channel, self.revision)
        wrong_text = self.dit()
        wrong_text["tensors"]["dit_in.txt"]["shape"] = [1, 25, 2560]
        with self.assertRaisesRegex(self.module.InvalidOracle, "shape"):
            self.module.validate_dit(wrong_text, self.revision)
        wrong_shapes = self.e2e()
        wrong_shapes["values"]["img_shapes"] = [[1, 64, 64], [0, 64, 64]]
        with self.assertRaisesRegex(self.module.InvalidOracle, "img_shapes.*stale"):
            self.module.validate_e2e(wrong_shapes, self.revision)
        wrong_schedule = self.e2e()
        wrong_schedule["values"]["sigmas_20"][5], wrong_schedule["values"]["sigmas_20"][6] = (
            wrong_schedule["values"]["sigmas_20"][6],
            wrong_schedule["values"]["sigmas_20"][5],
        )
        with self.assertRaisesRegex(self.module.InvalidOracle, "ordered values"):
            self.module.validate_e2e(wrong_schedule, self.revision)
        wrong_scalar = self.e2e()
        wrong_scalar["values"]["gs_key"] = [1]
        with self.assertRaisesRegex(self.module.InvalidOracle, "gs_key.*stale"):
            self.module.validate_e2e(wrong_scalar, self.revision)
        nondiscriminating = self.e2e()
        nondiscriminating["mutationChecks"]["imageDiscriminating"] = False
        with self.assertRaisesRegex(self.module.InvalidOracle, "non-discriminating"):
            self.module.validate_e2e(nondiscriminating, self.revision)

        records = [self.dit(), self.e2e()]
        stale = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "geometry": self.module.GEOMETRY,
            "files": records,
        }
        stale["files"][1]["sha256"] = "stale"
        (self.output / self.module.MANIFEST).write_text(
            json.dumps(stale), encoding="utf-8"
        )
        with (
            mock.patch.object(self.module, "inspect", side_effect=[self.dit(), self.e2e()]),
            self.assertRaisesRegex(self.module.InvalidOracle, "manifest .* stale"),
        ):
            self.module.verify(self.output, self.snapshot, False)


if __name__ == "__main__":
    unittest.main()
