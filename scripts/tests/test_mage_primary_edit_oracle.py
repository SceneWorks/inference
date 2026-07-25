from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1] / "release" / "provision_mage_oracles.py"
)


def load_script():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: None
    spec = importlib.util.spec_from_file_location("provision_mage_oracles_primary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": numpy, "safetensors": safetensors}):
        spec.loader.exec_module(module)
    return module


class MagePrimaryEditOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()

    def values(self) -> dict[str, object]:
        sigmas, timesteps = self.module._expected_schedule(30)
        return {
            "geometry": [256, 256, 4, 30],
            "seed": [42],
            "cfg": [5.0],
            "gs_key": [20260720],
            "drop_idx": [34, 64],
            "static_shift": [6.0],
            "target_tokens": [256],
            "img_shapes": self.module.EDIT_IMG_SHAPES,
            "sigmas_30": sigmas,
            "timesteps_30": timesteps,
        }

    @staticmethod
    def checks() -> dict[str, bool]:
        return {
            "trajectoryChanges": True,
            "targetDiffersReference": True,
            "finalChangesInitial": True,
            "imageDiscriminating": True,
            "referenceDiscriminating": True,
            "imageDiffersReference": True,
        }

    def test_exact_values_and_discrimination_are_load_bearing(self) -> None:
        self.module._validate_edit_policy_record(self.values(), self.checks())
        for key, replacement in (
            ("seed", [43]),
            ("cfg", [1.0]),
            ("gs_key", [0]),
            ("drop_idx", [34, 63]),
            ("static_shift", [5.0]),
            ("target_tokens", [255]),
            ("img_shapes", [[1, 16, 16]] * 3 + [[0, 16, 16]]),
        ):
            mutated = self.values()
            mutated[key] = replacement
            with self.assertRaises(self.module.InvalidOracle):
                self.module._validate_edit_policy_record(mutated, self.checks())
        schedule = self.values()
        schedule["sigmas_30"][4], schedule["sigmas_30"][5] = (
            schedule["sigmas_30"][5],
            schedule["sigmas_30"][4],
        )
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_edit_policy_record(schedule, self.checks())
        for name in self.checks():
            checks = self.checks()
            checks[name] = False
            with self.assertRaises(self.module.InvalidOracle):
                self.module._validate_edit_policy_record(self.values(), checks)

    def test_exact_schema_rejects_extra_dtype_shape_and_channel_mutations(self) -> None:
        schema = self.module.EDIT_SCHEMA
        shapes = {key: shape for key, (_dtype, shape) in schema.items()}
        dtypes = {key: dtype for key, (dtype, _shape) in schema.items()}
        self.module._validate_schema("edit", schema, set(schema), shapes, dtypes)
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_schema(
                "edit", schema, {*schema, "unexpected"}, shapes, dtypes
            )
        wrong_dtype = dict(dtypes)
        wrong_dtype["final_tokens"] = "F16"
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_schema(
                "edit", schema, set(schema), shapes, wrong_dtype
            )
        wrong_shape = dict(shapes)
        wrong_shape["seq_step0"] = [1, 768, 128]
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_schema(
                "edit", schema, set(schema), wrong_shape, dtypes
            )
        wrong_channel = dict(shapes)
        wrong_channel["image_u8"] = [256, 256, 4]
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_schema(
                "edit", schema, set(schema), wrong_channel, dtypes
            )


if __name__ == "__main__":
    unittest.main()
