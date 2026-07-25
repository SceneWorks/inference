from __future__ import annotations

import importlib.util
import sys
import tempfile
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
    numpy.floating = object()
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
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            values = self.values()
            values["timesteps_30"][3] = nonfinite
            with self.assertRaises(self.module.InvalidOracle):
                self.module._validate_edit_policy_record(values, self.checks())
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

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored reference",
            "snapshotRevision": "a" * 40,
            "editSnapshotRevision": "b" * 40,
            "device": "cpu",
            "vaeGeometries": list(self.module.GEOMETRIES),
            "generationSeconds": 12.5,
            "referenceEnvironment": dict(self.module.REFERENCE_PACKAGES),
            "files": [
                {
                    "name": self.module.EDIT_FILE,
                    "bytes": 123,
                    "sha256": "c" * 64,
                }
            ],
        }

    def test_standalone_manifest_header_is_exact_and_finite(self) -> None:
        expected = {self.module.EDIT_FILE}
        self.module._validate_manifest_header(
            self.manifest(), "a" * 40, "b" * 40, expected
        )
        mutations = []
        for key, value in (
            ("schema", 2),
            ("reference", "wrong"),
            ("snapshotRevision", "d" * 40),
            ("editSnapshotRevision", "e" * 40),
            ("device", "mps"),
            ("vaeGeometries", ["256"]),
            ("referenceEnvironment", {}),
            ("generationSeconds", float("nan")),
            ("generationSeconds", True),
        ):
            document = self.manifest()
            document[key] = value
            mutations.append(document)
        extra = self.manifest()
        extra["unexpected"] = True
        mutations.append(extra)
        wrong_hash = self.manifest()
        wrong_hash["files"][0]["sha256"] = "bad"
        mutations.append(wrong_hash)
        wrong_size = self.manifest()
        wrong_size["files"][0]["bytes"] = 0
        mutations.append(wrong_size)
        bool_size = self.manifest()
        bool_size["files"][0]["bytes"] = True
        mutations.append(bool_size)
        for document in mutations:
            with self.assertRaises(self.module.InvalidOracle):
                self.module._validate_manifest_header(
                    document, "a" * 40, "b" * 40, expected
                )

    def test_reference_metadata_population_and_values_are_exact(self) -> None:
        metadata = {
            "prompt": self.module.PROMPT,
            "negative_prompt": " ",
            "edit_instruction": self.module.EDIT_INSTRUCTION,
            "edit_ref": "dog.jpg",
            "device": "cpu",
            "attn": "sdpa",
            "reference": self.module.REFERENCE,
            "gen_revision": "a" * 40,
            "edit_revision": "b" * 40,
        }
        self.module._validate_reference_metadata(
            self.module.EDIT_FILE, metadata, "a" * 40, "b" * 40
        )
        for key in metadata:
            mutated = dict(metadata)
            mutated[key] = "wrong"
            with self.assertRaises(self.module.InvalidOracle):
                self.module._validate_reference_metadata(
                    self.module.EDIT_FILE, mutated, "a" * 40, "b" * 40
                )
        extra = dict(metadata)
        extra["unexpected"] = "value"
        with self.assertRaises(self.module.InvalidOracle):
            self.module._validate_reference_metadata(
                self.module.EDIT_FILE, extra, "a" * 40, "b" * 40
            )

    def test_manifest_hash_and_size_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / self.module.EDIT_FILE
            path.write_bytes(b"oracle")
            record = {
                "name": self.module.EDIT_FILE,
                "bytes": path.stat().st_size,
                "sha256": self.module._sha256(path),
            }
            self.module._validate_manifest_file_record(record, path)
            for key, value in (
                ("bytes", record["bytes"] + 1),
                ("bytes", True),
                ("sha256", "0" * 64),
            ):
                mutated = dict(record)
                mutated[key] = value
                with self.assertRaises(self.module.InvalidOracle):
                    self.module._validate_manifest_file_record(mutated, path)

    def test_every_float_tensor_must_be_finite(self) -> None:
        class Tensor:
            dtype = "float32"

        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def keys():
                return ["weights"]

            @staticmethod
            def get_tensor(_key):
                return Tensor()

        class FiniteResult:
            def __init__(self, value: bool):
                self.value = value

            def all(self):
                return self.value

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oracle.safetensors"
            path.touch()
            with (
                mock.patch.object(self.module, "safe_open", return_value=Handle()),
                mock.patch.object(
                    self.module.np, "issubdtype", return_value=True, create=True
                ),
                mock.patch.object(
                    self.module.np,
                    "isfinite",
                    return_value=FiniteResult(True),
                    create=True,
                ),
            ):
                self.module._validate_finite_tensors(path)
            with (
                mock.patch.object(self.module, "safe_open", return_value=Handle()),
                mock.patch.object(
                    self.module.np, "issubdtype", return_value=True, create=True
                ),
                mock.patch.object(
                    self.module.np,
                    "isfinite",
                    return_value=FiniteResult(False),
                    create=True,
                ),
                self.assertRaises(self.module.InvalidOracle),
            ):
                self.module._validate_finite_tensors(path)

    def test_producer_environment_rejects_hostile_ambient_mage_inputs(self) -> None:
        output = Path("/tmp/oracles")
        snapshot = Path("/tmp/gen")
        edit_snapshot = Path("/tmp/edit")
        hostile = {
            "MAGE_EDIT_REF": "/tmp/hostile/dog.jpg",
            "MAGE_PROMPT": "hostile prompt",
            "MAGE_SEED": "999",
            "MAGE_ATTN": "flash2",
            "PATH": "/trusted/bin",
        }
        with mock.patch.dict(self.module.os.environ, hostile, clear=True):
            env = self.module._producer_environment(output, snapshot, edit_snapshot)
        self.assertEqual(env["MAGE_EDIT_REF"], str(self.module.CANONICAL_EDIT_REF))
        self.assertEqual(env["MAGE_PROMPT"], self.module.PROMPT)
        self.assertEqual(env["MAGE_SEED"], "42")
        self.assertEqual(env["MAGE_ATTN"], "sdpa")
        self.assertEqual(env["PATH"], "/trusted/bin")
        self.assertEqual(
            {key for key in env if key.startswith("MAGE_")},
            {
                "MAGE_DEVICE",
                "MAGE_ATTN",
                "MAGE_SNAPSHOT",
                "MAGE_EDIT_SNAPSHOT",
                "MAGE_GOLDEN_DIR",
                "MAGE_PROMPT",
                "MAGE_NEG",
                "MAGE_EDIT_INSTRUCTION",
                "MAGE_EDIT_REF",
                "MAGE_SEED",
                "MAGE_H",
                "MAGE_W",
                "MAGE_STEPS",
                "MAGE_EDIT_STEPS",
                "MAGE_CFG",
                "MAGE_GS_KEY",
                "MAGE_VAE_SIZES",
            },
        )


if __name__ == "__main__":
    unittest.main()
