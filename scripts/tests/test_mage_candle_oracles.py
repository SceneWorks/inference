from __future__ import annotations

import importlib.util
import json
import copy
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
        self.edit_snapshot = self.root / "edit-snapshot"
        self.edit_snapshot.mkdir()
        self.edit_revision = "b" * 40
        (self.edit_snapshot / self.module.REVISION_MARKER).write_text(
            self.edit_revision + "\n", encoding="utf-8"
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
                "negative_prompt": " ",
                "edit_instruction": self.module.EDIT_INSTRUCTION,
                "edit_ref": "dog.jpg",
                "device": "cpu",
                "attn": "sdpa",
                "reference": self.module.REFERENCE,
                "gen_revision": self.revision,
                "edit_revision": self.edit_revision,
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
            "finiteChecks": {},
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
        record["finiteChecks"] = {
            key: True
            for key, tensor in record["tensors"].items()
            if tensor["dtype"].startswith("float")
        }
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
        record["finiteChecks"] = {
            key: True
            for key, tensor in record["tensors"].items()
            if tensor["dtype"].startswith("float")
        }
        return record

    def test_accepts_exact_schema_values_mutations_and_hash_manifest(self) -> None:
        records = [self.dit(), self.e2e()]
        expected = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "editSnapshotRevision": self.edit_revision,
            "geometry": self.module.GEOMETRY,
            "files": records,
        }
        (self.output / self.module.MANIFEST).write_text(
            json.dumps(expected), encoding="utf-8"
        )
        with mock.patch.object(self.module, "inspect", side_effect=records):
            self.module.verify(self.output, self.snapshot, self.edit_snapshot, False)

    def test_manifest_rejects_every_header_revision_and_population_mutation(self) -> None:
        records = [self.dit(), self.e2e()]
        expected = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "geometry": self.module.GEOMETRY,
            "editSnapshotRevision": self.edit_revision,
            "files": records,
        }
        mutations = []
        for key, value in (
            ("schema", 2),
            ("reference", "wrong"),
            ("snapshotRevision", "c" * 40),
            ("editSnapshotRevision", "d" * 40),
            ("geometry", [512, 512, 20, 4]),
        ):
            document = dict(expected)
            document[key] = value
            mutations.append(document)
        extra = dict(expected)
        extra["unexpected"] = True
        mutations.append(extra)
        missing = dict(expected)
        missing["files"] = records[:1]
        mutations.append(missing)
        for document in mutations:
            (self.output / self.module.MANIFEST).write_text(
                json.dumps(document), encoding="utf-8"
            )
            with (
                mock.patch.object(
                    self.module, "inspect", side_effect=[self.dit(), self.e2e()]
                ),
                self.assertRaisesRegex(self.module.InvalidOracle, "manifest .* stale"),
            ):
                self.module.verify(
                    self.output, self.snapshot, self.edit_snapshot, False
                )

    def test_manifest_rejects_every_cross_type_numeric_alias(self) -> None:
        records = [self.dit(), self.e2e()]
        for record in records:
            record["bytes"] = 1
        expected = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "geometry": self.module.GEOMETRY,
            "editSnapshotRevision": self.edit_revision,
            "files": records,
        }

        def numeric_aliases(value, path=()):
            found = []
            if type(value) is bool:
                found.append((path, int(value)))
            elif type(value) is int:
                found.append((path, float(value)))
                if value in (0, 1):
                    found.append((path, bool(value)))
            elif type(value) is float and value.is_integer():
                found.append((path, int(value)))
                if value in (0.0, 1.0):
                    found.append((path, bool(value)))
            elif isinstance(value, dict):
                for key, nested in value.items():
                    found.extend(numeric_aliases(nested, (*path, key)))
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    found.extend(numeric_aliases(nested, (*path, index)))
            return found

        mutations = numeric_aliases(expected)
        self.assertGreater(len(mutations), 100)
        for path, replacement in mutations:
            document = copy.deepcopy(expected)
            target = document
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = replacement
            (self.output / self.module.MANIFEST).write_text(
                json.dumps(document), encoding="utf-8"
            )
            with (
                self.subTest(path=path),
                mock.patch.object(
                    self.module,
                    "inspect",
                    side_effect=[copy.deepcopy(records[0]), copy.deepcopy(records[1])],
                ),
                self.assertRaisesRegex(self.module.InvalidOracle, "manifest .* stale"),
            ):
                self.module.verify(
                    self.output, self.snapshot, self.edit_snapshot, False
                )

    def test_nan_and_inf_never_pass_schedule_tolerance(self) -> None:
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            record = self.e2e()
            record["values"]["sigmas_20"][1] = nonfinite
            with self.assertRaisesRegex(self.module.InvalidOracle, "ordered values"):
                self.module.require_values(
                    record,
                    "sigmas_20",
                    self.module.expected_schedule(20)[0],
                    1.0e-4,
                )

    def test_every_scalar_cu_and_full_schedule_is_load_bearing(self) -> None:
        for key, replacement in (
            ("geometry", [512, 1024, 20, 4]),
            ("seed", [43]),
            ("cfg", [4.0]),
            ("gs_key", [0]),
            ("drop_idx", [34, 63]),
            ("static_shift", [5.0]),
        ):
            record = self.e2e()
            record["values"][key] = replacement
            with self.assertRaises(self.module.InvalidOracle):
                self.module.validate_e2e(
                    record, self.revision, self.edit_revision
                )
        for key, replacement in (
            ("dit_in.img_cu_seqlens", [0, 4095, 8192]),
            ("dit_in.txt_cu_seqlens", [0, 19, 26]),
        ):
            record = self.dit()
            record["values"][key] = replacement
            with self.assertRaises(self.module.InvalidOracle):
                self.module.validate_dit(
                    record, self.revision, self.edit_revision
                )
        for steps in (20, 4, 30):
            for prefix in ("sigmas", "timesteps"):
                key = f"{prefix}_{steps}"
                for mutation in ("order", "value", "nan", "inf"):
                    record = self.e2e()
                    values = record["values"][key]
                    if mutation == "order":
                        values[1], values[2] = values[2], values[1]
                    elif mutation == "value":
                        values[1] += 0.25
                    elif mutation == "nan":
                        values[1] = float("nan")
                    else:
                        values[1] = float("inf")
                    with self.assertRaises(self.module.InvalidOracle):
                        self.module.validate_e2e(
                            record, self.revision, self.edit_revision
                        )

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
            self.module.verify(self.output, self.snapshot, self.edit_snapshot, False)

    def test_rejects_schema_value_mutation_and_hash_drift(self) -> None:
        missing = self.dit()
        del missing["tensors"]["dit_out"]
        with self.assertRaisesRegex(self.module.InvalidOracle, "population"):
            self.module.validate_dit(missing, self.revision, self.edit_revision)
        wrong_geometry = self.e2e()
        wrong_geometry["values"]["geometry"] = [512, 512, 20, 4]
        with self.assertRaisesRegex(self.module.InvalidOracle, "geometry.*stale"):
            self.module.validate_e2e(wrong_geometry, self.revision, self.edit_revision)
        no_trajectory = self.e2e()
        no_trajectory["mutationChecks"]["trajectoryChanges"] = False
        with self.assertRaisesRegex(self.module.InvalidOracle, "step0 equals step1"):
            self.module.validate_e2e(no_trajectory, self.revision, self.edit_revision)
        extra = self.e2e()
        extra["tensors"]["unexpected"] = self.tensor("float32", [1])
        with self.assertRaisesRegex(self.module.InvalidOracle, "population"):
            self.module.validate_e2e(extra, self.revision, self.edit_revision)
        wrong_dtype = self.dit()
        wrong_dtype["tensors"]["dit_in.img"]["dtype"] = "float16"
        with self.assertRaisesRegex(self.module.InvalidOracle, "dtype"):
            self.module.validate_dit(wrong_dtype, self.revision, self.edit_revision)
        wrong_channel = self.e2e()
        wrong_channel["tensors"]["image_u8"]["shape"] = [1024, 1024, 4]
        with self.assertRaisesRegex(self.module.InvalidOracle, "shape"):
            self.module.validate_e2e(wrong_channel, self.revision, self.edit_revision)
        wrong_text = self.dit()
        wrong_text["tensors"]["dit_in.txt"]["shape"] = [1, 25, 2560]
        with self.assertRaisesRegex(self.module.InvalidOracle, "shape"):
            self.module.validate_dit(wrong_text, self.revision, self.edit_revision)
        wrong_shapes = self.e2e()
        wrong_shapes["values"]["img_shapes"] = [[1, 64, 64], [0, 64, 64]]
        with self.assertRaisesRegex(self.module.InvalidOracle, "img_shapes.*stale"):
            self.module.validate_e2e(wrong_shapes, self.revision, self.edit_revision)
        wrong_schedule = self.e2e()
        wrong_schedule["values"]["sigmas_20"][5], wrong_schedule["values"]["sigmas_20"][6] = (
            wrong_schedule["values"]["sigmas_20"][6],
            wrong_schedule["values"]["sigmas_20"][5],
        )
        with self.assertRaisesRegex(self.module.InvalidOracle, "ordered values"):
            self.module.validate_e2e(wrong_schedule, self.revision, self.edit_revision)
        wrong_scalar = self.e2e()
        wrong_scalar["values"]["gs_key"] = [1]
        with self.assertRaisesRegex(self.module.InvalidOracle, "gs_key.*stale"):
            self.module.validate_e2e(wrong_scalar, self.revision, self.edit_revision)
        nondiscriminating = self.e2e()
        nondiscriminating["mutationChecks"]["imageDiscriminating"] = False
        with self.assertRaisesRegex(self.module.InvalidOracle, "non-discriminating"):
            self.module.validate_e2e(nondiscriminating, self.revision, self.edit_revision)
        nonfinite = self.e2e()
        nonfinite["finiteChecks"]["final_latent"] = False
        with self.assertRaisesRegex(self.module.InvalidOracle, "non-finite"):
            self.module.validate_e2e(nonfinite, self.revision, self.edit_revision)
        metadata_extra = self.dit()
        metadata_extra["metadata"]["unexpected"] = "value"
        with self.assertRaisesRegex(self.module.InvalidOracle, "metadata population"):
            self.module.validate_dit(metadata_extra, self.revision, self.edit_revision)
        metadata_wrong = self.dit()
        metadata_wrong["metadata"]["attn"] = "flash2"
        with self.assertRaisesRegex(self.module.InvalidOracle, "metadata population"):
            self.module.validate_dit(metadata_wrong, self.revision, self.edit_revision)
        bad_cu = self.dit()
        bad_cu["values"]["dit_in.img_cu_seqlens"] = [0, 4095, 8192]
        with self.assertRaisesRegex(self.module.InvalidOracle, "cu_seqlens.*stale"):
            self.module.validate_dit(bad_cu, self.revision, self.edit_revision)
        for nonfinite_value in (float("nan"), float("inf"), -float("inf")):
            bad = self.e2e()
            bad["values"]["sigmas_4"][1] = nonfinite_value
            with self.assertRaisesRegex(self.module.InvalidOracle, "ordered values"):
                self.module.validate_e2e(bad, self.revision, self.edit_revision)

        records = [self.dit(), self.e2e()]
        stale = {
            "schema": 1,
            "reference": "microsoft/Mage frozen vendored CPU reference",
            "snapshotRevision": self.revision,
            "editSnapshotRevision": self.edit_revision,
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
            self.module.verify(self.output, self.snapshot, self.edit_snapshot, False)


if __name__ == "__main__":
    unittest.main()
