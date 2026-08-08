import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.release.verify_residency_ab import (
    PREFIX,
    parse_expected_parity,
    read_record,
    verify,
)


REVISION = "a" * 40
SCENEWORKS_REVISION = "b" * 40
MODEL_REVISION = "c" * 40
MODEL_INVENTORY_SHA256 = "d" * 64
FINGERPRINT = "test-layout-v1"
OUTPUT = b"exact parity output"
OUTPUT_SHA256 = hashlib.sha256(OUTPUT).hexdigest()
EXACT = parse_expected_parity("exact")
TOLERANCE = parse_expected_parity("tolerance:mean_abs_u8_subpixel:4.0")


def record(strategy: str, peak: int, output: bytes = OUTPUT, parity: dict | None = None) -> dict:
    composition = ["resident"]
    if strategy == "staged_residency":
        composition.append("staged_residency")
    return {
        "schema_version": 1,
        "key": {
            "resolved_route": "flux1_dev",
            "backend": "candle",
            "tier": {
                "precision": "bf16",
                "quant": "q8",
                "component_precision_floors": [],
            },
            "load_shape": "eager_materialization",
            "mode": "text_to_image",
            "overlay": None,
            "geometry": {
                "width": 1024,
                "height": 1024,
                "batch": 1,
                "frames": 1,
                "reference_count": 0,
            },
            "strategy": strategy,
            "engaged_composition": composition,
            "parameters": {
                "decode_tile_edge": None,
                "decode_overlap": None,
                "attention_chunk_size": None,
                "transformer_window_size": None,
                "transformer_window_component": None,
            },
        },
        "declared_calibration": {
            "abi": 3,
            "fingerprint": FINGERPRINT,
            "load_shape": "eager_materialization",
        },
        "observed_calibration": {
            "abi": 3,
            "fingerprint": FINGERPRINT,
            "load_shape": "eager_materialization",
        },
        "predicted_peak_bytes": peak,
        "observed_peak_bytes": peak,
        "inference_revision": REVISION,
        "sceneworks_revision": SCENEWORKS_REVISION,
        "model_revision": MODEL_REVISION,
        "model_inventory_sha256": MODEL_INVENTORY_SHA256,
        "harness_version": "test-harness-v1",
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "parity": dict(parity) if parity is not None else {"kind": "exact"},
        "parity_result": {"kind": "not_run"},
    }


class VerifyResidencyAbTests(unittest.TestCase):
    def write_log(self, root: Path, name: str, payload: dict) -> Path:
        path = root / name
        path.write_text(
            "test output\n" + PREFIX + json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path

    def valid_pair(self, root: Path) -> tuple[Path, Path, Path, Path]:
        resident = self.write_log(root, "resident.log", record("resident", 24_000 << 20))
        staged = self.write_log(
            root, "staged.log", record("staged_residency", 15_000 << 20)
        )
        resident_output = root / "resident.rgb"
        staged_output = root / "staged.rgb"
        resident_output.write_bytes(OUTPUT)
        staged_output.write_bytes(OUTPUT)
        return resident, staged, resident_output, staged_output

    def test_accepts_complete_material_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            resident, staged, resident_output, staged_output = self.valid_pair(Path(temporary))
            self.assertEqual(
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                ),
                (24_000 << 20, 15_000 << 20, None),
            )

    def test_rejects_legacy_missing_duplicate_and_malformed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "log"
            path.write_text("SEQ_AB mode=resident peak_mib=1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "legacy SEQ_AB"):
                read_record(path, "resident")
            payload = json.dumps(record("resident", 10))
            path.write_text(f"{PREFIX}{payload}\n{PREFIX}{payload}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                read_record(path, "resident")
            path.write_text(
                f"{PREFIX}{payload}\nSEQ_AB mode=resident peak_mib=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "legacy SEQ_AB"):
                read_record(path, "resident")
            path.write_text(
                f"test output SEQ_AB mode=resident peak_mib=1\n{PREFIX}{payload}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "legacy SEQ_AB"):
                read_record(path, "resident")
            path.write_text(f'{PREFIX}{{"schema_version":1,"schema_version":1}}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                read_record(path, "resident")

    def test_rejects_missing_extra_and_wrongly_typed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate, message in (
                ("missing", lambda value: value.pop("harness_version"), "missing"),
                ("extra", lambda value: value.update({"legacy_peak_mib": 4}), "extra"),
                (
                    "bool-peak",
                    lambda value: value.update({"observed_peak_bytes": True}),
                    "positive integer",
                ),
            ):
                payload = record("resident", 10)
                mutate(payload)
                path = self.write_log(root, name, payload)
                with self.assertRaisesRegex(RuntimeError, message):
                    read_record(path, "resident")

            payload = record("resident", 10)
            payload["parity"] = {
                "kind": "tolerance",
                "metric": "max_abs",
                "maximum_error": float("nan"),
            }
            path = self.write_log(root, "non-finite", payload)
            with self.assertRaisesRegex(RuntimeError, "non-finite JSON number"):
                read_record(path, "resident")

            overflow = record("resident", 10)
            overflow["parity"] = {
                "kind": "tolerance",
                "metric": "max_abs",
                "maximum_error": 0,
            }
            encoded = json.dumps(overflow, separators=(",", ":")).replace(
                '"maximum_error":0', '"maximum_error":1e9999'
            )
            path.write_text(f"{PREFIX}{encoded}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "finite and non-negative"):
                read_record(path, "resident")

    def test_rejects_fingerprint_abi_revision_and_load_shape_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident, staged, resident_output, staged_output = self.valid_pair(root)
            with self.assertRaisesRegex(RuntimeError, "exported calibration fingerprint"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    "other-layout-v1",
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )
            with self.assertRaisesRegex(RuntimeError, "exported ABI"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    4,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )
            with self.assertRaisesRegex(RuntimeError, "model revision"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    "e" * 40,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )
            with self.assertRaisesRegex(RuntimeError, "model inventory SHA-256"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    "f" * 64,
                    resident_output,
                    staged_output,
                    EXACT,
                )

            for name, mutate, message in (
                (
                    "observed-fingerprint",
                    lambda value: value["observed_calibration"].update(
                        {"fingerprint": "other-layout-v1"}
                    ),
                    "identities differ",
                ),
                (
                    "shape",
                    lambda value: value["observed_calibration"].update(
                        {"load_shape": "deferred_materialization"}
                    ),
                    "identities differ",
                ),
                (
                    "revision",
                    lambda value: value.update({"inference_revision": "main"}),
                    "exact lowercase",
                ),
                (
                    "model-inventory",
                    lambda value: value.update({"model_inventory_sha256": "mutable"}),
                    "64 lowercase",
                ),
                (
                    "grammar",
                    lambda value: [
                        value[field].update({"fingerprint": "Layout_V1"})
                        for field in ("declared_calibration", "observed_calibration")
                    ],
                    "lowercase ASCII kebab",
                ),
                (
                    "zero-version",
                    lambda value: [
                        value[field].update({"fingerprint": "layout-v0-v1"})
                        for field in ("declared_calibration", "observed_calibration")
                    ],
                    "exactly one positive",
                ),
            ):
                payload = record("resident", 10)
                mutate(payload)
                path = self.write_log(root, name, payload)
                with self.assertRaisesRegex(RuntimeError, message):
                    read_record(path, "resident")

    def test_rejects_every_non_strategy_ab_invariant_and_small_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident_payload = record("resident", 16_000 << 20)
            resident_output = root / "resident.rgb"
            staged_output = root / "staged.rgb"
            resident_output.write_bytes(OUTPUT)
            staged_output.write_bytes(OUTPUT)
            for field, value in (
                ("backend", "mlx"),
                ("mode", "edit"),
                ("overlay", "control"),
                ("load_shape", "deferred_materialization"),
            ):
                staged_payload = record("staged_residency", 14_000 << 20)
                staged_payload["key"][field] = value
                if field == "load_shape":
                    staged_payload["declared_calibration"]["load_shape"] = value
                    staged_payload["observed_calibration"]["load_shape"] = value
                resident = self.write_log(root, f"resident-{field}", resident_payload)
                staged = self.write_log(root, f"staged-{field}", staged_payload)
                with self.assertRaisesRegex(RuntimeError, "non-strategy A/B invariant"):
                    verify(
                        resident,
                        staged,
                        512,
                        "flux1_dev",
                        FINGERPRINT,
                        3,
                        MODEL_REVISION,
                        MODEL_INVENTORY_SHA256,
                        resident_output,
                        staged_output,
                        EXACT,
                    )

            resident = self.write_log(root, "resident-small", resident_payload)
            staged = self.write_log(
                root, "staged-small", record("staged_residency", 15_700 << 20)
            )
            with self.assertRaisesRegex(RuntimeError, "required at least"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )

            resident, staged, resident_output, staged_output = self.valid_pair(root)
            staged_output.write_bytes(b"different output")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 does not match"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )
            with self.assertRaisesRegex(RuntimeError, "non-negative"):
                verify(
                    resident,
                    staged,
                    -1,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )

    def tolerance_pair(
        self, root: Path, resident_bytes: bytes, staged_bytes: bytes
    ) -> tuple[Path, Path, Path, Path]:
        resident = self.write_log(
            root,
            "resident.log",
            record("resident", 24_000 << 20, output=resident_bytes, parity=TOLERANCE),
        )
        staged = self.write_log(
            root,
            "staged.log",
            record(
                "staged_residency", 15_000 << 20, output=staged_bytes, parity=TOLERANCE
            ),
        )
        resident_output = root / "resident.rgb"
        staged_output = root / "staged.rgb"
        resident_output.write_bytes(resident_bytes)
        staged_output.write_bytes(staged_bytes)
        return resident, staged, resident_output, staged_output

    def test_accepts_declared_tolerance_within_ceiling(self) -> None:
        # Mean drift 2.0 under the declared 4.0 ceiling: the sc-18149 adjudicated shape. The
        # measured drift is recomputed from the artifacts and returned for the result line.
        with tempfile.TemporaryDirectory() as temporary:
            resident, staged, resident_output, staged_output = self.tolerance_pair(
                Path(temporary), bytes([0] * 16), bytes([2] * 16)
            )
            self.assertEqual(
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    TOLERANCE,
                ),
                (24_000 << 20, 15_000 << 20, 2.0),
            )

    def test_rejects_tolerance_drift_above_the_declared_ceiling(self) -> None:
        # Mean drift 5.0 over the declared 4.0 ceiling: the mutation check — the recomputed metric,
        # not the records' own verdict, is what fails the lane.
        with tempfile.TemporaryDirectory() as temporary:
            resident, staged, resident_output, staged_output = self.tolerance_pair(
                Path(temporary), bytes([0] * 16), bytes([5] * 16)
            )
            with self.assertRaisesRegex(RuntimeError, "above the declared tolerance"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    TOLERANCE,
                )

    def test_rejects_tolerance_outputs_that_differ_in_length(self) -> None:
        # The mean over zipped bytes would silently truncate; the explicit length check refuses.
        with tempfile.TemporaryDirectory() as temporary:
            resident, staged, resident_output, staged_output = self.tolerance_pair(
                Path(temporary), bytes([0] * 16), bytes([0] * 12)
            )
            with self.assertRaisesRegex(RuntimeError, "differ in length"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    TOLERANCE,
                )

    def test_rejects_records_whose_contract_differs_from_the_lane_expectation(self) -> None:
        # The lane pins the contract; records cannot self-declare a looser (or different) one. Both
        # directions must refuse: exact records under a tolerance lane, tolerance records under an
        # exact lane, and a tolerance lane whose ceiling differs from the records'.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident, staged, resident_output, staged_output = self.valid_pair(root)
            with self.assertRaisesRegex(RuntimeError, "parity contract this lane declares"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    TOLERANCE,
                )
        with tempfile.TemporaryDirectory() as temporary:
            resident, staged, resident_output, staged_output = self.tolerance_pair(
                Path(temporary), OUTPUT, OUTPUT
            )
            for expectation in (
                EXACT,
                parse_expected_parity("tolerance:mean_abs_u8_subpixel:6.0"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "parity contract this lane declares"
                ):
                    verify(
                        resident,
                        staged,
                        512,
                        "flux1_dev",
                        FINGERPRINT,
                        3,
                        MODEL_REVISION,
                        MODEL_INVENTORY_SHA256,
                        resident_output,
                        staged_output,
                        expectation,
                    )

    def test_rejects_exact_expectation_when_outputs_differ(self) -> None:
        # Each record's SHA-256 binds to its own (differing) artifact, so the sha checks pass and
        # only the exact-parity comparison itself can refuse — it must.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = b"drifted parity output"
            resident = self.write_log(root, "resident.log", record("resident", 24_000 << 20))
            staged = self.write_log(
                root,
                "staged.log",
                record("staged_residency", 15_000 << 20, output=other),
            )
            resident_output = root / "resident.rgb"
            staged_output = root / "staged.rgb"
            resident_output.write_bytes(OUTPUT)
            staged_output.write_bytes(other)
            with self.assertRaisesRegex(RuntimeError, "violate exact parity"):
                verify(
                    resident,
                    staged,
                    512,
                    "flux1_dev",
                    FINGERPRINT,
                    3,
                    MODEL_REVISION,
                    MODEL_INVENTORY_SHA256,
                    resident_output,
                    staged_output,
                    EXACT,
                )

    def test_rejects_malformed_expected_parity(self) -> None:
        for value, message in (
            ("tolerance:not_a_metric:4.0", "not recomputable"),
            ("tolerance:mean_abs_u8_subpixel:x", "must be a number"),
            ("tolerance:mean_abs_u8_subpixel:-1", "finite and non-negative"),
            ("tolerance:mean_abs_u8_subpixel:inf", "finite and non-negative"),
            ("golden:fixture:m:1.0", "must be 'exact'"),
            ("", "must be 'exact'"),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                parse_expected_parity(value)

    def test_rejects_strategy_composition_and_parameter_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = record("staged_residency", 10)
            payload["key"]["engaged_composition"] = ["resident"]
            path = self.write_log(root, "composition", payload)
            with self.assertRaisesRegex(RuntimeError, "engaged_composition"):
                read_record(path, "staged_residency")

            payload = record("resident", 10)
            payload["key"]["parameters"]["decode_tile_edge"] = "512"
            path = self.write_log(root, "parameter", payload)
            with self.assertRaisesRegex(RuntimeError, "non-negative integer or null"):
                read_record(path, "resident")

            payload = record("staged_residency", 10)
            payload["key"]["parameters"]["decode_tile_edge"] = 512
            path = self.write_log(root, "resident-parameter", payload)
            with self.assertRaisesRegex(RuntimeError, "must be empty"):
                read_record(path, "staged_residency")


if __name__ == "__main__":
    unittest.main()
