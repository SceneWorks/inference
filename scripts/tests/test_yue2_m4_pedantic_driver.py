"""Source-neutral refusals for the one bounded M4 stage-2 diagnostic controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import yue2_m4_pedantic_driver as driver  # noqa: E402


class PedanticDriverTests(unittest.TestCase):
    def test_binary_must_be_exact_cargo_artifact_inside_declared_target(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": root}):
            target = Path(root)
            binary = target / "release" / "yue2-bf16-tile-diagnostic.exe"
            binary.parent.mkdir()
            binary.write_bytes(b"built-test-binary")
            log = target / "build.jsonl"
            row = {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic",
                   "kind": ["bin"]}, "package_id": "path+file:///" + (target / "source/harness").resolve().as_posix(),
                   "executable": str(binary)}
            stage = target / "source"
            core = {"reason": "compiler-artifact", "target": {"name": "candle_core"},
                    "features": ["cuda", "cudarc", "default"],
                    "package_id": "path+file:///" + (stage / "candle-overlay/candle-core").resolve().as_posix()}
            kernel = {"reason": "compiler-artifact", "target": {"name": "candle_kernels"},
                      "package_id": "path+file:///" +
                                    (stage / "engine-overlay/crates/media/candle-gen/vendor/candle-kernels").resolve().as_posix()}
            audio = {"reason": "compiler-artifact", "target": {"name": "candle_audio"},
                     "features": ["cuda", "default"], "package_id": "path+file:///" +
                     (stage / "engine-overlay/crates/audio/candle-audio").resolve().as_posix()}
            yue2 = {"reason": "compiler-artifact", "target": {"name": "candle_audio_yue2"},
                    "features": ["cuda", "default"], "package_id": "path+file:///" +
                    (stage / "engine-overlay/crates/audio/candle-audio-yue2").resolve().as_posix()}
            def write(*rows: dict) -> None:
                log.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
            write(row, core, kernel, audio, yue2)
            self.assertEqual(driver.verify_binary(log, binary, stage)["binary_sha256"], driver.sha256(binary))
            core["features"] = ["cuda", "cudarc", "default", "cudnn"]
            write(row, core, kernel, audio, yue2)
            with self.assertRaisesRegex(RuntimeError, "declared Candle"):
                driver.verify_binary(log, binary, stage)
            core["features"] = ["cuda", "cudarc", "default"]
            yue2["package_id"] = "path+file:///unreviewed/yue2"
            write(row, core, kernel, audio, yue2)
            with self.assertRaisesRegex(RuntimeError, "staged M4"):
                driver.verify_binary(log, binary, stage)
            yue2["package_id"] = "path+file:///" + \
                                  (stage / "engine-overlay/crates/audio/candle-audio-yue2").resolve().as_posix()
            row["executable"] = str(target / "release" / "another.exe")
            write(row, core, kernel, audio, yue2)
            with self.assertRaisesRegex(RuntimeError, "differs from the one Cargo"):
                driver.verify_binary(log, binary, stage)
            row["executable"] = str(binary)
            write(row, row, core, kernel, audio, yue2)
            with self.assertRaisesRegex(RuntimeError, "differs from the one Cargo"):
                driver.verify_binary(log, binary, stage)

    def test_teacher_inventory_rejects_injected_or_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for name in ("vae_real_reference.safetensors", "NONCOMMERCIAL.txt", "reference-provenance.json"):
                (directory / name).write_bytes(b"fixture")
            with mock.patch.object(driver, "verify_reference") as verifier:
                driver.verify_teacher(directory)
                verifier.assert_called_once()
                (directory / "extra.json").write_bytes(b"unsafe")
                with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                    driver.verify_teacher(directory)

    def test_deadline_requires_child_plus_tail_in_both_clocks(self) -> None:
        with mock.patch.object(driver.time, "time_ns", return_value=1_000_000_000_000), \
             mock.patch.object(driver, "require_remaining_window") as receipt:
            driver.remaining_window(1_000_000_000_000)
            receipt.assert_called_once_with(1200)
            with self.assertRaisesRegex(RuntimeError, "workflow timeout"):
                driver.remaining_window(1_000_000_000_000 - 601 * 1_000_000_000)
            receipt.side_effect = RuntimeError("reviewed owner window expired")
            with self.assertRaisesRegex(RuntimeError, "owner window expired"):
                driver.remaining_window(1_000_000_000_000)

    def test_wrong_parent_and_bad_selector_refuse_before_gpu_or_child(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(selector="pedantic_stage2", evidence=Path(root) / "new",
                                      engine_sha="a" * 40, control_sha="b" * 40)
            with mock.patch.object(driver, "PEDANTIC_ENGINE", driver.M4_SOURCE), \
                 mock.patch.object(driver, "OwnerGuard") as guard, \
                 mock.patch.object(driver.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "unreviewed M4 parent source"):
                    driver.execute(args)
                guard.assert_not_called()
                popen.assert_not_called()
            args.selector = "native_convt_math_mode"
            with self.assertRaisesRegex(RuntimeError, "unknown M4 diagnostic selector"):
                driver.execute(args)

    def test_busy_fresh_census_refuses_before_owner_guard_or_child(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.dict(os.environ, {"YUE2_DIAGNOSTIC_JOB_STARTED_UTC_NS": "1000"}), \
             mock.patch.object(driver, "verify_source"), \
             mock.patch.object(driver, "verify_teacher"), \
             mock.patch.object(driver, "verify_derivative", return_value={}), \
             mock.patch.object(driver, "verify_binary", return_value={}), \
             mock.patch.object(driver, "remaining_window"), \
             mock.patch.object(driver, "require_remaining_window", return_value=({}, Path(root))), \
             mock.patch.object(driver, "retain_reviewed_baseline", return_value=[]), \
             mock.patch.object(driver, "physical", return_value=("saved raw", ["GPU0 busy"])), \
             mock.patch.object(driver, "OwnerGuard") as guard, \
             mock.patch.object(driver.subprocess, "Popen") as popen:
            args = argparse.Namespace(selector="pedantic_stage2", evidence=Path(root) / "evidence",
                                      engine_sha="a" * 40, control_sha="b" * 40, reference=Path(root),
                                      source_provenance=Path(root) / "m4-pedantic-source.json",
                                      binary=Path(root) / "binary", build_json=Path(root) / "build.jsonl")
            with self.assertRaisesRegex(RuntimeError, "GPU0 busy"):
                driver.execute(args)
            guard.assert_not_called()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
