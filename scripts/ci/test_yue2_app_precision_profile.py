"""CPU-only contract tests for the bounded remote app precision control."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("yue2_app_precision_profile.py")
spec = importlib.util.spec_from_file_location("yue2_app_precision_profile", MODULE_PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class PrecisionControlTests(unittest.TestCase):
    def test_seven_fixed_cases_copy_outside_repo_and_only_change_backend_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            templates = MODULE_PATH.parent / "yue2-app-precision-cases"
            prepared = control.prepare_cases(templates, root / "metal-cases", "metal")
            self.assertEqual(len(prepared["cases"]), 7)
            for name in control.NAMES:
                original = json.loads((templates / f"{name}.json").read_text())
                copied = json.loads((root / "metal-cases" / f"{name}.json").read_text())
                self.assertEqual(copied.pop("id"), original.pop("id").replace(":cuda:", ":metal:"))
                self.assertEqual(copied, original)
            with self.assertRaisesRegex(ValueError, "already exists"):
                control.prepare_cases(templates, root / "metal-cases", "metal")

    def test_changed_fixed_source_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("templates").mkdir()
            for name in control.NAMES:
                source = MODULE_PATH.parent / "yue2-app-precision-cases" / f"{name}.json"
                root.joinpath("templates", f"{name}.json").write_bytes(source.read_bytes())
            target = root / "templates" / "strict-bf16-standard.json"
            target.write_text(target.read_text().replace("831004", "831005"))
            with self.assertRaisesRegex(ValueError, "changed"):
                control.prepare_cases(root / "templates", root / "cases", "cuda")

    def test_effective_dtype_and_decoder_are_proven_from_record(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            stages = {name: {"peakBytes": 1024, "samples": 1} for name in control.STAGES}
            body = {
                "caseId": "yue2:bf16:cuda:strict-bf16-legacy", "backend": "cuda",
                "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae-legacy"}},
                "request": {"name": "strict-bf16-legacy", "computePolicy": "bf16"},
                "admission": {"outcome": "admitted"},
                "outcome": {"status": "completed", "engineComputePolicy": "bf16",
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "bfloat16"},
                "measured": {"peakBytes": 1024, "stages": stages},
            }
            record.write_text(json.dumps(body))
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-legacy")["effective_vae_dtype"], "bfloat16")
            body["outcome"]["engineVaeDtype"] = "float32"
            record.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "effective engineVaeDtype"):
                control.verify_record(record, "cuda", "strict-bf16-legacy")
            body["outcome"]["engineVaeDtype"] = "bfloat16"
            body["identity"]["decoder"]["repo"] = "m-a-p/YuE2-Vae"
            record.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "decoder identity"):
                control.verify_record(record, "cuda", "strict-bf16-legacy")

    def test_exact_app_pin_and_clean_sources_required(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            engine = Path(directory) / "engine"
            app.mkdir()
            engine.mkdir()
            app_sha, engine_sha = "a" * 40, "b" * 40
            (app / "Cargo.toml").write_text(
                f'candle-kernels = {{ git = "https://github.com/SceneWorks/inference", rev = "{engine_sha}" }}\n'
            )
            with patch.object(control, "git", side_effect=lambda root, *args:
                              (app_sha if root == app else engine_sha) if args[0] == "rev-parse" else ""):
                self.assertEqual(control.verify_sources(app, engine, app_sha, engine_sha)["app_pins"], [engine_sha])
                (app / "Cargo.toml").write_text((app / "Cargo.toml").read_text().replace(engine_sha, "c" * 40))
                with self.assertRaisesRegex(ValueError, "pin"):
                    control.verify_sources(app, engine, app_sha, engine_sha)

    def test_busy_cuda_preflight_refuses_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            census = types.SimpleNamespace(cuda_census=lambda: ("typed pmon rows", ["123 C worker"]),
                                           metal_census=lambda: ("", []))
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 12)):
                with self.assertRaisesRegex(ValueError, "competing physical-device"):
                    control.preflight("cuda", evidence, "before-test")
            record = json.loads((evidence / "preflight-before-test.json").read_text())
            self.assertFalse(record["admitted"])
            self.assertEqual(record["competing_processes"], ["123 C worker"])


if __name__ == "__main__":
    unittest.main()
