"""No accelerator execution: guard regressions for the bounded tile diagnostic."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1] / "ci"
sys.path.insert(0, str(CI))
spec = importlib.util.spec_from_file_location("yue2_bf16_tile_diagnostic", CI / "yue2_bf16_tile_diagnostic.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticGuards(unittest.TestCase):
    def test_foreign_context_never_launches_a_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.write_bytes(b"not executed")
            args = argparse.Namespace(binary=binary, reference=root / "reference", evidence=root / "evidence",
                                      engine_sha=diag.ENGINE_SHA, control_sha="a" * 40)
            with patch.dict(os.environ, RUNNER_NAME="cuda-windows", CUDA_VISIBLE_DEVICES="0"), \
                 patch.object(diag, "verify_revisions"), patch.object(diag, "verify_reference"), \
                 patch.object(diag.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")), \
                 patch.object(diag, "cuda_census", return_value=("foreign PID", ["foreign PID"])), \
                 patch.object(diag.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "foreign accelerator ownership"):
                    diag.execute(args)
                launch.assert_not_called()
            self.assertEqual((args.evidence / "census-before.txt").read_text(encoding="utf-8"), "foreign PID")

    def test_other_source_cannot_run_under_the_failed_source_identity(self):
        with patch.object(diag, "verify_revisions") as verify:
            with self.assertRaisesRegex(RuntimeError, "exact failed M3"):
                diag.execute(argparse.Namespace(engine_sha="b" * 40))
            verify.assert_not_called()

    def test_staging_cannot_dirty_stationary_engine_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            template = root / "template"
            template.mkdir()
            args = argparse.Namespace(engine_sha=diag.ENGINE_SHA, control_sha="a" * 40,
                                      template=template, destination=engine / "harness")
            old = Path.cwd()
            try:
                os.chdir(engine)
                with patch.object(diag, "verify_revisions"):
                    with self.assertRaisesRegex(RuntimeError, "outside stationary checkouts"):
                        diag.prepare_harness(args)
                self.assertFalse(args.destination.exists())
            finally:
                os.chdir(old)

    def test_dependency_drift_refuses_before_building_an_alternate_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            (engine / "Cargo.lock").write_text('[[package]]\nname="candle-core"\nversion="0.10.2"\n', encoding="utf-8")
            template = root / "template"
            template.mkdir()
            (template / "Cargo.lock.snapshot").write_text('[[package]]\nname="candle-core"\nversion="0.10.3"\n', encoding="utf-8")
            args = argparse.Namespace(engine_sha=diag.ENGINE_SHA, control_sha="a" * 40,
                                      template=template, destination=root / "outside" / "harness")
            old = Path.cwd()
            try:
                os.chdir(engine)
                with patch.object(diag, "verify_revisions"):
                    with self.assertRaisesRegex(RuntimeError, "dependencies differ"):
                        diag.prepare_harness(args)
                self.assertFalse(args.destination.exists())
            finally:
                os.chdir(old)

    def test_ambiguous_build_output_cannot_select_an_unrelated_binary(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.touch()
            row = {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic"},
                   "executable": str(binary)}
            build = root / "build.jsonl"
            build.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "one diagnostic executable"):
                diag.resolve_binary(build, root / "binary.txt")
            self.assertFalse((root / "binary.txt").exists())

    def test_different_cuda_implementation_cannot_reach_execution(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "diagnostic.exe"
            binary.touch()
            target = {"reason": "compiler-artifact", "target": {"name": "yue2-bf16-tile-diagnostic"},
                      "executable": str(binary)}
            core = {"reason": "compiler-artifact", "target": {"name": "candle_core"},
                    "features": ["cuda", "cudarc", "default", "cudnn"]}
            kernel = {"reason": "compiler-artifact", "target": {"name": "candle_kernels"},
                      "package_id": "path+file:///D:/actions/engine/crates/media/candle-gen/vendor/candle-kernels#0.10.2"}
            build = root / "build.jsonl"
            build.write_text("".join(json.dumps(row) + "\n" for row in (target, core, kernel)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Candle features differ"):
                diag.resolve_binary(build, root / "binary.txt")
            core["features"].remove("cudnn")
            kernel["package_id"] = "git+https://github.com/huggingface/candle#candle-kernels@0.10.2"
            build.write_text("".join(json.dumps(row) + "\n" for row in (target, core, kernel)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "vendored CUDA kernels"):
                diag.resolve_binary(build, root / "binary.txt")
            self.assertFalse((root / "binary.txt").exists())


if __name__ == "__main__":
    unittest.main()
