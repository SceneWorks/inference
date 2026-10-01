"""No accelerator execution: guard regressions for the bounded tile diagnostic."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1] / "ci"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/yue2-bf16-tile-diagnostic.yml"
sys.path.insert(0, str(CI))
spec = importlib.util.spec_from_file_location("yue2_bf16_tile_diagnostic", CI / "yue2_bf16_tile_diagnostic.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticGuards(unittest.TestCase):
    def test_explicit_selector_keeps_waveform_default_and_forwards_to_child(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("default: waveform", workflow)
        self.assertIn("options: [waveform, first_conv]", workflow)
        self.assertIn('run --diagnostic "$env:YUE2_DIAGNOSTIC_SELECTOR"', workflow)
        self.assertEqual(diag.DIAGNOSTICS, ("waveform", "first_conv"))

    def test_first_conv_f32_json_scalars_compare_by_actual_tensor_bits(self):
        exact = struct.unpack("<f", struct.pack("<f", 1.2345679))[0]
        expected = {"maxAbs": exact, "differentValues": 1, "comparedValues": 16,
                    "firstDifferent": {"channel": 2, "globalLatentFrame": 17,
                                       "fullValue": exact, "tileValue": 0.0, "absError": exact}}
        rounded_json = json.loads(json.dumps(expected).replace(str(exact), "1.2345679"))
        self.assertNotEqual(expected, rounded_json)
        self.assertTrue(diag.same_first_conv_comparison(rounded_json, expected))
        rounded_json["firstDifferent"]["globalLatentFrame"] += 1
        self.assertFalse(diag.same_first_conv_comparison(rounded_json, expected))

    def test_first_conv_report_recomputes_raw_residuals_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            fixture = root / "crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json"
            fixture.parent.mkdir(parents=True)
            standard = {"repo": "m-a-p/YuE2-Vae",
                        "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
                        "weights_sha256": diag.DECODER_SHA256["standard"],
                        "config_sha256": "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"}
            meta = {"decoders": {"standard": standard}}
            fixture.write_text(json.dumps(meta), encoding="utf-8")

            def array(name, dtype, channels, frames):
                raw = struct.pack("<" + "f" * (channels * frames), *([0.0] * (channels * frames)))
                path = data / f"{name}.f32le"
                path.write_bytes(raw)
                return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "layout": "bct_f32le",
                        "shape": [1, channels, frames], "dtype": dtype}

            runs = {}
            for label, dtype in (("bf16", "BF16"), ("f32", "F32")):
                full = {stage: array(f"{label}-full-{stage}", dtype, channels, 75)
                        for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                windows = []
                for index, start in enumerate(range(0, 75, 16)):
                    end = min(start + 16, 75)
                    left, right = max(0, start - 16), min(75, end + 16)
                    captures = {stage: array(f"{label}-tile-{index}-{stage}", dtype, channels, right - left)
                                for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                    comparisons = {stage: {"maxAbs": 0.0, "differentValues": 0, "firstDifferent": None,
                                           "comparedValues": channels * (end - start)}
                                   for stage, channels in (("input", 64), ("preBias", 1), ("postBias", 1))}
                    windows.append({"start": start, "end": end, "left": left, "right": right,
                                    "coreLength": end - start, "captures": captures,
                                    "alignedCore": comparisons})
                runs[label] = {"resident": {"sourceDtype": "F32", "foldDtype": "F32",
                                            "residentDtype": dtype, "weightShape": [1, 64, 7],
                                            "biasShape": [1], "weightF32LeSha256": "a" * 64,
                                            "biasF32LeSha256": "b" * 64},
                               "full": full, "windows": windows}
            report = {"schemaVersion": 2, "selector": "first_conv",
                      "purpose": "diagnostic_only_no_gate_change", "engineSha": diag.ENGINE_SHA,
                      "referenceSha256": diag.REFERENCE_SHA256,
                      "referenceMetadataSha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                      "latentIdentity": {"sha256": diag.LATENT_SHA256, "shape": [75, 64],
                                         "source": {"stage_identity": "precision_reference:long_latent"}},
                      "backend": "cuda", "deviceOrdinal": 0, "decoderIdentity": standard,
                      "frames": 75, "coreFrames": 16, "haloFrames": 16,
                      "operator": {"name": "decoder.layers.0.Conv1d", "kernel": 7,
                                   "padding": 3, "stride": 1, "dilation": 1, "groups": 1},
                      "originalWaveformObservation": {"runId": "36884387320", "clampedMaxAbs": 0.03125,
                                                      "originalBound": 1 / 64,
                                                      "interpretation": "prior_failed_waveform_proof_not_a_first_conv_gate"},
                      "runs": runs}
            report_path = data / "report.json"
            old = Path.cwd()
            try:
                os.chdir(root)
                report_path.write_text(json.dumps(report), encoding="utf-8")
                diag.verify_first_conv_data(data, meta)
                window = report["runs"]["bf16"]["windows"][0]
                tile = window["captures"]["preBias"]
                tile_path = data / tile["file"]
                raw = bytearray(tile_path.read_bytes())
                raw[4 * 4:5 * 4] = struct.pack("<f", 0.03125)
                tile_path.write_bytes(raw)
                tile["sha256"] = hashlib.sha256(raw).hexdigest()
                window["alignedCore"]["preBias"] = {
                    "maxAbs": 0.03125, "differentValues": 1, "comparedValues": 16,
                    "firstDifferent": {"channel": 0, "globalLatentFrame": 4,
                                       "fullValue": 0.0, "tileValue": 0.03125, "absError": 0.03125}}
                report_path.write_text(json.dumps(report), encoding="utf-8")
                diag.verify_first_conv_data(data, meta)
                window["alignedCore"]["preBias"]["differentValues"] = 0
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "residual does not match"):
                    diag.verify_first_conv_data(data, meta)
                window["alignedCore"]["preBias"]["differentValues"] = 1
                report_path.write_text(json.dumps(report), encoding="utf-8")
                (data / runs["bf16"]["full"]["preBias"]["file"]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "size/hash mismatch"):
                    diag.verify_first_conv_data(data, meta)
            finally:
                os.chdir(old)

    @staticmethod
    def assert_locked_fetch_before_offline_build(workflow: str) -> None:
        step = workflow.split("      - name: Build the external M3 VAE diagnostic harness\n", 1)[1].split(
            "      - name: Run only the bounded same-input VAE diagnostic\n", 1)[0]
        lines = [line.strip() for line in step.splitlines()]
        prepare = next(i for i, line in enumerate(lines) if "prepare-harness" in line)
        fetch = next(i for i, line in enumerate(lines) if line.startswith("cargo fetch "))
        build = next(i for i, line in enumerate(lines) if line.startswith("cargo build "))
        assert prepare < fetch < build
        assert lines[fetch] == (
            'cargo fetch --locked --manifest-path "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\harness\\Cargo.toml" '
            '--target x86_64-pc-windows-msvc > "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log" 2>&1')
        assert lines[fetch + 1] == (
            'if errorlevel 1 (type "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log"& exit /b 1)')
        assert lines[build] == (
            'cargo build --locked --offline --release --manifest-path '
            '"%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\harness\\Cargo.toml" --features cuda '
            '--message-format=json > "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\build.jsonl" '
            '2> "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\build.log"')
        assert 'path: ${{ runner.temp }}/yue2-bf16-tile-diagnostic' in workflow

    def test_locked_target_fetch_stages_before_unchanged_offline_build(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assert_locked_fetch_before_offline_build(workflow)
        mutations = {
            "missing fetch": ('          cargo fetch --locked', '          cargo missing --locked'),
            "unlocked fetch": ('cargo fetch --locked', 'cargo fetch'),
            "wrong target": ('--target x86_64-pc-windows-msvc', '--target x86_64-unknown-linux-gnu'),
            "ignored fetch error": ('if errorlevel 1 (type "%RUNNER_TEMP%\\yue2-bf16-tile-diagnostic\\fetch.log"& exit /b 1)', 'rem ignored fetch error'),
            "online build": ('cargo build --locked --offline', 'cargo build --locked'),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                self.assertIn(old, workflow)
                with self.assertRaises((AssertionError, StopIteration)):
                    self.assert_locked_fetch_before_offline_build(workflow.replace(old, new, 1))
        fetch_line = next(line for line in workflow.splitlines() if "cargo fetch --locked" in line)
        build_line = next(line for line in workflow.splitlines() if "cargo build --locked --offline" in line)
        moved = workflow.replace(fetch_line + "\n", "", 1).replace(build_line, build_line + "\n" + fetch_line, 1)
        with self.assertRaises(AssertionError):
            self.assert_locked_fetch_before_offline_build(moved)

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
