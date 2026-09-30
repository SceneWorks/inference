"""CPU-only contract tests for the bounded remote app precision control."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
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
                original = json.loads((templates / f"{name}.json").read_text(encoding="utf-8"))
                copied = json.loads((root / "metal-cases" / f"{name}.json").read_text(encoding="utf-8"))
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
            target.write_text(target.read_text(encoding="utf-8").replace("831004", "831005"), encoding="utf-8")
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
            record.write_text(json.dumps(body), encoding="utf-8")
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-legacy")["effective_vae_dtype"], "bfloat16")
            body["outcome"]["engineVaeDtype"] = "float32"
            record.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective engineVaeDtype"):
                control.verify_record(record, "cuda", "strict-bf16-legacy")
            body["outcome"]["engineVaeDtype"] = "bfloat16"
            body["identity"]["decoder"]["repo"] = "m-a-p/YuE2-Vae"
            record.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decoder identity"):
                control.verify_record(record, "cuda", "strict-bf16-legacy")

    def test_exact_app_pin_and_clean_sources_required(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            engine = Path(directory) / "engine"
            source_control = Path(directory) / "control"
            app.mkdir()
            engine.mkdir()
            source_control.mkdir()
            app_sha, engine_sha, control_sha = "a" * 40, "b" * 40, "c" * 40
            (app / "Cargo.toml").write_text(
                f'candle-kernels = {{ git = "https://github.com/SceneWorks/inference", rev = "{engine_sha}" }}\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"GITHUB_SHA": control_sha}), \
                 patch.object(control, "git", side_effect=lambda root, *args:
                              {app: app_sha, engine: engine_sha, source_control: control_sha}[root]
                              if args[0] == "rev-parse" else "") as git_mock:
                self.assertEqual(control.verify_sources(app, engine, source_control, app_sha, engine_sha,
                                                       control_sha)["control_sha"], control_sha)
                with self.assertRaisesRegex(ValueError, "workflow control SHA"):
                    control.verify_sources(app, engine, source_control, app_sha, engine_sha, "d" * 40)
                git_mock.side_effect = lambda root, *args: (
                    {app: app_sha, engine: "d" * 40, source_control: control_sha}[root]
                    if args[0] == "rev-parse" else "")
                with self.assertRaisesRegex(ValueError, "engine checkout SHA"):
                    control.verify_sources(app, engine, source_control, app_sha, engine_sha, control_sha)
                git_mock.side_effect = lambda root, *args: (
                    {app: app_sha, engine: engine_sha, source_control: "d" * 40}[root]
                    if args[0] == "rev-parse" else "")
                with self.assertRaisesRegex(ValueError, "control checkout SHA"):
                    control.verify_sources(app, engine, source_control, app_sha, engine_sha, control_sha)
                git_mock.side_effect = lambda root, *args: (
                    {app: app_sha, engine: engine_sha, source_control: control_sha}[root]
                    if args[0] == "rev-parse" else "")
                (app / "Cargo.toml").write_text((app / "Cargo.toml").read_text(encoding="utf-8").replace(engine_sha, "c" * 40), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pin"):
                    control.verify_sources(app, engine, source_control, app_sha, engine_sha, control_sha)

    def test_busy_cuda_preflight_refuses_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            census = types.SimpleNamespace(cuda_census=lambda: ("typed pmon rows", ["123 C worker"]),
                                           metal_census=lambda: ("", []))
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 12)):
                with self.assertRaisesRegex(ValueError, "competing physical-device"):
                    control.preflight("cuda", evidence, "before-test")
            record = json.loads((evidence / "preflight-before-test.json").read_text(encoding="utf-8"))
            self.assertFalse(record["admitted"])
            self.assertEqual(record["competing_processes"], ["123 C worker"])

    def test_listening_wav_inventory_is_run_owned_and_stream_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            run = profile / control.case_id("metal", "strict-bf16-standard").replace(":", "__") / "run"
            run.mkdir(parents=True)
            audio = run / "audio.wav"
            wav = b"RIFF" + (48).to_bytes(4, "little") + b"WAVE" + b"\0" * 48
            audio.write_bytes(wav)
            row = control.verify_audio(profile, "metal", "strict-bf16-standard")
            self.assertEqual(row["path"], str(audio.resolve()))
            self.assertEqual(row["size_bytes"], len(wav))
            self.assertEqual(row["sha256"], hashlib.sha256(wav).hexdigest())
            audio.write_bytes(b"not-a-wave" + b"\0" * 50)
            with self.assertRaisesRegex(ValueError, "WAV header"):
                control.verify_audio(profile, "metal", "strict-bf16-standard")
            audio.unlink()
            outside = root / "outside.wav"
            outside.write_bytes(wav)
            audio.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escaped"):
                control.verify_audio(profile, "metal", "strict-bf16-standard")

    def test_collected_verdict_inventories_all_audio_without_copying_wavs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            evidence = root / "evidence"
            wav = b"RIFF" + (48).to_bytes(4, "little") + b"WAVE" + b"\0" * 48
            for name, (_, case_name, decoder, policy, model_dtype, vae_dtype) in control.CASES.items():
                run = profile / control.case_id("cuda", name).replace(":", "__") / "run"
                run.mkdir(parents=True)
                (run / "audio.wav").write_bytes(wav)
                body = {
                    "caseId": control.case_id("cuda", name), "backend": "cuda",
                    "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae" +
                                              ("-legacy" if decoder == "legacy" else "")}},
                    "request": {"name": case_name, "computePolicy": policy},
                    "admission": {"outcome": "admitted"},
                    "outcome": {"status": "completed", "engineComputePolicy": policy,
                                "engineModelDtype": model_dtype, "engineVaeDtype": vae_dtype},
                    "measured": {"peakBytes": 1024, "stages": {
                        stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
                }
                (run.parent / "record.json").write_text(json.dumps(body), encoding="utf-8")
            verdict = control.collect(profile, evidence, "cuda")
            self.assertEqual(len(verdict["listening_audio"]), 7)
            self.assertEqual(len(json.loads((evidence / "audio-inventory.json").read_text(encoding="utf-8"))["cases"]), 7)
            self.assertEqual(list(evidence.rglob("*.wav")), [])

    def test_wav_artifacts_use_only_fresh_run_owned_profile_glob(self):
        workflow = (MODULE_PATH.parents[2] / ".github/workflows/yue2-app-precision-profile.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("path: ${{ env.APP_RUN_ROOT }}/profile/**/run/audio.wav"), 2)
        self.assertEqual(workflow.count("if: always() && steps.run-root.outcome == 'success'"), 4)
        self.assertEqual(workflow.count("id: run-root"), 2)
        self.assertNotIn("${{ env.APP_EVIDENCE }}/**/audio.wav", workflow)
        self.assertIn("expected_control_sha:", workflow)
        self.assertIn("--control control --app-sha", workflow)
        self.assertLess(workflow.index("Select the app checkout's pinned Rust channel"),
                        workflow.index("uses: ./app/.github/actions/prepare-rust-runner"))
        self.assertIn("RUSTUP_TOOLCHAIN=", workflow)
        self.assertIn("Verify selected app Rust channel", workflow)


if __name__ == "__main__":
    unittest.main()
