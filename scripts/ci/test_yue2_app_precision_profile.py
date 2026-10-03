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
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).with_name("yue2_app_precision_profile.py")
spec = importlib.util.spec_from_file_location("yue2_app_precision_profile", MODULE_PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class PrecisionControlTests(unittest.TestCase):
    def test_eight_cuda_cases_and_seven_metal_cases_keep_exact_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            templates = MODULE_PATH.parent / "yue2-app-precision-cases"
            for backend, count in (("cuda", 8), ("metal", 7)):
                destination = root / f"{backend}-cases"
                prepared = control.prepare_cases(templates, destination, backend)
                self.assertEqual(len(prepared["cases"]), count)
                self.assertEqual([row["name"] for row in prepared["cases"]],
                                 list(control.names_for_backend(backend)))
                for name in control.names_for_backend(backend):
                    original = json.loads((templates / f"{name}.json").read_text(encoding="utf-8"))
                    copied = json.loads((destination / f"{name}.json").read_text(encoding="utf-8"))
                    self.assertEqual(copied.pop("id"), original.pop("id").replace(":cuda:", f":{backend}:"))
                    self.assertEqual(copied, original)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    control.prepare_cases(templates, destination, backend)
            self.assertFalse((root / "metal-cases" / "experimental-fp8-auto.json").exists())

    def test_changed_fixed_source_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for changed_name in ("strict-bf16-standard", "experimental-fp8-auto"):
                templates = root / changed_name
                templates.mkdir()
                for name in control.NAMES:
                    source = MODULE_PATH.parent / "yue2-app-precision-cases" / f"{name}.json"
                    (templates / f"{name}.json").write_bytes(source.read_bytes())
                target = templates / f"{changed_name}.json"
                target.write_text(target.read_text(encoding="utf-8").replace("831004", "831005"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    control.prepare_cases(templates, root / f"cases-{changed_name}", "cuda")

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
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "bfloat16",
                            "engineVaeCudaBf16MathPolicy": "disallow_reduced_precision_reduction_v1"},
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

    def test_effective_cuda_bf16_vae_math_policy_is_required_only_for_cuda_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            for backend, name in (("cuda", "strict-bf16-standard"),
                                  ("cuda", "strict-fp32-standard"),
                                  ("metal", "strict-bf16-standard")):
                _, case_name, decoder, policy, model_dtype, vae_dtype = control.CASES[name]
                body = {
                    "caseId": control.case_id(backend, name), "backend": backend,
                    "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae"}},
                    "request": {"name": case_name, "computePolicy": policy},
                    "admission": {"outcome": "admitted"},
                    "outcome": {"status": "completed", "engineComputePolicy": policy,
                                "engineModelDtype": model_dtype, "engineVaeDtype": vae_dtype},
                    "measured": {"peakBytes": 1024, "stages": {
                        stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
                }
                wanted = "disallow_reduced_precision_reduction_v1"
                if backend == "cuda" and policy == "bf16":
                    record.write_text(json.dumps(body), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                        control.verify_record(record, backend, name)
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = "stale"
                    record.write_text(json.dumps(body), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                        control.verify_record(record, backend, name)
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = wanted
                    record.write_text(json.dumps(body), encoding="utf-8")
                    self.assertEqual(
                        control.verify_record(record, backend, name)["effective_vae_cuda_bf16_math_policy"],
                        wanted,
                    )
                else:
                    record.write_text(json.dumps(body), encoding="utf-8")
                    self.assertIsNone(
                        control.verify_record(record, backend, name)["effective_vae_cuda_bf16_math_policy"]
                    )
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = wanted
                    record.write_text(json.dumps(body), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "present on another backend"):
                        control.verify_record(record, backend, name)

    def test_fp8_auto_record_requires_actual_fp8_and_retained_host_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            name = "experimental-fp8-auto"
            body = {
                "caseId": control.case_id("cuda", name), "backend": "cuda",
                "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae"}},
                "request": {"name": name, "computePolicy": "auto", "arMode": "experimentalFp8"},
                "admission": {"outcome": "admitted", "estimate": {"weights": {"hostBytes": 2 * 1024 ** 3}}},
                "outcome": {"status": "completed", "engineComputePolicy": "auto",
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "float32",
                            "engineQuantization": "fp8"},
                "measured": {"peakBytes": 1024, "stages": {
                    stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
            }
            def check():
                record.write_text(json.dumps(body), encoding="utf-8")
                return control.verify_record(record, "cuda", name)
            self.assertEqual(check()["effective_compute_policy"], "auto")
            self.assertEqual(check()["effective_ar_quantization"], "fp8")
            self.assertEqual(check()["host_original_bytes"], 2 * 1024 ** 3)
            for field, value, error in (
                ("arMode", "native", "record AR mode"),
                ("computePolicy", "bf16", "record request name/policy"),
            ):
                original = body["request"][field]
                body["request"][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    check()
                body["request"][field] = original
            for field, value, error in (
                ("engineQuantization", "bf16", "experimental FP8 AR did not execute"),
                ("engineModelDtype", "float32", "effective engineModelDtype"),
                ("engineVaeDtype", "bfloat16", "effective engineVaeDtype"),
                ("engineVaeCudaBf16MathPolicy", "disallow_reduced_precision_reduction_v1",
                 "present on another backend"),
            ):
                body["outcome"][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    check()
                if field == "engineVaeCudaBf16MathPolicy":
                    del body["outcome"][field]
                else:
                    body["outcome"][field] = "fp8" if field == "engineQuantization" else (
                        "bfloat16" if field == "engineModelDtype" else "float32")
            body["admission"]["estimate"]["weights"]["hostBytes"] -= 1
            with self.assertRaisesRegex(ValueError, "retained BF16 AR originals"):
                check()
            with self.assertRaisesRegex(ValueError, "unknown case/backend"):
                control.verify_record(record, "metal", name)

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
            census = types.SimpleNamespace(cuda_physical_census=lambda: ("typed pmon rows", ["123 C worker"]),
                                           metal_census=lambda: ("", []),
                                           physical_busy_message=lambda raw, busy, context: f"{context}: {busy}",
                                           retain_cuda_physical_evidence=lambda *_: [],
                                           retain_reviewed_baseline=lambda *_: [])
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 12)):
                with self.assertRaisesRegex(ValueError, "competing physical-device"):
                    control.preflight("cuda", evidence, "before-test")
            record = json.loads((evidence / "preflight-before-test.json").read_text(encoding="utf-8"))
            self.assertFalse(record["admitted"])
            self.assertEqual(record["competing_processes"], ["123 C worker"])

    def test_cuda_preflight_surfaces_original_physical_refusal(self):
        proof_spec = importlib.util.spec_from_file_location(
            "physical_message_control", MODULE_PATH.with_name("yue2_precision_proof.py"))
        proof = importlib.util.module_from_spec(proof_spec)
        proof_spec.loader.exec_module(proof)
        raw = json.dumps({"commandExit": 0, "refusal": "adapterDedicated rose over reviewed baseline"})
        busy = ["0 38212 C+G - - ChatGPT.exe"]
        census = types.SimpleNamespace(cuda_physical_census=lambda: (raw, busy),
                    metal_census=Mock(), physical_busy_message=proof.physical_busy_message,
                    retain_cuda_physical_evidence=Mock(), retain_reviewed_baseline=Mock())
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 12)):
                with self.assertRaisesRegex(ValueError, "adapterDedicated rose over reviewed baseline"):
                    control.preflight("cuda", evidence, "before-test")
            record = json.loads((evidence / "preflight-before-test.json").read_text(encoding="utf-8"))
            self.assertFalse(record["admitted"])
            self.assertEqual(record["competing_processes"], busy)
            self.assertEqual(record["census"], raw)
            census.retain_cuda_physical_evidence.assert_not_called()
            census.metal_census.assert_not_called()

    def test_cuda_initial_preflight_refuses_short_owner_window_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            census = types.SimpleNamespace(cuda_physical_census=Mock(), metal_census=Mock(),
                                           physical_busy_message=Mock(),
                                           retain_cuda_physical_evidence=Mock(),
                                           retain_reviewed_baseline=Mock())
            guard = types.SimpleNamespace(require_remaining_window=Mock(
                side_effect=RuntimeError("reviewed owner window cannot cover bounded run and postflight")))
            with patch.dict("sys.modules", {"yue2_precision_proof": census,
                                            "yue2_cuda_idle_context": guard}):
                with self.assertRaisesRegex(RuntimeError, "cannot cover"):
                    control.preflight("cuda", Path(directory) / "evidence", "initial")
            guard.require_remaining_window.assert_called_once_with(480 * 60 + 600)
            census.cuda_physical_census.assert_not_called()

    def test_eight_cuda_case_verdict_requires_final_physical_release_census(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases"
            control.prepare_cases(MODULE_PATH.parent / "yue2-app-precision-cases", cases, "cuda")
            labels = []
            def preflight(_backend, _evidence, label):
                labels.append(label)
                if label == "after-cases":
                    raise ValueError("post-case physical owner changed")
            with patch.object(control, "preflight", side_effect=preflight), \
                 patch.object(control.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
                 patch.object(control, "verify_record"), \
                 patch.object(control, "verify_audio", return_value={"sha256": "a" * 64}), \
                 patch.object(control, "collect") as collect:
                with self.assertRaisesRegex(ValueError, "post-case physical owner changed"):
                    control.run_captures(root, root, root / "data", root / "profile",
                                         root / "evidence", cases, "cuda")
                collect.assert_not_called()
            self.assertEqual(len([label for label in labels if label.startswith("before-")]), 8)
            self.assertEqual(labels[-1], "after-cases")
            self.assertTrue((root / "evidence" / "cases-manifest.json").is_file())

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
            for name in control.names_for_backend("cuda"):
                _, case_name, decoder, policy, model_dtype, vae_dtype = control.CASES[name]
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
                if policy == "bf16":
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = "disallow_reduced_precision_reduction_v1"
                if name == "experimental-fp8-auto":
                    body["request"]["arMode"] = "experimentalFp8"
                    body["outcome"]["engineQuantization"] = "fp8"
                    body["admission"]["estimate"] = {"weights": {"hostBytes": 2 * 1024 ** 3}}
                (run.parent / "record.json").write_text(json.dumps(body), encoding="utf-8")
            verdict = control.collect(profile, evidence, "cuda")
            self.assertEqual(len(verdict["listening_audio"]), 8)
            self.assertEqual(len(json.loads((evidence / "audio-inventory.json").read_text(encoding="utf-8"))["cases"]), 8)
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
