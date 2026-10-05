"""CPU-only contract tests for the bounded remote app precision control."""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
import importlib.util
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).with_name("yue2_app_precision_profile.py")
spec = importlib.util.spec_from_file_location("yue2_app_precision_profile", MODULE_PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
M4_SHA = "825341ff8d0110ea448213485891b39d57806fa4"
M4_POLICY = control.SUPPORTED_RUNTIME_POLICIES[M4_SHA]
M5_SHA = "190e20e7c6b5bac006194c729baabd22c0c44a5d"
M5_POLICY = "fixed_order_bf16_convolution_v1"


def estimated_stages() -> dict:
    return {stage: {"deviceBytes": 1024} for stage in control.STAGES}


def measured_scope(backend: str) -> dict:
    if backend != "cuda":
        return {}
    luid = "luid_0x00000000_0x0001f78f"
    proof = "a" * 64
    return {"scope": "selected-device-global", "sampler": "nvidia-smi memory.used",
            "deviceProof": {"physicalIndex": 1, "cudaOrdinal": 0,
                            "uuid": "GPU-e4b79931-7be6-f216-460a-f5405cfafffe",
                            "pci": "00000000:C1:00.0", "luid": luid, "sha256": proof},
            "ownedFaults": [], "owned": {"complete": True,
                "sampler": "windows-gpu-process-memory dedicated", "selectedLuid": luid,
                "proofSha256": proof, "journalSha256": "b" * 64, "faults": [], "peakBytes": 1,
                "process": {"pid": 42, "parentPid": 41, "createdUtc": "2026-10-04T20:00:00Z",
                            "executablePath": "C:\\test.exe", "executableSha256": "c" * 64},
                "stages": {stage: {"peakBytes": 1, "samples": 1} for stage in control.STAGES}}}


def write_record(record: Path, body: dict) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    if body.get("backend") == "cuda":
        owned = body.get("measured", {}).get("owned")
        if isinstance(owned, dict):
            (record.parent / "stages.jsonl").write_text("".join(
                json.dumps({"stage": stage, "at": index + 1}) + "\n"
                for index, stage in enumerate((*control.STAGES, "done"))), encoding="utf-8")
            (record.parent / "profile-process.json").write_text(
                json.dumps({"processId": owned["process"]["pid"]}) + "\n", encoding="utf-8")
            journal = record.parent / "cuda-owned-samples.jsonl"
            pid = owned["process"]["pid"]
            luid = owned["selectedLuid"]
            counter = {"pid": pid, "parentPid": owned["process"]["parentPid"],
                       "counter": r"\GPU Process Memory(*)\Dedicated Usage",
                       "rows": [{"instance": f"pid_{pid}_{luid}_phys_0", "status": "0",
                                 "cookedValue": 1.0}]}
            journal.write_text("".join(json.dumps({"at": index + 1, "startedAt": index + 1,
                                                    "bytes": 1, "pid": owned["process"]["pid"],
                                                    "luid": owned["selectedLuid"], "counter": counter}) + "\n"
                                       for index, _ in enumerate(control.STAGES)), encoding="utf-8")
            (record.parent / "cuda-owned-faults.jsonl").write_bytes(b"")
            owned["journalSha256"] = control.sha256(journal)
    record.write_text(json.dumps(body), encoding="utf-8")


class PrecisionControlTests(unittest.TestCase):
    def test_preflight_console_is_compact_but_retains_full_physical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            files = {f"counter-{index:02}.json": json.dumps({"index": index,
                     "physicalEvidence": "retained raw counter bytes" * 80}) for index in range(29)}
            encoded = {name: base64.b64encode(body.encode()).decode() for name, body in files.items()}
            raw = json.dumps({"physicalMode": "shared-gpu1", "diagnosticFiles": files,
                              "diagnosticFileBytesB64": encoded})
            proof_spec = importlib.util.spec_from_file_location(
                "preflight_retain_control", MODULE_PATH.with_name("yue2_precision_proof.py"))
            proof = importlib.util.module_from_spec(proof_spec)
            proof_spec.loader.exec_module(proof)

            census = types.SimpleNamespace(cuda_physical_census=lambda **_: (raw, []),
                                           metal_census=Mock(), physical_busy_message=Mock(),
                                           same_selected_cuda_device=Mock(),
                                           retain_cuda_physical_evidence=proof.retain_cuda_physical_evidence,
                                           retain_reviewed_baseline=Mock())
            idle = types.SimpleNamespace(check_shared_gpu1_dispatch=Mock())
            argv = ["yue2_app_precision_profile.py", "preflight", "--backend", "cuda",
                    "--evidence", str(evidence), "--label", "initial"]
            stdout = StringIO()
            with patch.dict(sys.modules, {"yue2_precision_proof": census,
                                          "yue2_cuda_idle_context": idle}), \
                 patch.dict(os.environ, {"COMPUTERNAME": "unit-host", "YUE2_IDLE_CONTEXT_RUN_ID": ""}), \
                 patch.object(control, "remaining_app_budget", return_value=3600), \
                 patch.object(control.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 12)), \
                 patch.object(sys, "argv", argv), redirect_stdout(stdout):
                self.assertEqual(control.main(), 0)

            receipt = evidence / "preflight-initial.json"
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            line = stdout.getvalue()
            printed = json.loads(line)
            self.assertEqual(stored["census"], raw)
            self.assertIn(encoded["counter-00.json"], receipt.read_text(encoding="utf-8"))
            self.assertEqual(len(list((evidence / "physical-initial").iterdir())), 29)
            self.assertEqual((evidence / "physical-initial" / "counter-00.json").read_bytes(),
                             files["counter-00.json"].encode())
            self.assertNotIn("diagnosticFileBytesB64", line)
            self.assertNotIn(encoded["counter-00.json"], line)
            self.assertLess(len(line), 1024)
            self.assertEqual(printed["receipt_sha256"], control.sha256(receipt))
            self.assertEqual(printed["physical_file_count"], 29)
            idle.check_shared_gpu1_dispatch.assert_called_once()

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
                "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages()}},
                "outcome": {"status": "completed", "engineComputePolicy": "bf16",
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "bfloat16",
                            "engineVaeCudaBf16MathPolicy": M4_POLICY},
                "measured": {"peakBytes": 1024, "stages": stages, **measured_scope("cuda")},
            }
            write_record(record, body)
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)["effective_vae_dtype"], "bfloat16")
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
                             ["owned_peak_bytes"], 1)
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
                             ["selected_device_global_used_peak_bytes"], 1024)
            body["measured"]["owned"]["peakBytes"] = 0
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "owned overall peak"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            body["measured"]["owned"]["peakBytes"] = 1
            write_record(record, body)
            (record.parent / "cuda-owned-samples.jsonl").write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "journal changed"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            write_record(record, body)
            lines = (record.parent / "cuda-owned-samples.jsonl").read_text(encoding="utf-8").splitlines()
            altered = json.loads(lines[0])
            altered["counter"]["rows"][0]["cookedValue"] = 2
            lines[0] = json.dumps(altered)
            journal = record.parent / "cuda-owned-samples.jsonl"
            journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
            body["measured"]["owned"]["journalSha256"] = control.sha256(journal)
            record.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counter journal lacks bound"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            write_record(record, body)
            lines = (record.parent / "cuda-owned-samples.jsonl").read_text(encoding="utf-8").splitlines()
            altered = json.loads(lines[0])
            altered["bytes"] = 2048
            altered["counter"]["rows"][0]["cookedValue"] = 2048.0
            lines[0] = json.dumps(altered)
            journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
            body["measured"]["owned"]["journalSha256"] = control.sha256(journal)
            record.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "peaks differ from retained counter timeline"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            write_record(record, body)
            (record.parent / "profile-process.json").write_text('{"processId":999}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "process receipt"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            write_record(record, body)
            (record.parent / "cuda-owned-faults.jsonl").write_text('{"fault":"missing"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "journal changed or has faults"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            write_record(record, body)
            del body["measured"]["scope"]
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "global selected-device"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            body["measured"].update(measured_scope("cuda"))
            del body["admission"]["estimate"]["stages"]["load"]
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "modeled stage"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            body["admission"]["estimate"]["stages"] = estimated_stages()
            body["outcome"]["engineVaeDtype"] = "float32"
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "effective engineVaeDtype"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)
            body["outcome"]["engineVaeDtype"] = "bfloat16"
            body["identity"]["decoder"]["repo"] = "m-a-p/YuE2-Vae"
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "decoder identity"):
                control.verify_record(record, "cuda", "strict-bf16-legacy", M4_POLICY)

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
                    "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages()}},
                    "outcome": {"status": "completed", "engineComputePolicy": policy,
                                "engineModelDtype": model_dtype, "engineVaeDtype": vae_dtype},
                    "measured": {"peakBytes": 1024, **measured_scope(backend), "stages": {
                        stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
                }
                wanted = M4_POLICY
                if backend == "cuda" and policy == "bf16":
                    write_record(record, body)
                    with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                        control.verify_record(record, backend, name, M4_POLICY)
                    for wrong in ("stale", "fixed_order_bf16_convolution_v1"):
                        body["outcome"]["engineVaeCudaBf16MathPolicy"] = wrong
                        write_record(record, body)
                        with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                            control.verify_record(record, backend, name, M4_POLICY)
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = wanted
                    write_record(record, body)
                    self.assertEqual(
                        control.verify_record(record, backend, name, M4_POLICY)["effective_vae_cuda_bf16_math_policy"],
                        wanted,
                    )
                else:
                    write_record(record, body)
                    self.assertIsNone(
                        control.verify_record(record, backend, name, M4_POLICY)["effective_vae_cuda_bf16_math_policy"]
                    )
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = wanted
                    write_record(record, body)
                    with self.assertRaisesRegex(ValueError, "present on another backend"):
                        control.verify_record(record, backend, name, M4_POLICY)

    def test_fp8_auto_record_requires_actual_fp8_and_retained_host_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            name = "experimental-fp8-auto"
            body = {
                "caseId": control.case_id("cuda", name), "backend": "cuda",
                "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae"}},
                "request": {"name": name, "computePolicy": "auto", "arMode": "experimentalFp8"},
                "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages(),
                    "weights": {"hostBytes": 2 * 1024 ** 3}}},
                "outcome": {"status": "completed", "engineComputePolicy": "auto",
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "float32",
                            "engineQuantization": "fp8"},
                "measured": {"peakBytes": 1024, **measured_scope("cuda"), "stages": {
                    stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
            }
            def check():
                write_record(record, body)
                return control.verify_record(record, "cuda", name, M4_POLICY)
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
                ("engineVaeCudaBf16MathPolicy", M4_POLICY,
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
                control.verify_record(record, "metal", name, M4_POLICY)

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

    def test_runtime_math_policy_requires_the_verified_exact_engine_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, engine, evidence = (root / name for name in ("app", "engine", "evidence"))
            for path in (app, engine, evidence):
                path.mkdir()
            app_sha, control_sha = "a" * 40, "c" * 40
            source_control = MODULE_PATH.parents[2]
            (app / "Cargo.toml").write_text(
                f'candle-kernels = {{ git = "https://github.com/SceneWorks/inference", rev = "{M4_SHA}" }}\n',
                encoding="utf-8",
            )
            checked_out = {app: app_sha, engine: M4_SHA, source_control: control_sha}
            def fake_git(path, *args):
                return checked_out[path] if args[0] == "rev-parse" else ""
            with patch.dict("os.environ", {"GITHUB_SHA": control_sha}), \
                 patch.object(control, "git", side_effect=fake_git):
                with self.assertRaises(FileNotFoundError):
                    control.verified_runtime_policy(app, engine, evidence)
                source = control.verify_sources(app, engine, source_control,
                                                app_sha, M4_SHA, control_sha)
                (evidence / "sources.json").write_text(json.dumps(source), encoding="utf-8")
                self.assertEqual(control.verified_runtime_policy(app, engine, evidence),
                                 (M4_SHA, M4_POLICY))
                checked_out[engine] = "d" * 40
                with self.assertRaisesRegex(ValueError, "engine checkout SHA"):
                    control.verified_runtime_policy(app, engine, evidence)
                checked_out[engine] = M4_SHA
                (evidence / "sources.json").write_text(json.dumps({**source, "app_pins": []}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source manifest differs"):
                    control.verified_runtime_policy(app, engine, evidence)
                (evidence / "sources.json").write_text(json.dumps(source), encoding="utf-8")
                unknown_sha = "e" * 40
                checked_out[engine] = unknown_sha
                (app / "Cargo.toml").write_text(
                    (app / "Cargo.toml").read_text(encoding="utf-8").replace(M4_SHA, unknown_sha),
                    encoding="utf-8",
                )
                unknown_source = control.verify_sources(app, engine, source_control,
                                                        app_sha, unknown_sha, control_sha)
                (evidence / "sources.json").write_text(json.dumps(unknown_source), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsupported engine revision"):
                    control.verified_runtime_policy(app, engine, evidence)
                # A future policy can be tested without adding an unknown SHA to production.
                new_policy = "fixed_order_bf16_convolution_v1"
                self.assertEqual(control.verified_runtime_policy(
                    app, engine, evidence, supported={unknown_sha: new_policy}),
                    (unknown_sha, new_policy))

    def test_future_policy_rejects_an_old_cuda_record_and_other_backend_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            stages = {stage: {"peakBytes": 1, "samples": 1} for stage in control.STAGES}
            body = {
                "caseId": control.case_id("cuda", "strict-bf16-standard"), "backend": "cuda",
                "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae"}},
                "request": {"name": "strict-bf16-standard", "computePolicy": "bf16"},
                "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages()}},
                "outcome": {"status": "completed", "engineComputePolicy": "bf16",
                            "engineModelDtype": "bfloat16", "engineVaeDtype": "bfloat16",
                            "engineVaeCudaBf16MathPolicy": M4_POLICY},
                "measured": {"peakBytes": 1, **measured_scope("cuda"), "stages": stages},
            }
            write_record(record, body)
            new_policy = "fixed_order_bf16_convolution_v1"
            with self.assertRaisesRegex(ValueError, "does not match"):
                control.verify_record(record, "cuda", "strict-bf16-standard", new_policy)
            body["outcome"]["engineVaeCudaBf16MathPolicy"] = new_policy
            write_record(record, body)
            self.assertEqual(control.verify_record(record, "cuda", "strict-bf16-standard", new_policy)
                             ["effective_vae_cuda_bf16_math_policy"], new_policy)
            body["caseId"] = control.case_id("metal", "strict-bf16-standard")
            body["backend"] = "metal"
            write_record(record, body)
            with self.assertRaisesRegex(ValueError, "present on another backend"):
                control.verify_record(record, "metal", "strict-bf16-standard", new_policy)

    def test_m5_policy_requires_verified_clean_source_and_pin(self):
        self.assertEqual(control.SUPPORTED_RUNTIME_POLICIES[M5_SHA], M5_POLICY)
        self.assertEqual(control.SUPPORTED_RUNTIME_POLICIES[M4_SHA],
                         "disallow_reduced_precision_reduction_v1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, engine, evidence = (root / name for name in ("app", "engine", "evidence"))
            for path in (app, engine, evidence):
                path.mkdir()
            app_sha, control_sha = "a" * 40, "c" * 40
            source_control = MODULE_PATH.parents[2]
            pin = app / "Cargo.toml"
            pin.write_text(
                f'candle-kernels = {{ git = "https://github.com/SceneWorks/inference", rev = "{M5_SHA}" }}\n',
                encoding="utf-8",
            )
            checked_out = {app: app_sha, engine: M5_SHA, source_control: control_sha}
            dirty = set()

            def fake_git(path, *args):
                return checked_out[path] if args[0] == "rev-parse" else (" M changed" if path in dirty else "")

            with patch.dict("os.environ", {"GITHUB_SHA": control_sha}), patch.object(
                control, "git", side_effect=fake_git
            ):
                source = control.verify_sources(app, engine, source_control,
                                                app_sha, M5_SHA, control_sha)
                (evidence / "sources.json").write_text(json.dumps(source), encoding="utf-8")
                self.assertEqual(control.verified_runtime_policy(app, engine, evidence),
                                 (M5_SHA, M5_POLICY))

                dirty.add(engine)
                with self.assertRaisesRegex(ValueError, "engine checkout is dirty"):
                    control.verified_runtime_policy(app, engine, evidence)
                dirty.clear()
                pin.write_text(pin.read_text(encoding="utf-8").replace(M5_SHA, M4_SHA),
                               encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pin"):
                    control.verified_runtime_policy(app, engine, evidence)
                pin.write_text(pin.read_text(encoding="utf-8").replace(M4_SHA, M5_SHA),
                               encoding="utf-8")
                (evidence / "sources.json").write_text(
                    json.dumps({**source, "control_sha": "d" * 40}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "workflow control SHA"):
                    control.verified_runtime_policy(app, engine, evidence)

                unknown_sha = "e" * 40
                checked_out[engine] = unknown_sha
                pin.write_text(pin.read_text(encoding="utf-8").replace(M5_SHA, unknown_sha),
                               encoding="utf-8")
                unknown_source = control.verify_sources(app, engine, source_control,
                                                        app_sha, unknown_sha, control_sha)
                (evidence / "sources.json").write_text(json.dumps(unknown_source), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsupported engine revision"):
                    control.verified_runtime_policy(app, engine, evidence)

    def test_m5_cuda_bf16_record_refuses_old_unknown_or_leaked_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            for name in ("strict-bf16-standard", "strict-bf16-legacy",
                         "strict-bf16-q8-standard", "strict-bf16-q4-standard"):
                _, case_name, decoder, policy, model_dtype, vae_dtype = control.CASES[name]
                body = {
                    "caseId": control.case_id("cuda", name), "backend": "cuda",
                    "identity": {"decoder": {"repo": "m-a-p/YuE2-Vae" + ("-legacy" if decoder == "legacy" else "")}},
                    "request": {"name": case_name, "computePolicy": policy},
                    "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages()}},
                    "outcome": {"status": "completed", "engineComputePolicy": policy,
                                "engineModelDtype": model_dtype, "engineVaeDtype": vae_dtype,
                                "engineVaeCudaBf16MathPolicy": M5_POLICY},
                    "measured": {"peakBytes": 1, **measured_scope("cuda"), "stages": {
                        stage: {"peakBytes": 1, "samples": 1} for stage in control.STAGES}},
                }

                def check(backend="cuda"):
                    write_record(record, body)
                    return control.verify_record(record, backend, name, M5_POLICY)

                with self.subTest(name=name):
                    self.assertEqual(check()["effective_vae_cuda_bf16_math_policy"], M5_POLICY)
                    del body["outcome"]["engineVaeCudaBf16MathPolicy"]
                    with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                        check()
                    for wrong in (M4_POLICY, "unknown_policy"):
                        body["outcome"]["engineVaeCudaBf16MathPolicy"] = wrong
                        with self.assertRaisesRegex(ValueError, "effective CUDA BF16 VAE math policy"):
                            check()
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = M5_POLICY

            body["caseId"] = control.case_id("metal", name)
            body["backend"] = "metal"
            with self.assertRaisesRegex(ValueError, "present on another backend"):
                check("metal")
            del body["outcome"]["engineVaeCudaBf16MathPolicy"]
            self.assertIsNone(check("metal")["effective_vae_cuda_bf16_math_policy"])

            name = "strict-fp32-standard"
            _, case_name, _, policy, model_dtype, vae_dtype = control.CASES[name]
            body["caseId"] = control.case_id("cuda", name)
            body["backend"] = "cuda"
            body["request"] = {"name": case_name, "computePolicy": policy}
            body["outcome"].update(engineComputePolicy=policy, engineModelDtype=model_dtype,
                                   engineVaeDtype=vae_dtype)
            self.assertIsNone(check()["effective_vae_cuda_bf16_math_policy"])
            body["outcome"]["engineVaeCudaBf16MathPolicy"] = M5_POLICY
            with self.assertRaisesRegex(ValueError, "present on another backend"):
                check()

    def test_busy_cuda_preflight_refuses_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            census = types.SimpleNamespace(cuda_physical_census=lambda **_: ("typed pmon rows", ["123 C worker"]),
                                           metal_census=lambda: ("", []),
                                           physical_busy_message=lambda raw, busy, context: f"{context}: {busy}",
                                           same_selected_cuda_device=Mock(),
                                           retain_cuda_physical_evidence=lambda *_: [],
                                           retain_reviewed_baseline=lambda *_: [])
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control, "remaining_app_budget", return_value=3600), \
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
        census = types.SimpleNamespace(cuda_physical_census=lambda **_: (raw, busy),
                    metal_census=Mock(), physical_busy_message=proof.physical_busy_message,
                    same_selected_cuda_device=Mock(),
                    retain_cuda_physical_evidence=Mock(), retain_reviewed_baseline=Mock())
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            with patch.dict("sys.modules", {"yue2_precision_proof": census}), \
                 patch.object(control, "remaining_app_budget", return_value=3600), \
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
                                           same_selected_cuda_device=Mock(),
                                           retain_cuda_physical_evidence=Mock(),
                                           retain_reviewed_baseline=Mock())
            guard = types.SimpleNamespace(require_remaining_window=Mock(
                side_effect=RuntimeError("reviewed owner window cannot cover bounded run and postflight")))
            with patch.dict("sys.modules", {"yue2_precision_proof": census,
                                            "yue2_cuda_idle_context": guard}), \
                 patch.dict("os.environ", {"YUE2_IDLE_CONTEXT_RUN_ID": "reviewed"}), \
                 patch.object(control, "remaining_app_budget", return_value=3600):
                with self.assertRaisesRegex(RuntimeError, "cannot cover"):
                    control.preflight("cuda", Path(directory) / "evidence", "initial")
            guard.require_remaining_window.assert_called_once_with(480 * 60 + 600)
            census.cuda_physical_census.assert_not_called()

    def test_shared_host_cuda_keeps_absolute_app_job_tail(self):
        now = 900_000_000_000_000
        with patch.object(control.time, "time_ns", return_value=now), \
             patch.dict("os.environ", {"YUE2_APP_PRECISION_JOB_STARTED_UTC_NS": str(
                 now - (480 * 60 - 601) * 1_000_000_000)}):
            self.assertEqual(control.remaining_app_budget(), 1)
        with patch.object(control.time, "time_ns", return_value=now), \
             patch.dict("os.environ", {"YUE2_APP_PRECISION_JOB_STARTED_UTC_NS": str(
                 now - (480 * 60 - 599) * 1_000_000_000)}), \
             self.assertRaisesRegex(ValueError, "cleanup/upload tail"):
            control.remaining_app_budget()

    def test_shared_host_timeout_reaps_only_its_child_tree(self):
        class Owned:
            def __init__(self):
                self.pid = 8123
                self.released = False
            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("node", timeout)
            def poll(self):
                return -9 if self.released else None
        owned = Owned()
        owner = types.SimpleNamespace(reap_tree=lambda child: (
            setattr(child, "released", True) or -9, None))
        with patch.object(control, "remaining_app_budget", return_value=77), \
             patch.object(control.subprocess, "Popen", return_value=owned) as launch, \
             patch.dict("sys.modules", {"yue2_gpu0_owner_guard": owner}), \
             self.assertRaises(subprocess.TimeoutExpired):
            control.run_shared_command(["node", "capture"], Path("app"), {}, None)
        self.assertEqual(launch.call_args.args[0], ["node", "capture"])
        self.assertTrue(owned.released)

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
                 patch.object(control, "verified_runtime_policy", return_value=(M4_SHA, M4_POLICY)), \
                 patch.object(control, "run_shared_command", return_value=0), \
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
                    "admission": {"outcome": "admitted", "estimate": {"stages": estimated_stages()}},
                    "outcome": {"status": "completed", "engineComputePolicy": policy,
                                "engineModelDtype": model_dtype, "engineVaeDtype": vae_dtype},
                    "measured": {"peakBytes": 1024, **measured_scope("cuda"), "stages": {
                        stage: {"peakBytes": 1024, "samples": 1} for stage in control.STAGES}},
                }
                if policy == "bf16":
                    body["outcome"]["engineVaeCudaBf16MathPolicy"] = M4_POLICY
                if name == "experimental-fp8-auto":
                    body["request"]["arMode"] = "experimentalFp8"
                    body["outcome"]["engineQuantization"] = "fp8"
                    body["admission"]["estimate"]["weights"] = {"hostBytes": 2 * 1024 ** 3}
                evidence.mkdir(exist_ok=True)
                selected = body["measured"]["deviceProof"]
                proof = evidence / f"preflight-before-{name}.json"
                proof.write_text(json.dumps({"backend": "cuda", "admitted": True,
                                             "census": json.dumps({"validatedDevice": selected})}),
                                 encoding="utf-8")
                selected["sha256"] = control.sha256(proof)
                body["measured"]["owned"]["proofSha256"] = selected["sha256"]
                write_record(run.parent / "record.json", body)
            with patch.object(control, "verified_runtime_policy", return_value=(M4_SHA, M4_POLICY)):
                verdict = control.collect(profile, evidence, "cuda", root, root)
            self.assertEqual(len(verdict["listening_audio"]), 8)
            self.assertEqual(verdict["engine_sha"], M4_SHA)
            self.assertEqual(verdict["expected_cuda_bf16_vae_math_policy"], M4_POLICY)
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
        self.assertRegex(workflow,
                         r"(?m)^\s+- uses: \./app/\.github/actions/prepare-rust-runner\n"
                         r"\s+with:\n\s+workspace-directory: app$")
        self.assertIn("RUSTUP_TOOLCHAIN=", workflow)
        self.assertIn("Verify selected app Rust channel", workflow)


if __name__ == "__main__":
    unittest.main()
