"""Fail-closed tests for the SC-23942 evidence wrapper and validator."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "release" / "qwen38_bonsai_terminal.py"
SPEC = importlib.util.spec_from_file_location("qwen38_bonsai_terminal", SCRIPT)
assert SPEC and SPEC.loader
terminal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(terminal)


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime_sha = "b" * 40

    def tearDown(self) -> None:
        self.temp.cleanup()

    @unittest.skipIf(os.name == "nt", "fixture executable uses a POSIX shebang")
    def test_cpu_diagnostic_timeout_seals_partial_evidence_and_reaps_child_tree(self) -> None:
        grandchild_pid_path = self.root / "grandchild.pid"
        binary = self.root / "hanging-test"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
            f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)
        preflight_path = self.root / "preflight.json"
        preflight_path.write_text('{"load_profile":"candle-dense-cpu"}', encoding="utf-8")
        output = self.root / "timeout-run"
        args = terminal.parser().parse_args(
            ["run", "--binary", str(binary), "--test-name", "ignored_test",
             "--model-id", "parent", "--model-key", "parent", "--model-revision", "c" * 40,
             "--snapshot", str(self.root), "--model-path", str(self.root / "weights.gguf"),
             "--runtime-sha", self.runtime_sha, "--preflight", str(preflight_path),
             "--cases", "arithmetic", "--output", str(output), "--candle-device", "cpu",
             "--sample-interval", "0.05", "--diagnostic-timeout-seconds", "0.5"]
        )
        inventory = {"inventory_sha256": "e" * 64}
        sizes = {"language_weight_bytes": 1, "vision_weight_bytes": 0, "projector_bytes": 0}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(terminal, "source_identity", return_value={}))
            stack.enter_context(mock.patch.object(terminal, "load_model", return_value={"revision": "c" * 40, "key": "parent"}))
            stack.enter_context(mock.patch.object(terminal, "validate_variant_paths"))
            stack.enter_context(mock.patch.object(terminal, "verify_snapshot"))
            stack.enter_context(mock.patch.object(terminal, "snapshot_inventory", return_value=inventory))
            stack.enter_context(mock.patch.object(terminal, "artifact_sizes", return_value=sizes))
            stack.enter_context(mock.patch.object(terminal, "pinned_admission_sizes", return_value=sizes))
            stack.enter_context(mock.patch.object(terminal, "validate_preflight_record"))
            stack.enter_context(mock.patch.object(terminal, "selected_artifact", return_value={"path": "weights.gguf"}))
            stack.enter_context(mock.patch.object(terminal, "nvidia_sample", return_value=(None, "not sampled")))
            stack.enter_context(mock.patch.object(terminal, "physical_memory", return_value=(1, 1, None)))
            self.assertEqual(terminal.run(args), 1)
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "timed_out")
        self.assertIsNone(receipt["model"]["inventory_after"]["inventory_sha256"])
        self.assertIsNone(receipt["provider_evidence"])
        self.assertEqual(receipt["process"]["diagnostic_timeout_seconds"], 0.5)
        self.assertEqual(receipt["process"]["child_tree_cleanup"]["method"], "process_group")
        self.assertTrue(receipt["process"]["child_tree_cleanup"]["tree_termination_requested"])
        self.assertTrue(receipt["process"]["child_tree_cleanup"]["root_reaped"])
        self.assertLess((output / "progress-rss.jsonl").stat().st_size, 2 * 1024 * 1024)
        self.assertTrue((output / "progress-rss.jsonl").read_text(encoding="utf-8").strip())
        manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
        self.assertIn("progress-rss.jsonl", [item["path"] for item in manifest["files"]])
        with self.assertRaisesRegex(ValueError, "did not complete successfully"):
            terminal.validate_receipt(output)
        self.assertTrue(grandchild_pid_path.is_file())
        child_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
        for _ in range(20):
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True,
                encoding="utf-8",
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            self.fail(f"owned grandchild {child_pid} survived timeout cleanup")

    @unittest.skipIf(os.name == "nt", "POSIX process-group race regression")
    def test_cleanup_still_kills_group_after_root_exits(self) -> None:
        grandchild_pid_path = self.root / "orphan.pid"
        script = (
            "import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
            f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid))"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=5)
        child_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
        cleanup = terminal.terminate_owned_child(proc)
        self.assertTrue(cleanup["tree_termination_requested"])
        for _ in range(20):
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True,
                encoding="utf-8",
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            self.fail(f"owned grandchild {child_pid} survived post-exit cleanup")

    def make_run(
        self,
        name: str,
        *,
        request: dict | None = None,
        quality: bool = True,
        complete: bool = True,
        status: str = "completed",
        manifest_key: str | None = None,
        revision: str = "c" * 40,
        runtime_sha: str | None = None,
        backend: str = "candle",
        device: str = "cpu",
        load_profile: str = "candle-dense-cpu",
        preflight_sha256: str = "f" * 64,
        evidence_root: Path | None = None,
    ) -> Path:
        root = (evidence_root or self.root) / name
        root.mkdir()
        request = request or {"max_new_tokens": 128, "sampling": {"temperature": 0.0}}
        runtime_sha = runtime_sha or self.runtime_sha
        manifest_key = manifest_key or name
        run_id = "1" * 64
        process_id = 123

        def memory(scope: str, peak: int = 1) -> dict:
            if backend == "mlx":
                return {
                    "backend": backend,
                    "device": device,
                    "active_bytes": 1,
                    "cache_bytes": 0,
                    "peak_active_bytes": peak,
                    "peak_scope": scope,
                    "peak_metric": "native_active_allocator_bytes",
                }
            return {
                "backend": backend,
                "device": device,
                "peak_active_bytes": None,
                "peak_scope": "unavailable",
                "peak_metric": None,
                "native_allocator_counters_available": False,
                "peak_unavailable_reason": "native allocator counter unavailable",
                "memory_evidence": "external samples",
            }

        interval = {
            "run_id": run_id,
            "process_id": process_id,
            "model_id": name,
            "kind": "selected_case",
            "started_unix_seconds": 100.0,
            "ended_unix_seconds": 200.0,
        }
        provider = {
            "model_id": name,
            "model_revision": revision,
            "runtime_sha": runtime_sha,
            "run_id": run_id,
            "process_id": process_id,
            "status": "completed",
            "provider": {"backend": backend},
            "context_admission": {
                "evidence_complete": True,
                "measurement_interval": {**interval, "kind": "context_admission_probe"},
                "native_memory_before_case": memory(
                    "preceding_interval_since_explicit_reset"
                ),
                "native_memory_after_case": memory(
                    "context_admission_probe_interval_since_explicit_reset"
                ),
            },
            "resource_admission": {
                "evidence_complete": True,
                "architecturally_within_context": True,
                "available_memory_override_bytes": 1,
                "paired_case_id": "arithmetic",
                "paired_prompt_tokens": 10,
                "declared_context_tokens": 262144,
                "record": {
                    "status": "failed",
                    "request": request,
                    "error": "request requires an estimated 256 bytes of native workspace but only 1 bytes are available",
                    "measurement_interval": {**interval, "kind": "forced_budget_probe"},
                    "native_memory_before_case": memory(
                        "preceding_interval_since_explicit_reset"
                    ),
                    "native_memory_after_case": memory(
                        "forced_budget_probe_interval_since_explicit_reset"
                    ),
                },
            },
            "native_memory_before_load": memory(
                "pre_load_interval_since_explicit_reset"
            ),
            "native_memory_after_load": memory("load_interval_since_explicit_reset"),
            "native_memory_before_unload": memory(
                "preceding_interval_since_explicit_reset"
            ),
            "native_memory_after_unload": memory(
                "unload_interval_since_explicit_reset"
            ),
            "native_memory_summary": {
                "peak_active_bytes": 1 if backend == "mlx" else None,
                "peak_scope": "maximum_of_explicitly_reset_intervals"
                if backend == "mlx"
                else "unavailable",
                "peak_metric": "native_active_allocator_bytes"
                if backend == "mlx"
                else None,
                "coverage": [
                    {"kind": "pre_load"},
                    {"kind": "load"},
                    {"kind": "forced_budget_probe"},
                    {"kind": "selected_case", "case_id": "arithmetic"},
                    {"kind": "context_admission_probe"},
                    {"kind": "unload"},
                ],
                "coverage_note": "complete interval coverage",
                "unavailable_reason": None
                if backend == "mlx"
                else "native allocator counter unavailable",
            },
            "cases": [
                {
                    "case_id": "arithmetic",
                    "category": "math",
                    "request": request,
                    "status": "completed",
                    "evidence_complete": complete,
                    "quality_passed": quality,
                    "functional_acceptance_passed": quality,
                    "prefill_seconds": 0.004 if complete else None,
                    "decode_seconds": 0.002 if complete else None,
                    "measurement_interval": interval,
                    "native_memory_before_case": memory(
                        "preceding_interval_since_explicit_reset"
                    ),
                    "native_memory_after_case": memory(
                        "selected_case_interval_since_explicit_reset"
                    ),
                    "request_peak_claim_eligible": True,
                    "output": {
                        "text": "391" if quality else "392",
                        "thinking": None,
                        "tool_calls": [],
                        "prompt_tokens": 10,
                        "generated_tokens": 1,
                    },
                }
            ],
        }
        terminal.write_new(root / "provider.json", provider)
        (root / "stdout.log").write_text("ok\n", encoding="utf-8")
        (root / "stderr.log").write_text("", encoding="utf-8")
        manifest = terminal.artifact_manifest(
            root, ["provider.json", "stdout.log", "stderr.log"]
        )
        terminal.write_new(root / "artifact-manifest.json", manifest)
        inventory = {"inventory_sha256": "a" * 64}
        rss_samples = [
            {
                "seconds": 1.0,
                "started_unix_seconds": 120.0,
                "ended_unix_seconds": 121.0,
                "run_id": run_id,
                "process_id": process_id,
                "bytes": 4096,
            }
        ]
        gpu_samples: list[dict] = []
        receipt = {
            "schema_version": 1,
            "suite": terminal.SUITE,
            "status": status,
            "runtime": {"head_sha": runtime_sha, "clean_tree": True},
            "command": {
                "candle_device": None if backend == "mlx" else ("auto" if device == "cuda" else "cpu"),
                "load_profile": load_profile,
                "preflight_sha256": preflight_sha256,
            },
            "model": {
                "id": name,
                "manifest_key": manifest_key,
                "revision": revision,
                "language_variant": None,
                "vision_variant": None,
                "selected_model_artifact": {"kind": "file", "sha256": "d" * 64},
                "selected_projector_artifact": {"sha256": "e" * 64},
                "inventory_before": inventory,
                "inventory_after": inventory,
                "artifact_sizes": {
                    "language_weight_bytes": 100,
                    "vision_weight_bytes": 20,
                    "auxiliary_bytes": 0,
                },
            },
            "process": {
                "exit_code": 0,
                "run_id": run_id,
                "process_id": process_id,
                "rss_scope": "sampled_process_working_set_lower_bound",
                "sample_interval_seconds": 0.1,
                "rss_samples": rss_samples,
                "peak_rss_bytes": 4096,
            },
            "gpu": {
                "run_id": run_id,
                "process_id": process_id,
                "scope": "nvidia_smi_per_process_sampled_lower_bound",
                "available": False,
                "samples": gpu_samples,
                "peak_bytes": None,
                "unavailable_reason": "WDDM counter unavailable",
                "admission_recheck": {
                    "admitted": True,
                    "gpu_index": 0,
                    "selected_gpu_uuid": "GPU-0",
                    "selected_gpu_compute_processes": [],
                }
                if device == "cuda"
                else None,
            },
            "case_memory": terminal.case_memory_evidence(
                provider,
                rss_samples,
                gpu_samples,
                run_id=run_id,
                process_id=process_id,
            ),
            "provider_evidence": "provider.json",
            "artifact_manifest_sha256": terminal.sha256(root / "artifact-manifest.json"),
        }
        terminal.write_new(root / "receipt.json", receipt)
        terminal.write_new(
            root / "seal.json",
            {
                "schema_version": 1,
                "receipt_sha256": terminal.sha256(root / "receipt.json"),
                "artifact_manifest_sha256": terminal.sha256(
                    root / "artifact-manifest.json"
                ),
            },
        )
        return root

    def reseal_run(self, root: Path) -> None:
        manifest = terminal.artifact_manifest(
            root, ["provider.json", "stdout.log", "stderr.log"]
        )
        (root / "artifact-manifest.json").unlink(missing_ok=True)
        terminal.write_new(root / "artifact-manifest.json", manifest)
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        receipt["artifact_manifest_sha256"] = terminal.sha256(
            root / "artifact-manifest.json"
        )
        (root / "receipt.json").unlink()
        terminal.write_new(root / "receipt.json", receipt)
        (root / "seal.json").unlink()
        terminal.write_new(
            root / "seal.json",
            {
                "schema_version": 1,
                "receipt_sha256": terminal.sha256(root / "receipt.json"),
                "artifact_manifest_sha256": terminal.sha256(
                    root / "artifact-manifest.json"
                ),
            },
        )

    def add_context_decline(self, root: Path, *, available_bytes: int = 100) -> None:
        path = root / "provider.json"
        provider = json.loads(path.read_text(encoding="utf-8"))
        provider["provider"]["max_context_tokens"] = 262144
        short = copy.deepcopy(provider["cases"][0])
        short.update(case_id="context_64", category="context")
        short["stream_contract_passed"] = True
        short["request"] = {"max_new_tokens": 128, "sampling": {"temperature": 0.0}}
        short["output"]["prompt_tokens"] = 1187
        short["output"]["finish_reason"] = "Stop"
        short["events"] = [
            {
                "event": "token",
                "seconds": 0.1,
                "id": 7,
                "index": 0,
                "channel": "Content",
                "text": "391",
            },
            {
                "event": "done",
                "seconds": 0.2,
                "finish_reason": "Stop",
                "prompt_tokens": 1187,
                "generated_tokens": 1,
            },
        ]
        recovery = copy.deepcopy(short)
        long_request = {"max_new_tokens": 128, "sampling": {"temperature": 0.0}}
        required_bytes = 113_900_545_408
        decline = {
            "schema_version": 1,
            "case_id": "context_2048",
            "category": "context",
            "request": long_request,
            "oracle": {"kind": "exact", "value": "NEBULA-47"},
            "status": "resource_declined",
            "error": (
                f"invalid request: request requires an estimated {required_bytes} bytes of "
                f"native workspace but only {available_bytes} bytes are available; reduce "
                "prompt/media length or max_new_tokens"
            ),
            "evidence_complete": True,
            "resource_admission": {
                "preallocation_rejected": True,
                "prompt_tokens": 36899,
                "max_new_tokens": 128,
                "max_context_tokens": 262144,
                "required_bytes": required_bytes,
                "available_bytes": available_bytes,
            },
            "events": [],
            "recovery": recovery,
            "measurement_interval": {
                **short["measurement_interval"],
                "kind": "selected_case_including_recovery",
            },
            "native_memory_before_case": copy.deepcopy(
                short["native_memory_before_case"]
            ),
            "native_memory_after_case": copy.deepcopy(short["native_memory_after_case"]),
            "request_peak_claim_eligible": False,
        }
        if decline["native_memory_after_case"]["peak_scope"] != "unavailable":
            decline["native_memory_after_case"]["peak_scope"] = (
                "selected_case_including_recovery_interval_since_explicit_reset"
            )
        provider["cases"].extend([short, decline])
        provider["native_memory_summary"]["coverage"] = [
            {"kind": "pre_load"},
            {"kind": "load"},
            {"kind": "forced_budget_probe"},
            *[
                {
                    "kind": "selected_case_including_recovery"
                    if case["status"] == "resource_declined"
                    else "selected_case",
                    "case_id": case["case_id"],
                }
                for case in provider["cases"]
            ],
            {"kind": "context_admission_probe"},
            {"kind": "unload"},
        ]
        path.write_text(json.dumps(provider), encoding="utf-8")
        receipt_path = root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["case_memory"] = terminal.case_memory_evidence(
            provider,
            receipt["process"]["rss_samples"],
            receipt["gpu"]["samples"],
            run_id=receipt["process"]["run_id"],
            process_id=receipt["process"]["process_id"],
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.reseal_run(root)

    def write_manifest(self, rows: list[tuple[str, str]], name: str = "models.toml") -> Path:
        manifest = self.root / name
        content = ["schema_version = 1", ""]
        for key, revision in rows:
            content.extend(
                [
                    "[[models]]",
                    f'key = "{key}"',
                    f'revision = "{revision}"',
                    "admission_language_weight_bytes = 100",
                    "admission_vision_weight_bytes = 20",
                    "",
                ]
            )
        manifest.write_text("\n".join(content), encoding="utf-8")
        return manifest

    def write_hardware(self, evidence_root: Path | None = None) -> None:
        terminal.write_new(
            (evidence_root or self.root) / "hardware-before.json",
            {
                "schema_version": 1,
                "captured_unix_seconds": 1.0,
                "host": {"system": "test"},
                "gpus": [],
                "compute_processes": [],
            },
        )

    def make_preflight(
        self,
        name: str,
        *,
        model_key: str,
        revision: str,
        load_profile: str,
        admitted: bool = True,
        evidence_root: Path | None = None,
    ) -> Path:
        policy = terminal.LOAD_PROFILES[load_profile]
        gpu = policy["device"] == "cuda"
        reserve = 0
        weight_bytes = 120
        host_required = weight_bytes * policy["host_weight_copies"]
        gpu_required = weight_bytes if gpu else None
        selected = {
            "index": 0,
            "name": "test",
            "uuid": "GPU-0",
            "total_bytes": 1000,
            "free_bytes": 1000,
            "used_bytes": 0,
            "driver_version": "test",
        }
        record = {
            "schema_version": 1,
            "model_key": model_key,
            "model_revision": revision,
            "language_variant": None,
            "vision_variant": None,
            "artifact_sizes": {
                "language_weight_bytes": 100,
                "vision_weight_bytes": 20,
                "auxiliary_bytes": 0,
            },
            "load_profile": load_profile,
            "host_weight_copies": policy["host_weight_copies"],
            "gpu_weight_copies": policy["gpu_weight_copies"],
            "reserve_bytes": reserve,
            "host_required_available_bytes": host_required,
            "gpu_required_available_bytes": gpu_required,
            "physical_memory_bytes": 2000,
            "available_memory_bytes": 2000 if admitted else 0,
            "gpus": [selected] if gpu else [],
            "compute_processes": [],
            "selected_gpu_index": 0 if gpu else None,
            "selected_gpu_uuid": "GPU-0" if gpu else None,
            "selected_gpu_available_bytes": 1000 if gpu else None,
            "selected_gpu_compute_processes": [] if gpu else None,
            "reservation_token_sha256": "9" * 64 if gpu else None,
            "host_admitted": admitted,
            "gpu_admitted": True,
            "admitted": admitted,
        }
        evidence_root = evidence_root or self.root
        path = evidence_root / f"{name}-preflight.json"
        terminal.write_new(path, record)
        if gpu and not (evidence_root / "gpu-reservation.json").exists():
            terminal.write_new(
                evidence_root / "gpu-reservation.json",
                {
                    "schema_version": 1,
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-0",
                    "token_sha256": "9" * 64,
                },
            )
        return path

    def validate(self, runs: list[Path]) -> dict:
        output = self.root / "report.json"
        markdown = self.root / "report.md"
        args = argparse.Namespace(
            run=runs,
            expected_model=[path.name for path in runs],
            runtime_sha=self.runtime_sha,
            output=output,
            markdown=markdown,
        )
        self.assertEqual(terminal.validate(args), 0)
        self.assertTrue(markdown.read_text(encoding="utf-8").startswith("# Qwen3.8"))
        return json.loads(output.read_text(encoding="utf-8"))

    def test_quality_miss_is_reported_without_breaking_complete_evidence(self) -> None:
        runs = [self.make_run(name, quality=name != "bonsai-gguf") for name in (
            "qwen38-parent", "bonsai-mlx", "bonsai-gguf", "qwen3vl-8b"
        )]
        report = self.validate(runs)
        self.assertTrue(report["evidence_complete"])
        self.assertFalse(report["claims"]["quality_threshold_applied"])
        misses = [row for row in report["raw_outputs"] if not row["quality_passed"]]
        self.assertEqual([row["model_id"] for row in misses], ["bonsai-gguf"])

    def test_incomplete_or_failed_case_is_not_quality_evidence(self) -> None:
        run = self.make_run("broken", complete=False)
        with self.assertRaisesRegex(ValueError, "broken evidence"):
            terminal.validate_receipt(run)
        failed = self.make_run("failed", status="failed")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            terminal.validate_receipt(failed)

    def test_mismatched_budget_is_rejected(self) -> None:
        first = self.make_run("one")
        second = self.make_run(
            "two", request={"max_new_tokens": 64, "sampling": {"temperature": 0.0}}
        )
        args = argparse.Namespace(
            run=[first, second],
            expected_model=["one", "two"],
            runtime_sha=self.runtime_sha,
            output=self.root / "out.json",
            markdown=self.root / "out.md",
        )
        with self.assertRaisesRegex(ValueError, "unequal requests or budgets"):
            terminal.validate(args)

    def test_mixed_runtime_shas_are_rejected(self) -> None:
        first = self.make_run("one")
        second = self.make_run("two", runtime_sha="a" * 40)
        args = argparse.Namespace(
            run=[first, second],
            expected_model=["one", "two"],
            runtime_sha=self.runtime_sha,
            output=self.root / "out.json",
            markdown=self.root / "out.md",
        )
        with self.assertRaisesRegex(ValueError, "runtime SHA mismatch"):
            terminal.validate(args)

    def test_provider_identity_must_match_receipt(self) -> None:
        run = self.make_run("identity")
        provider = json.loads((run / "provider.json").read_text(encoding="utf-8"))
        provider["model_revision"] = "a" * 40
        (run / "provider.json").write_text(json.dumps(provider), encoding="utf-8")
        self.reseal_run(run)
        with self.assertRaisesRegex(ValueError, "provider model_revision"):
            terminal.validate_receipt(run)

    def test_resource_admission_requires_a_matching_within_window_workload(self) -> None:
        mutations = {
            "different-request": lambda resource: resource["record"].update(request={}),
            "architectural-rejection": lambda resource: resource["record"].update(error="context limit exceeded"),
            "outside-window": lambda resource: resource.update(declared_context_tokens=10),
            "invented-token-count": lambda resource: resource.update(paired_prompt_tokens=999),
            "missing-pair": lambda resource: resource.update(paired_case_id="absent"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                run = self.make_run(name)
                path = run / "provider.json"
                provider = json.loads(path.read_text(encoding="utf-8"))
                mutate(provider["resource_admission"])
                path.write_text(json.dumps(provider), encoding="utf-8")
                self.reseal_run(run)
                with self.assertRaisesRegex(ValueError, "resource rejection"):
                    terminal.validate_receipt(run)

    def test_context_resource_decline_is_a_limitation_with_real_recovery(self) -> None:
        first = self.make_run("decline-a")
        second = self.make_run("decline-b")
        self.add_context_decline(first, available_bytes=0)
        self.add_context_decline(second)
        terminal.validate_receipt(first)
        args = argparse.Namespace(
            run=[first, second],
            expected_model=["decline-a", "decline-b"],
            runtime_sha=self.runtime_sha,
            case_ids=["arithmetic", "context_64", "context_2048"],
            output=self.root / "comparison.json",
            markdown=self.root / "comparison.md",
        )
        self.assertEqual(terminal.validate(args), 0)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["resource_declines"]), 2)
        self.assertNotIn("context_2048", {row["case_id"] for row in report["raw_outputs"]})
        self.assertIn("not quality or latency samples", args.markdown.read_text(encoding="utf-8"))

    def test_context_resource_decline_mutations_fail_closed(self) -> None:
        mutations = {
            "missing-field": lambda case, provider: case["resource_admission"].pop("prompt_tokens"),
            "not-exhausted": lambda case, provider: case["resource_admission"].update(
                required_bytes=100, available_bytes=100
            ),
            "architecture-overflow": lambda case, provider: case["resource_admission"].update(
                prompt_tokens=262100
            ),
            "fabricated-output": lambda case, provider: case.update(output={"prompt_tokens": 36899}),
            "fabricated-timing": lambda case, provider: case.update(prefill_seconds=0.0),
            "fabricated-stream": lambda case, provider: case["events"].append(
                {"event": "token", "index": 0, "channel": "Content", "text": "fake"}
            ),
            "fabricated-request-peak": lambda case, provider: case.update(
                request_peak_claim_eligible=True
            ),
            "non-context": lambda case, provider: case.update(
                case_id="arithmetic", category="math"
            ),
            "missing-recovery": lambda case, provider: case.pop("recovery"),
            "failed-recovery": lambda case, provider: case["recovery"].update(status="failed"),
            "zero-available-failed-recovery": lambda case, provider: (
                case["resource_admission"].update(available_bytes=0),
                case.update(
                    error=(
                        "invalid request: request requires an estimated 113900545408 bytes of "
                        "native workspace but only 0 bytes are available; reduce prompt/media "
                        "length or max_new_tokens"
                    )
                ),
                case["recovery"].update(status="failed"),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                run = self.make_run(f"decline-{name}")
                self.add_context_decline(run)
                path = run / "provider.json"
                provider = json.loads(path.read_text(encoding="utf-8"))
                decline = provider["cases"][-1]
                mutate(decline, provider)
                path.write_text(json.dumps(provider), encoding="utf-8")
                self.reseal_run(run)
                with self.assertRaisesRegex(ValueError, "context|non-context"):
                    terminal.validate_receipt(run)

    def test_memory_evidence_scopes_identity_and_interval_fail_closed(self) -> None:
        mlx = self.make_run(
            "mlx-memory",
            backend="mlx",
            device="unified",
            load_profile="mlx-unified",
        )
        terminal.validate_receipt(mlx)
        args = argparse.Namespace(
            run=[mlx],
            expected_model=["mlx-memory"],
            runtime_sha=self.runtime_sha,
            case_ids=["arithmetic"],
            output=self.root / "memory-comparison.json",
            markdown=self.root / "memory-comparison.md",
        )
        self.assertEqual(terminal.validate(args), 0)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        model = report["models"][0]
        self.assertEqual(
            model["peak_rss_scope"], "sampled_process_working_set_lower_bound"
        )
        self.assertEqual(
            model["peak_gpu_scope"], "nvidia_smi_per_process_sampled_lower_bound"
        )
        native = report["raw_outputs"][0]["memory"]["native"]
        self.assertEqual(native["scope"], "selected_case_interval_since_explicit_reset")
        self.assertEqual(native["peak_bytes"], 1)

        for label, target, mutate, expected in (
            (
                "missing-scope",
                "provider",
                lambda value: value["cases"][0]["native_memory_after_case"].pop(
                    "peak_scope"
                ),
                "scope",
            ),
            (
                "wrong-pid",
                "receipt",
                lambda value: value["case_memory"][0]["rss"]["samples"][0].update(
                    process_id=999
                ),
                "identity|interval",
            ),
            (
                "crossed-boundary",
                "receipt",
                lambda value: value["case_memory"][0]["rss"]["samples"][0].update(
                    ended_unix_seconds=201.0
                ),
                "crosses",
            ),
        ):
            with self.subTest(label=label):
                run = self.make_run(
                    label,
                    backend="mlx",
                    device="unified",
                    load_profile="mlx-unified",
                )
                path = run / f"{target}.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                self.reseal_run(run)
                with self.assertRaisesRegex(ValueError, expected):
                    terminal.validate_receipt(run)

    def test_sampled_zero_is_preserved_and_unavailable_is_null(self) -> None:
        interval = {"started_unix_seconds": 1.0, "ended_unix_seconds": 2.0}
        zero = terminal.bounded_interval_samples(
            [
                {
                    "run_id": "r",
                    "process_id": 7,
                    "started_unix_seconds": 1.1,
                    "ended_unix_seconds": 1.2,
                    "bytes": 0,
                }
            ],
            interval,
            run_id="r",
            process_id=7,
            scope="sampled-test-lower-bound",
            unavailable_reason="missing",
        )
        self.assertTrue(zero["available"])
        self.assertEqual(zero["peak_bytes"], 0)
        self.assertEqual(zero["observed_growth_from_first_sample_bytes"], 0)
        missing = terminal.bounded_interval_samples(
            [],
            interval,
            run_id="r",
            process_id=7,
            scope="sampled-test-lower-bound",
            unavailable_reason="missing",
        )
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["peak_bytes"])
        self.assertEqual(missing["unavailable_reason"], "missing")

        run = self.make_run("missing-e7-sample")
        self.add_context_decline(run)
        receipt_path = run / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        context = next(
            row for row in receipt["case_memory"] if row["case_id"] == "context_64"
        )
        context["rss"] = terminal.bounded_interval_samples(
            [],
            context["measurement_interval"],
            run_id=receipt["process"]["run_id"],
            process_id=receipt["process"]["process_id"],
            scope="sampled_process_working_set_within_selected_request_lower_bound",
            unavailable_reason="no complete RSS sample fell within the selected case interval",
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.reseal_run(run)
        with self.assertRaisesRegex(ValueError, "required E7 case"):
            terminal.validate_receipt(run)

    def test_comparison_rejects_a_dropped_requested_case(self) -> None:
        run = self.make_run("dropped-case")
        args = argparse.Namespace(
            run=[run],
            expected_model=["dropped-case"],
            runtime_sha=self.runtime_sha,
            case_ids=["arithmetic", "context_64"],
            output=self.root / "dropped-comparison.json",
            markdown=self.root / "dropped-comparison.md",
        )
        with self.assertRaisesRegex(ValueError, "exact selected cases"):
            terminal.validate(args)

    def test_tampered_or_truncated_artifact_is_rejected(self) -> None:
        run = self.make_run("tampered")
        (run / "provider.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            terminal.validate_receipt(run)
        (run / "provider.json").unlink()
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            terminal.validate_receipt(run)

    def test_safetensors_bytes_are_split_by_language_and_vision(self) -> None:
        model = self.root / "model"
        model.mkdir()
        header = {
            "language_model.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "vision_tower.weight": {"dtype": "F32", "shape": [3], "data_offsets": [8, 20]},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        with (model / "model.safetensors").open("wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(bytes(20))
        (model / "tokenizer.json").write_bytes(b"tokens")
        # Snapshot inventory excludes materialization/cache markers from evidence.
        (model / ".cache").mkdir()
        (model / ".cache" / "metadata.json").write_bytes(b"ignored")
        sizes = terminal.artifact_sizes(model, None, {"files": [
            {"path": "model.safetensors", "size": (model / "model.safetensors").stat().st_size},
            {"path": "tokenizer.json", "size": 6},
        ]})
        self.assertEqual(sizes["language_weight_bytes"], 8)
        self.assertEqual(sizes["vision_weight_bytes"], 12)
        self.assertEqual(sizes["auxiliary_bytes"], 6)

    def test_auxiliary_evidence_uses_snapshot_inventory_not_admission_zero(self) -> None:
        files = [
            {"path": "model.safetensors", "size": 128, "sha256": "a" * 64},
            {"path": "tokenizer.json", "size": 17, "sha256": "b" * 64},
            {"path": "nested/config.json", "size": 5, "sha256": "c" * 64},
        ]
        projection = [{key: item[key] for key in ("path", "size", "sha256")} for item in files]
        digest = terminal.hashlib.sha256(
            json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        model = {
            "selected_model_artifact": {"kind": "snapshot", "sha256": digest},
            "inventory_before": {"inventory_sha256": digest, "files": files},
            "inventory_after": {"inventory_sha256": digest, "files": files},
            "artifact_sizes": {
                "language_weight_bytes": 100,
                "vision_weight_bytes": 20,
                "auxiliary_bytes": 22,
            },
        }
        pinned = {"language_weight_bytes": 100, "vision_weight_bytes": 20, "auxiliary_bytes": 0}
        terminal.validate_artifact_size_evidence(model, pinned)
        for auxiliary in (0, 21, 23):
            with self.subTest(auxiliary=auxiliary):
                model["artifact_sizes"]["auxiliary_bytes"] = auxiliary
                with self.assertRaisesRegex(ValueError, "auxiliary bytes"):
                    terminal.validate_artifact_size_evidence(model, pinned)
        model["artifact_sizes"]["auxiliary_bytes"] = 22
        model["artifact_sizes"]["language_weight_bytes"] = 99
        with self.assertRaisesRegex(ValueError, "weight bytes"):
            terminal.validate_artifact_size_evidence(model, pinned)
        model["artifact_sizes"]["language_weight_bytes"] = 100
        model["inventory_before"]["files"][1]["size"] = 18
        with self.assertRaisesRegex(ValueError, "inventory identity"):
            terminal.validate_artifact_size_evidence(model, pinned)

    def test_selected_hf_symlink_keeps_snapshot_filename_and_inventory_identity(self) -> None:
        snapshot = self.root / "snapshot"
        blobs = self.root / "blobs"
        snapshot.mkdir()
        blobs.mkdir()
        blob = blobs / "deadbeef"
        blob.write_bytes(b"gguf")
        selected = snapshot / "model.gguf"
        selected.symlink_to(blob)
        identity = terminal.selected_artifact(
            selected,
            snapshot,
            {
                "inventory_sha256": "a" * 64,
                "files": [
                    {
                        "path": "model.gguf",
                        "kind": "symlink",
                        "size": 4,
                        "sha256": "b" * 64,
                    }
                ],
            },
        )
        self.assertEqual(Path(identity["path"]).name, "model.gguf")
        self.assertEqual(identity["sha256"], "b" * 64)

    def test_pinned_admission_rejects_missing_or_nonpositive_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks positive pinned"):
            terminal.pinned_admission_sizes({"key": "broken"})
        with self.assertRaisesRegex(ValueError, "lacks positive pinned"):
            terminal.pinned_admission_sizes(
                {
                    "key": "broken",
                    "admission_language_weight_bytes": 10,
                    "admission_vision_weight_bytes": 0,
                }
            )

    def test_macos_memory_header_and_overlapping_purgeable_pages(self) -> None:
        output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1024.
Pages active:                                4096.
Pages inactive:                              2048.
Pages speculative:                            512.
Pages purgeable:                             1000.
"Translation faults":                      999999.
"""
        with mock.patch.object(terminal.sys, "platform", "darwin"), mock.patch.object(
            terminal.subprocess, "check_output", side_effect=[str(1024**3), "16384", output]
        ):
            total, available, reason = terminal.physical_memory()
        self.assertEqual(total, 1024**3)
        self.assertEqual(available, (1024 + 2048 + 512) * 16384)
        self.assertIsNone(reason)
        for malformed in (
            output.replace("1024.", "broken."),
            output.replace("1024.", "-1."),
            output.replace("Pages speculative:", "Unknown counter:"),
            output + "Pages free: 1.\n",
            output.replace("1024.", "999999999999."),
        ):
            with self.subTest(output=malformed), mock.patch.object(terminal.sys, "platform", "darwin"), mock.patch.object(
                terminal.subprocess, "check_output", side_effect=[str(1024**3), "16384", malformed]
            ):
                total, available, reason = terminal.physical_memory()
                self.assertEqual(total, 1024**3, "retain independently known physical capacity")
                self.assertIsNone(available)
                self.assertIn("macOS memory query failed", reason)

    def test_snapshot_candidates_are_exact_bounded_and_never_automatically_selected(self) -> None:
        revision = "a" * 40
        home = self.root / "home"
        hf_home = self.root / "hf-home"
        hub = self.root / "explicit-hub"
        suffix = Path("models--example--fixture") / "snapshots" / revision
        snapshot = hf_home / "hub" / suffix
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        manifest = self.root / "candidate-models.toml"
        manifest.write_text(f'[[models]]\nkey="fixture"\nrepository="example/fixture"\nrevision="{revision}"\nexpected_files=["config.json"]\n', encoding="utf-8")
        args = argparse.Namespace(binding=["MISSING_SNAPSHOT=fixture"], manifest=manifest,
                                  platform="windows", output=self.root / "candidate-metadata.json")
        with mock.patch.dict("os.environ", {"HF_HOME": str(hf_home), "HF_HUB_CACHE": str(hub)}, clear=True), \
                mock.patch.object(terminal.Path, "home", return_value=home), \
                mock.patch.object(terminal, "sha256", side_effect=AssertionError("payload hashing forbidden")), \
                mock.patch.object(terminal.Path, "rglob", side_effect=AssertionError("recursive search forbidden")):
            self.assertEqual(terminal.qualify_snapshots(args), 1)
        record = json.loads(args.output.read_text(encoding="utf-8"))
        binding = record["snapshots"][0]
        self.assertIsNone(binding["configured_path"])
        self.assertFalse(record["all_metadata_qualified"])
        candidates = binding["candidate_snapshots"]
        self.assertEqual([row["configured_path"] for row in candidates], [
            str(home / ".cache" / "huggingface" / "hub" / suffix), str(snapshot), str(hub / suffix)
        ])
        self.assertEqual([row["metadata_qualified"] for row in candidates], [False, True, False])
        self.assertTrue(all(row["artifact_identity_verified"] is False for row in candidates))

    def test_snapshot_metadata_qualification_is_read_only_and_revision_bound(self) -> None:
        revision = "a" * 40
        snapshot = self.root / revision
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        manifest = self.root / "metadata-models.toml"
        manifest.write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    "[[models]]",
                    'key = "fixture"',
                    'repository = "example/fixture"',
                    f'revision = "{revision}"',
                    'expected_files = ["config.json"]',
                ]
            ),
            encoding="utf-8",
        )
        output = self.root / "snapshot-metadata.json"
        args = argparse.Namespace(
            binding=["FIXTURE_SNAPSHOT=fixture"],
            manifest=manifest,
            platform="macos",
            output=output,
        )
        with mock.patch.dict("os.environ", {"FIXTURE_SNAPSHOT": str(snapshot)}):
            self.assertEqual(terminal.qualify_snapshots(args), 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(record["all_metadata_qualified"])
        self.assertIn("no payload hashing", record["scope"])
        self.assertEqual(record["snapshots"][0]["revision_from_snapshot_path"], revision)

    def test_snapshot_metadata_requires_all_indexed_shards_without_hashing_payloads(self) -> None:
        revision = "a" * 40
        snapshot = self.root / revision
        snapshot.mkdir()
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model-00001.safetensors"}}), encoding="utf-8"
        )
        manifest = self.root / "indexed.toml"
        manifest.write_text(
            f'[[models]]\nkey="fixture"\nrepository="example/fixture"\nrevision="{revision}"\n'
            'expected_files=["model.safetensors.index.json"]\n', encoding="utf-8"
        )
        args = argparse.Namespace(binding=["FIXTURE_SNAPSHOT=fixture"], manifest=manifest,
                                  platform="macos", output=self.root / "indexed-metadata.json")
        with mock.patch.dict("os.environ", {"FIXTURE_SNAPSHOT": str(snapshot)}), \
                mock.patch.object(terminal, "sha256", side_effect=AssertionError("payload hashing forbidden")):
            self.assertEqual(terminal.qualify_snapshots(args), 1)
            (snapshot / "model-00001.safetensors").write_bytes(b"fixture payload")
            args.output = self.root / "complete-indexed-metadata.json"
            self.assertEqual(terminal.qualify_snapshots(args), 0)
        record = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertEqual(len(record["snapshots"][0]["expected_file_metadata"]), 2)

    def test_snapshot_metadata_qualification_records_missing_configuration(self) -> None:
        manifest = self.root / "metadata-models.toml"
        manifest.write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    "[[models]]",
                    'key = "fixture"',
                    'repository = "example/fixture"',
                    f'revision = "{"a" * 40}"',
                    'expected_files = ["config.json"]',
                ]
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            binding=["ABSENT_FIXTURE_SNAPSHOT=fixture"],
            manifest=manifest,
            platform="windows",
            output=self.root / "missing-metadata.json",
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(terminal.qualify_snapshots(args), 1)
        record = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertFalse(record["all_metadata_qualified"])
        self.assertIsNone(record["snapshots"][0]["configured_path"])

    def test_full_acceptance_contract_resists_case_and_route_deletion(self) -> None:
        matrix = json.loads(
            (SCRIPT.parents[2] / "release" / "qwen38-bonsai-matrix.json").read_text(
                encoding="utf-8"
            )
        )
        terminal.validate_full_acceptance_contract(matrix)
        self.assertEqual(len(matrix["cells"]), 16)
        self.assertFalse(any(cell["device"] == "cpu" for cell in matrix["cells"]))
        mutants = []
        mutant = copy.deepcopy(matrix)
        del mutant["acceptance_contract"]
        mutants.append(mutant)
        for cell in matrix["cells"]:
            mutant = copy.deepcopy(matrix)
            mutant["cells"] = [row for row in mutant["cells"] if row["id"] != cell["id"]]
            mutants.append(mutant)
            if cell.get("functional_acceptance"):
                mutant = copy.deepcopy(matrix)
                next(row for row in mutant["cells"] if row["id"] == cell["id"])["functional_acceptance"] = False
                mutants.append(mutant)
        for case_id in terminal.QWEN38_ACCEPTANCE_CASES:
            mutant = copy.deepcopy(matrix)
            parent = next(cell for cell in mutant["cells"] if cell["id"] == "mlx-qwen38-parent")
            parent["acceptance_case_ids"].remove(case_id)
            mutants.append(mutant)
        for case_id in terminal.BONSAI_ACCEPTANCE_CASES:
            mutant = copy.deepcopy(matrix)
            bonsai = next(cell for cell in mutant["cells"] if cell["id"] == "mlx-bonsai-mlx-2bit")
            bonsai["acceptance_case_ids"].remove(case_id)
            mutants.append(mutant)
        for case_id in terminal.FORMAT_ACCEPTANCE_CASES:
            mutant = copy.deepcopy(matrix)
            mutant["groups"]["format-functional"]["case_ids"].remove(case_id)
            mutants.append(mutant)
        mutant = copy.deepcopy(matrix)
        mutant["cells"] = [
            cell for cell in mutant["cells"] if cell["id"] != "functional-candle-ptq1-q8"
        ]
        mutants.append(mutant)
        mutant = copy.deepcopy(matrix)
        cpu = copy.deepcopy(next(cell for cell in mutant["cells"] if cell["id"] == "candle-cuda-qwen38-parent"))
        cpu.update(id="candle-cpu-qwen38-parent", device="cpu", load_profile="candle-dense-cpu")
        mutant["cells"].append(cpu)
        mutants.append(mutant)
        for mutant in mutants:
            with self.subTest(mutant=mutant):
                with self.assertRaises(ValueError):
                    terminal.validate_full_acceptance_contract(mutant)

    def test_accelerator_only_manifest_rejects_cpu_preflight_before_host_probe(self) -> None:
        args = argparse.Namespace(
            reserve_bytes=0,
            manifest=SCRIPT.parents[2] / "release" / "real-weight-models.toml",
            model_key="bonsai-qwen38-parent",
            load_profile="candle-dense-cpu",
        )
        with mock.patch.object(
            terminal, "physical_memory", side_effect=AssertionError("host probe must not run")
        ), self.assertRaisesRegex(ValueError, "does not support the candle-dense-cpu"):
            terminal.preflight(args)

    def test_matrix_status_lists_every_missing_or_unadmitted_cell(self) -> None:
        self.write_hardware()
        manifest = self.write_manifest([("one", "1" * 40), ("two", "2" * 40)])
        matrix = self.root / "matrix.json"
        terminal.write_new(
            matrix,
            {
                "schema_version": 1,
                "suite": terminal.SUITE,
                "groups": {"functional": {"case_ids": ["arithmetic"]}},
                "cells": [
                    {
                        "id": "missing",
                        "group": "functional",
                        "backend": "candle",
                        "device": "cpu",
                        "model_key": "one",
                        "load_profile": "candle-dense-cpu",
                    },
                    {
                        "id": "rejected",
                        "group": "functional",
                        "backend": "candle",
                        "device": "cuda",
                        "model_key": "two",
                        "load_profile": "candle-dense-cuda",
                    },
                ],
            },
        )
        self.make_preflight(
            "rejected",
            model_key="two",
            revision="2" * 40,
            load_profile="candle-dense-cuda",
            admitted=False,
        )
        args = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / "matrix-report.json",
            markdown=self.root / "matrix-report.md",
            seal=self.root / "matrix-seal.json",
        )
        self.assertEqual(terminal.matrix_status(args), 1)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertFalse(report["evidence_complete"])
        self.assertEqual(
            [row["status"] for row in report["required_cells"]],
            ["incomplete", "not_admitted"],
        )

    def test_matrix_status_requires_and_compares_every_declared_cell(self) -> None:
        self.write_hardware()
        manifest = self.write_manifest(
            [("mlx-parent", "c" * 40), ("candle-parent", "c" * 40)]
        )
        cells = []
        for name, profile in (("mlx-parent", "mlx-unified"), ("candle-parent", "candle-dense-cpu")):
            preflight = self.make_preflight(
                name,
                model_key=name,
                revision="c" * 40,
                load_profile=profile,
            )
            backend = name.split("-", 1)[0]
            device = "unified" if backend == "mlx" else "cpu"
            self.make_run(
                name,
                backend=backend,
                device=device,
                load_profile=profile,
                preflight_sha256=terminal.sha256(preflight),
            )
            cell = {
                "id": name,
                "group": "matched",
                "backend": backend,
                "device": device,
                "model_key": name,
                "load_profile": profile,
            }
            cells.append(cell)
        matrix = self.root / "complete-matrix.json"
        terminal.write_new(
            matrix,
            {
                "schema_version": 1,
                "suite": terminal.SUITE,
                "groups": {"matched": {"case_ids": ["arithmetic"]}},
                "cells": cells,
            },
        )
        args = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / "complete-matrix-report.json",
            markdown=self.root / "complete-matrix-report.md",
            seal=self.root / "complete-matrix-seal.json",
        )
        self.assertEqual(terminal.matrix_status(args), 0)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertTrue(report["evidence_complete"])
        self.assertTrue(report["comparison"]["matched"]["evidence_complete"])
        verify = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=args.output,
            markdown=args.markdown,
            seal=args.seal,
        )
        self.assertEqual(terminal.verify_matrix_seal(verify), 0)
        selection = self.root / "selected-artifacts.json"
        terminal.write_new(selection, {"runtime_sha": self.runtime_sha, "artifact_ids": [11, 12]})
        selected_args = argparse.Namespace(
            root=[self.root], matrix=matrix, manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / "selected-matrix-report.json",
            markdown=self.root / "selected-matrix-report.md",
            seal=self.root / "selected-matrix-seal.json",
            artifact_selection=selection,
        )
        self.assertEqual(terminal.matrix_status(selected_args), 0)
        self.assertEqual(terminal.verify_matrix_seal(selected_args), 0)
        selection.write_text('{"artifact_ids": [99]}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matrix seal artifact mismatch"):
            terminal.verify_matrix_seal(selected_args)
        sealed = json.loads(args.seal.read_text(encoding="utf-8"))
        omitted = sealed["files"].pop(0)
        args.seal.write_text(json.dumps(sealed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact required evidence set"):
            terminal.verify_matrix_seal(verify)
        sealed["files"].insert(0, omitted)
        args.seal.write_text(json.dumps(sealed), encoding="utf-8")
        hardware = self.root / "hardware-before.json"
        hardware.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matrix seal artifact mismatch"):
            terminal.verify_matrix_seal(verify)

    def test_functional_acceptance_is_distinct_from_complete_evidence(self) -> None:
        self.write_hardware()
        manifest = self.write_manifest([("functional", "c" * 40)])
        preflight = self.make_preflight(
            "functional",
            model_key="functional",
            revision="c" * 40,
            load_profile="candle-dense-cpu",
        )
        self.make_run(
            "functional",
            quality=False,
            manifest_key="functional",
            preflight_sha256=terminal.sha256(preflight),
        )
        matrix = self.root / "functional-matrix.json"
        terminal.write_new(
            matrix,
            {
                "schema_version": 1,
                "suite": terminal.SUITE,
                "groups": {
                    "functional": {
                        "case_ids": ["arithmetic"],
                        "functional_acceptance": True,
                    }
                },
                "cells": [
                    {
                        "id": "functional",
                        "group": "functional",
                        "backend": "candle",
                        "device": "cpu",
                        "model_key": "functional",
                        "load_profile": "candle-dense-cpu",
                    }
                ],
            },
        )
        args = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / "functional-report.json",
            markdown=self.root / "functional-report.md",
            seal=self.root / "functional-seal.json",
        )
        self.assertEqual(terminal.matrix_status(args), 1)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertTrue(report["evidence_complete"])
        self.assertFalse(report["functional_acceptance"]["passed"])
        self.assertTrue(args.seal.is_file(), "complete failing evidence must remain sealed")

    def test_cell_acceptance_does_not_promote_diagnostic_quality(self) -> None:
        self.write_hardware()
        manifest = self.write_manifest([("functional", "c" * 40)])
        preflight = self.make_preflight(
            "functional",
            model_key="functional",
            revision="c" * 40,
            load_profile="candle-dense-cpu",
        )
        run = self.make_run(
            "functional",
            quality=False,
            manifest_key="functional",
            preflight_sha256=terminal.sha256(preflight),
        )
        provider_path = run / "provider.json"
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        capability = copy.deepcopy(provider["cases"][0])
        capability.update(
            case_id="capability",
            category="capability_acceptance",
            quality_passed=True,
            functional_acceptance_passed=True,
        )
        capability["output"]["text"] = "391"
        provider["cases"].append(capability)
        provider["native_memory_summary"]["coverage"].insert(
            -2, {"kind": "selected_case", "case_id": "capability"}
        )
        provider_path.write_text(json.dumps(provider), encoding="utf-8")
        receipt_path = run / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["case_memory"] = terminal.case_memory_evidence(
            provider,
            receipt["process"]["rss_samples"],
            receipt["gpu"]["samples"],
            run_id=receipt["process"]["run_id"],
            process_id=receipt["process"]["process_id"],
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.reseal_run(run)
        matrix = self.root / "cell-functional-matrix.json"
        terminal.write_new(
            matrix,
            {
                "schema_version": 1,
                "suite": terminal.SUITE,
                "groups": {"matched": {"case_ids": ["arithmetic"]}},
                "cells": [
                    {
                        "id": "functional",
                        "group": "matched",
                        "backend": "candle",
                        "device": "cpu",
                        "model_key": "functional",
                        "load_profile": "candle-dense-cpu",
                        "functional_acceptance": True,
                        "acceptance_case_ids": ["capability"],
                    }
                ],
            },
        )
        args = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / "cell-functional-report.json",
            markdown=self.root / "cell-functional-report.md",
            seal=self.root / "cell-functional-seal.json",
        )
        self.assertEqual(terminal.matrix_status(args), 0)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertTrue(report["evidence_complete"])
        self.assertTrue(report["functional_acceptance"]["passed"])
        self.assertEqual(
            [row["case_id"] for row in report["functional_acceptance"]["rows"]],
            ["capability"],
        )
        self.assertFalse(
            report["comparison"]["matched"]["raw_outputs"][0]["quality_passed"]
        )

    def test_preserve_thinking_evidence_mutations_fail_closed(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "setup"}], "thinking": None},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ack"}],
                "thinking": "private marker",
            },
            {"role": "user", "content": [{"type": "text", "text": "OK"}], "thinking": None},
        ]
        preserved_request = {
            "messages": messages,
            "max_new_tokens": 32,
            "preserve_thinking": True,
        }
        stripped_request = {**preserved_request, "preserve_thinking": False}

        def step(request: dict, prompt_tokens: int) -> dict:
            return {
                "status": "completed",
                "evidence_complete": True,
                "quality_passed": True,
                "stream_contract_passed": True,
                "request": request,
                "output": {
                    "text": "OK",
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": 1,
                },
            }

        record = {
            "case_id": "preserve_thinking",
            "request": {"preserved": preserved_request, "stripped": stripped_request},
            "status": "completed",
            "evidence_complete": True,
            "quality_passed": True,
            "functional_acceptance_passed": True,
            "stream_contract_passed": True,
            "history_coverage_passed": True,
            "prompt_token_proof": {
                "preserved_prompt_tokens": 14,
                "stripped_prompt_tokens": 10,
                "additional_preserved_tokens": 4,
                "passed": True,
            },
            "output": {
                "text": "OK",
                "prompt_tokens": 14,
                "generated_tokens": 1,
            },
            "paired_steps": {
                "preserved": step(preserved_request, 14),
                "stripped": step(stripped_request, 10),
            },
        }
        terminal.validate_preserve_thinking_evidence(record, self.root)
        for field in (
            "request",
            "prompt_token_proof",
            "stream_contract_passed",
            "history_coverage_passed",
        ):
            mutant = copy.deepcopy(record)
            del mutant[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                terminal.validate_preserve_thinking_evidence(mutant, self.root)

    def make_single_matrix(self, label: str) -> tuple[argparse.Namespace, Path, Path]:
        evidence = self.root / label
        evidence.mkdir()
        self.write_hardware(evidence)
        cell_id = f"{label}-cell"
        model_key = f"{label}-model"
        manifest = self.write_manifest(
            [(model_key, "c" * 40)], name=f"{label}-models.toml"
        )
        preflight = self.make_preflight(
            cell_id,
            model_key=model_key,
            revision="c" * 40,
            load_profile="candle-dense-cuda",
            evidence_root=evidence,
        )
        run = self.make_run(
            cell_id,
            manifest_key=model_key,
            backend="candle",
            device="cuda",
            load_profile="candle-dense-cuda",
            preflight_sha256=terminal.sha256(preflight),
            evidence_root=evidence,
        )
        matrix = self.root / f"{label}-matrix.json"
        terminal.write_new(
            matrix,
            {
                "schema_version": 1,
                "suite": terminal.SUITE,
                "groups": {"matched": {"case_ids": ["arithmetic"]}},
                "cells": [
                    {
                        "id": cell_id,
                        "group": "matched",
                        "backend": "candle",
                        "device": "cuda",
                        "model_key": model_key,
                        "load_profile": "candle-dense-cuda",
                    }
                ],
            },
        )
        args = argparse.Namespace(
            root=[evidence],
            matrix=matrix,
            manifest=manifest,
            runtime_sha=self.runtime_sha,
            output=self.root / f"{label}-report.json",
            markdown=self.root / f"{label}-report.md",
            seal=self.root / f"{label}-seal.json",
        )
        return args, preflight, run

    def test_matrix_accepts_inventoried_auxiliary_and_rejects_undercount(self) -> None:
        for label, measured, accepted in (("aux-exact", 22, True), ("aux-undercount", 0, False)):
            with self.subTest(label=label):
                args, _, run = self.make_single_matrix(label)
                receipt_path = run / "receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                files = [
                    {"path": "model.safetensors", "size": 128, "sha256": "a" * 64},
                    {"path": "tokenizer.json", "size": 22, "sha256": "b" * 64},
                ]
                digest = terminal.hashlib.sha256(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                receipt["model"]["selected_model_artifact"] = {"kind": "snapshot", "sha256": digest}
                inventory = {"inventory_sha256": digest, "files": files}
                receipt["model"]["inventory_before"] = inventory
                receipt["model"]["inventory_after"] = inventory
                receipt["model"]["artifact_sizes"]["auxiliary_bytes"] = measured
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                self.reseal_run(run)
                self.assertEqual(terminal.matrix_status(args), 0 if accepted else 1)
                report = json.loads(args.output.read_text(encoding="utf-8"))
                self.assertEqual(report["evidence_complete"], accepted)
                if not accepted:
                    self.assertIn("auxiliary bytes", report["required_cells"][0]["reason"])

    def test_matrix_rejects_stale_revision_wrong_device_backend_and_preflight(self) -> None:
        def mutate_backend(provider: dict) -> None:
            provider["provider"]["backend"] = "mlx"
            for point in (
                provider["native_memory_before_load"],
                provider["native_memory_after_load"],
                provider["native_memory_after_unload"],
            ):
                point["backend"] = "mlx"

        mutations = {
            "stale-revision": ("revision", lambda _: None),
            "cpu-in-cuda": ("receipt", lambda receipt: receipt["command"].__setitem__("candle_device", "cpu")),
            "backend-mismatch": ("provider", mutate_backend),
            "altered-preflight": ("preflight", lambda preflight: preflight.__setitem__("admitted", False)),
        }
        for label, (target, mutate) in mutations.items():
            with self.subTest(mutation=label):
                args, preflight_path, run = self.make_single_matrix(label)
                if target == "revision":
                    receipt_path = run / "receipt.json"
                    provider_path = run / "provider.json"
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    provider = json.loads(provider_path.read_text(encoding="utf-8"))
                    receipt["model"]["revision"] = "a" * 40
                    provider["model_revision"] = "a" * 40
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                    provider_path.write_text(json.dumps(provider), encoding="utf-8")
                    self.reseal_run(run)
                    path = None
                elif target == "receipt":
                    path = run / "receipt.json"
                elif target == "provider":
                    path = run / "provider.json"
                else:
                    path = preflight_path
                if path is not None:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    mutate(value)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    if target in ("receipt", "provider"):
                        self.reseal_run(run)
                self.assertEqual(terminal.matrix_status(args), 1)
                report = json.loads(args.output.read_text(encoding="utf-8"))
                self.assertFalse(report["evidence_complete"])

    def test_cuda_admission_uses_device_zero_and_rejects_only_selected_tenants(self) -> None:
        gpus = [
            {"index": 0, "uuid": "GPU-0", "free_bytes": 50},
            {"index": 1, "uuid": "GPU-1", "free_bytes": 10_000},
        ]
        with mock.patch.object(terminal, "nvidia_hardware", return_value=(gpus, [], None)):
            state = terminal.current_cuda_admission(gpu_index=0, required_bytes=120)
        self.assertFalse(state["admitted"])
        selected_tenant = [{"gpu_uuid": "GPU-0", "pid": 10}]
        with mock.patch.object(
            terminal, "nvidia_hardware", return_value=(gpus, selected_tenant, None)
        ):
            state = terminal.current_cuda_admission(gpu_index=0, required_bytes=10)
        self.assertFalse(state["admitted"])
        other_tenant = [{"gpu_uuid": "GPU-1", "pid": 11}]
        with mock.patch.object(
            terminal, "nvidia_hardware", return_value=(gpus, other_tenant, None)
        ):
            state = terminal.current_cuda_admission(gpu_index=0, required_bytes=10)
        self.assertTrue(state["admitted"])
        with mock.patch.object(terminal, "nvidia_hardware", return_value=([], [], "missing")):
            state = terminal.current_cuda_admission(gpu_index=0, required_bytes=0)
        self.assertFalse(state["admitted"])

    def test_cuda_preflight_applies_selected_device_and_cotenant_policy(self) -> None:
        manifest = self.write_manifest([("cuda-model", "c" * 40)])
        token = "campaign"
        reservation = self.root / "reservation.json"
        terminal.write_new(
            reservation,
            {
                "schema_version": 1,
                "gpu_index": 0,
                "gpu_uuid": "GPU-0",
                "token_sha256": terminal.reservation_token_sha256(token),
            },
        )
        base = {
            "model_key": "cuda-model",
            "language_variant": None,
            "vision_variant": None,
            "manifest": manifest,
            "load_profile": "candle-packed-cuda",
            "reserve_bytes": 0,
            "reservation": reservation,
            "reservation_token": token,
        }
        gpu0 = {"index": 0, "uuid": "GPU-0", "free_bytes": 50}
        gpu1 = {"index": 1, "uuid": "GPU-1", "free_bytes": 10_000}
        scenarios = [
            ("other-device-free", [gpu0, gpu1], [], 1),
            (
                "selected-tenant",
                [{**gpu0, "free_bytes": 1000}, gpu1],
                [{"gpu_uuid": "GPU-0", "pid": 1}],
                1,
            ),
            (
                "other-device-tenant",
                [{**gpu0, "free_bytes": 1000}, gpu1],
                [{"gpu_uuid": "GPU-1", "pid": 2}],
                0,
            ),
            ("missing-counters", [], [], 1),
        ]
        for label, gpus, processes, expected in scenarios:
            with self.subTest(scenario=label), mock.patch.object(
                terminal, "physical_memory", return_value=(2000, 2000, None)
            ), mock.patch.object(
                terminal, "nvidia_hardware", return_value=(gpus, processes, None)
            ):
                args = argparse.Namespace(
                    **base, output=self.root / f"{label}-preflight.json"
                )
                self.assertEqual(terminal.preflight(args), expected)
    def test_cuda_recheck_rejects_process_appearing_after_preflight(self) -> None:
        args, preflight_path, _ = self.make_single_matrix("appearing-process")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        gpus = preflight["gpus"]
        process = [{"gpu_uuid": "GPU-0", "pid": 12}]
        with mock.patch.object(
            terminal, "nvidia_hardware", return_value=(gpus, process, None)
        ):
            state = terminal.current_cuda_admission(
                gpu_index=0,
                required_bytes=preflight["gpu_required_available_bytes"],
                expected_uuid=preflight["selected_gpu_uuid"],
            )
        self.assertFalse(state["admitted"])

    def run_short_case_campaign(self, name: str) -> dict:
        """Run the wrapper over a stand-in child whose one E7 case lasts 120 ms, while every
        nvidia-smi query takes 1.2 s (a WDDM host), and return the sealed receipt."""
        child = self.root / f"{name}-child.py"
        child.write_text(
            "import json, os, time\n"
            "time.sleep(0.4)\n"
            "started = time.time()\n"
            "time.sleep(0.12)\n"
            "ended = time.time()\n"
            "time.sleep(1.0)\n"
            "provider = {'status': 'completed', 'model_id': os.environ['BONSAI_COMPARISON_MODEL_ID'],\n"
            "    'cases': [{'case_id': 'image', 'status': 'completed',\n"
            "               'request_peak_claim_eligible': True,\n"
            "               'measurement_interval': {\n"
            "                   'run_id': os.environ['BONSAI_COMPARISON_RUN_ID'],\n"
            "                   'process_id': os.getpid(), 'kind': 'selected_case',\n"
            "                   'started_unix_seconds': started, 'ended_unix_seconds': ended}}]}\n"
            "with open(os.environ['BONSAI_COMPARISON_OUTPUT'], 'x', encoding='utf-8') as out:\n"
            "    json.dump(provider, out)\n",
            encoding="utf-8",
        )
        preflight_path = self.root / f"{name}-preflight.json"
        preflight_path.write_text('{"load_profile":"candle-dense-cpu"}', encoding="utf-8")
        output = self.root / name
        # The wrapper runs `<binary> <test-name> --exact ...`: the interpreter runs the stand-in.
        args = terminal.parser().parse_args(
            ["run", "--binary", sys.executable, "--test-name", str(child),
             "--model-id", "parent", "--model-key", "parent", "--model-revision", "c" * 40,
             "--snapshot", str(self.root), "--model-path", str(self.root / "weights.gguf"),
             "--runtime-sha", self.runtime_sha, "--preflight", str(preflight_path),
             "--cases", "image", "--output", str(output), "--candle-device", "cpu"]
        )
        gpu_queries: list[float] = []

        def slow_nvidia_smi(_pid: int) -> tuple[None, str]:
            gpu_queries.append(time.monotonic())
            time.sleep(1.2)
            return None, "per-process GPU memory unavailable (WDDM or unsupported driver)"

        inventory = {"inventory_sha256": "e" * 64}
        sizes = {"language_weight_bytes": 1, "vision_weight_bytes": 0, "projector_bytes": 0}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(terminal, "source_identity", return_value={}))
            stack.enter_context(mock.patch.object(terminal, "load_model", return_value={"revision": "c" * 40, "key": "parent"}))
            stack.enter_context(mock.patch.object(terminal, "validate_variant_paths"))
            stack.enter_context(mock.patch.object(terminal, "verify_snapshot"))
            stack.enter_context(mock.patch.object(terminal, "snapshot_inventory", return_value=inventory))
            stack.enter_context(mock.patch.object(terminal, "artifact_sizes", return_value=sizes))
            stack.enter_context(mock.patch.object(terminal, "pinned_admission_sizes", return_value=sizes))
            stack.enter_context(mock.patch.object(terminal, "validate_preflight_record"))
            stack.enter_context(mock.patch.object(terminal, "selected_artifact", return_value={"path": "weights.gguf"}))
            stack.enter_context(mock.patch.object(terminal, "nvidia_sample", side_effect=slow_nvidia_smi))
            stack.enter_context(mock.patch.object(terminal, "physical_memory", return_value=(1, 1, None)))
            terminal.run(args)
        self.assertTrue(gpu_queries, "the stand-in nvidia-smi was never queried")
        return json.loads((output / "receipt.json").read_text(encoding="utf-8"))

    def test_short_case_gets_in_window_rss_despite_a_slow_nvidia_smi(self) -> None:
        """sc-24164: release gate 2 failed "required E7 case image lacks bounded memory evidence"
        because RSS and nvidia-smi shared one loop, so on Windows RSS was sampled well under 1 Hz and
        the E7 image case (fast since sc-24128) finished between two samples. RSS now has its own
        fast loop; however slow nvidia-smi is, a 120 ms case holds complete in-window samples."""
        receipt = self.run_short_case_campaign("short-case")
        process = receipt["process"]
        self.assertEqual(process["exit_code"], 0)
        self.assertEqual(
            process["rss_sample_interval_seconds"], terminal.RSS_SAMPLE_INTERVAL_SECONDS
        )
        (image,) = receipt["case_memory"]
        self.assertEqual(image["case_id"], "image")
        rss = image["rss"]
        self.assertTrue(
            rss["available"], f"no complete RSS sample inside the 120 ms case: {rss}"
        )
        self.assertGreaterEqual(len(rss["samples"]), 1)
        terminal.validate_sample_set(
            rss,
            image["measurement_interval"],
            run_id=process["run_id"],
            process_id=process["process_id"],
            allowed_scopes={
                "sampled_process_working_set_within_selected_request_lower_bound"
            },
            root=self.root,
        )
        # The slow query ran beside the RSS loop: it produced no GPU sample, only its reason.
        self.assertFalse(receipt["gpu"]["available"])
        self.assertIn("WDDM", receipt["gpu"]["unavailable_reason"])
        # The fast stream is thinned for the receipt without dropping the evidence.
        self.assertLessEqual(len(process["rss_samples"]), process["rss_samples_collected"])
        self.assertEqual(
            process["peak_rss_bytes"], max(sample["bytes"] for sample in process["rss_samples"])
        )

    def test_rss_sample_interval_is_bounded_to_the_fast_loop(self) -> None:
        self.assertEqual(terminal.rss_sample_interval("0.02"), 0.02)
        self.assertEqual(terminal.rss_sample_interval("0.05"), 0.05)
        for value in ("0", "-0.01", "0.06", "1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                terminal.rss_sample_interval(value)

    def test_retained_rss_keeps_every_sample_the_evidence_depends_on(self) -> None:
        def sample(index: int) -> dict:
            started = 100.0 + index * 0.02
            return {
                "seconds": index * 0.02,
                "started_unix_seconds": started,
                "ended_unix_seconds": started + 0.001,
                "run_id": "r",
                "process_id": 7,
                "bytes": 1_000 + (index * 7919) % 97,
            }

        samples = [sample(index) for index in range(5_000)]  # 100 s at 50 Hz
        samples[1_234]["bytes"] = 10**9  # the run's peak
        samples[2_001]["bytes"] = 10**6  # a peak inside the long case, off the thinning grid
        short = {"started_unix_seconds": 150.0005, "ended_unix_seconds": 150.045}
        long = {"started_unix_seconds": 130.0, "ended_unix_seconds": 145.0}
        between = {"started_unix_seconds": 160.0015, "ended_unix_seconds": 160.019}
        retained = terminal.retained_rss_samples(
            samples, [short, long, between], spacing=0.1
        )
        self.assertLess(len(retained), len(samples) // 4)
        self.assertEqual(
            [s["started_unix_seconds"] for s in retained],
            sorted(s["started_unix_seconds"] for s in retained),
        )
        self.assertEqual(max(s["bytes"] for s in retained), 10**9)

        def evidence(stream: list, interval: dict) -> dict:
            return terminal.bounded_interval_samples(
                stream, interval, run_id="r", process_id=7, scope="s", unavailable_reason="none"
            )

        for interval in (short, long):
            full, kept = evidence(samples, interval), evidence(retained, interval)
            self.assertTrue(kept["available"])
            for field in ("first_sample_bytes", "peak_bytes", "observed_growth_from_first_sample_bytes"):
                self.assertEqual(kept[field], full[field], field)
        self.assertEqual(evidence(retained, long)["peak_bytes"], 10**6)
        # A window no sample fits in stays unavailable, with its bracketing samples kept.
        full, kept = evidence(samples, between), evidence(retained, between)
        self.assertFalse(kept["available"])
        self.assertEqual(kept["bracketing_samples"], full["bracketing_samples"])
        self.assertIsNotNone(kept["bracketing_samples"]["before"])
        self.assertIsNotNone(kept["bracketing_samples"]["after"])

    def test_unbracketed_e7_case_records_its_neighbours_and_still_fails_closed(self) -> None:
        run = self.make_run("bracketed-e7-sample")
        self.add_context_decline(run)
        receipt_path = run / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        context = next(
            row for row in receipt["case_memory"] if row["case_id"] == "context_64"
        )
        interval = context["measurement_interval"]
        identity = {
            "run_id": receipt["process"]["run_id"],
            "process_id": receipt["process"]["process_id"],
        }
        before = {
            **identity,
            "started_unix_seconds": interval["started_unix_seconds"] - 0.5,
            "ended_unix_seconds": interval["started_unix_seconds"] - 0.4,
            "bytes": 11,
        }
        after = {
            **identity,
            "started_unix_seconds": interval["ended_unix_seconds"] + 0.4,
            "ended_unix_seconds": interval["ended_unix_seconds"] + 0.5,
            "bytes": 13,
        }
        context["rss"] = terminal.bounded_interval_samples(
            [before, after],
            interval,
            scope="sampled_process_working_set_within_selected_request_lower_bound",
            unavailable_reason="no complete RSS sample fell within the selected case interval",
            **identity,
        )
        self.assertFalse(context["rss"]["available"])
        self.assertEqual(context["rss"]["samples"], [])
        self.assertEqual(context["rss"]["bracketing_samples"], {"before": before, "after": after})
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.reseal_run(run)
        # Neighbouring samples are diagnostics, never in-window evidence: the gate fails closed.
        with self.assertRaisesRegex(ValueError, "required E7 case"):
            terminal.validate_receipt(run)


if __name__ == "__main__":
    unittest.main()
