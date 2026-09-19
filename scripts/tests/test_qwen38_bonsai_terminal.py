"""Fail-closed tests for the SC-23942 evidence wrapper and validator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import tempfile
import unittest
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
        provider = {
            "model_id": name,
            "model_revision": revision,
            "runtime_sha": runtime_sha,
            "status": "completed",
            "provider": {"backend": backend},
            "context_admission": {"evidence_complete": True},
            "native_memory_before_load": {
                "backend": backend,
                "device": device,
                "peak_active_bytes": 1,
                "native_allocator_counters_available": False,
                "memory_evidence": "external samples",
            },
            "native_memory_after_load": {
                "backend": backend,
                "device": device,
                "peak_active_bytes": 1,
                "native_allocator_counters_available": False,
                "memory_evidence": "external samples",
            },
            "native_memory_after_unload": {
                "backend": backend,
                "device": device,
                "peak_active_bytes": 1,
                "native_allocator_counters_available": False,
                "memory_evidence": "external samples",
            },
            "cases": [
                {
                    "case_id": "arithmetic",
                    "category": "math",
                    "request": request,
                    "status": "completed",
                    "evidence_complete": complete,
                    "quality_passed": quality,
                    "prefill_seconds": 0.004 if complete else None,
                    "decode_seconds": 0.002 if complete else None,
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
                "selected_model_artifact": {"sha256": "d" * 64},
                "selected_projector_artifact": {"sha256": "e" * 64},
                "inventory_before": inventory,
                "inventory_after": inventory,
                "artifact_sizes": {
                    "language_weight_bytes": 100,
                    "vision_weight_bytes": 20,
                    "auxiliary_bytes": 0,
                },
            },
            "process": {"exit_code": 0, "peak_rss_bytes": 4096},
            "gpu": {
                "available": False,
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
        sizes = terminal.artifact_sizes(model, None)
        self.assertEqual(sizes["language_weight_bytes"], 8)
        self.assertEqual(sizes["vision_weight_bytes"], 12)

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


if __name__ == "__main__":
    unittest.main()
