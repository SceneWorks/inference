"""Fail-closed tests for the SC-23942 evidence wrapper and validator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "release" / "qwen38_bonsai_terminal.py"
SPEC = importlib.util.spec_from_file_location("qwen38_bonsai_terminal", SCRIPT)
assert SPEC and SPEC.loader
terminal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(terminal)


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

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
    ) -> Path:
        root = self.root / name
        root.mkdir()
        request = request or {"max_new_tokens": 128, "sampling": {"temperature": 0.0}}
        provider = {
            "status": "completed",
            "context_admission": {"evidence_complete": True},
            "native_memory_before_load": {
                "backend": "candle",
                "native_allocator_counters_available": False,
                "memory_evidence": "external samples",
            },
            "native_memory_after_load": {
                "backend": "candle",
                "native_allocator_counters_available": False,
                "memory_evidence": "external samples",
            },
            "native_memory_after_unload": {
                "backend": "candle",
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
            "runtime": {"head_sha": "b" * 40, "clean_tree": True},
            "model": {
                "id": name,
                "manifest_key": name,
                "revision": "c" * 40,
                "language_variant": None,
                "vision_variant": None,
                "selected_model_artifact": {"sha256": "d" * 64},
                "selected_projector_artifact": {"sha256": "e" * 64},
                "inventory_before": inventory,
                "inventory_after": inventory,
                "artifact_sizes": {
                    "language_weight_bytes": 100,
                    "vision_weight_bytes": 20,
                    "auxiliary_bytes": 5,
                },
            },
            "process": {"exit_code": 0, "peak_rss_bytes": 4096},
            "gpu": {
                "available": False,
                "peak_bytes": None,
                "unavailable_reason": "WDDM counter unavailable",
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

    def validate(self, runs: list[Path]) -> dict:
        output = self.root / "report.json"
        markdown = self.root / "report.md"
        args = argparse.Namespace(
            run=runs,
            expected_model=[path.name for path in runs],
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
            output=self.root / "out.json",
            markdown=self.root / "out.md",
        )
        with self.assertRaisesRegex(ValueError, "unequal requests or budgets"):
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
        terminal.write_new(
            self.root / "rejected-preflight.json",
            {
                "model_key": "two",
                "load_profile": "candle-dense-cuda",
                "admitted": False,
            },
        )
        args = argparse.Namespace(
            root=[self.root],
            matrix=matrix,
            output=self.root / "matrix-report.json",
            markdown=self.root / "matrix-report.md",
        )
        self.assertEqual(terminal.matrix_status(args), 1)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertFalse(report["evidence_complete"])
        self.assertEqual(
            [row["status"] for row in report["required_cells"]],
            ["incomplete", "not_admitted"],
        )

    def test_matrix_status_requires_and_compares_every_declared_cell(self) -> None:
        cells = []
        for name, profile in (("mlx-parent", "mlx-unified"), ("candle-parent", "candle-dense-cpu")):
            self.make_run(name)
            cell = {
                "id": name,
                "group": "matched",
                "backend": name.split("-", 1)[0],
                "device": "unified" if name.startswith("mlx") else "cpu",
                "model_key": name,
                "load_profile": profile,
            }
            cells.append(cell)
            terminal.write_new(
                self.root / f"{name}-preflight.json",
                {"model_key": name, "load_profile": profile, "admitted": True},
            )
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
            output=self.root / "complete-matrix-report.json",
            markdown=self.root / "complete-matrix-report.md",
        )
        self.assertEqual(terminal.matrix_status(args), 0)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertTrue(report["evidence_complete"])
        self.assertTrue(report["comparison"]["matched"]["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
