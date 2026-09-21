"""NEVER MERGE: regress the Windows PID proof used by the bounded probe."""

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "sc23935_bounded_probe.py"
SPEC = importlib.util.spec_from_file_location("sc23935_bounded_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProcessExitProofTests(unittest.TestCase):
    def test_windows_wait_requests_synchronize_access(self) -> None:
        kernel = SimpleNamespace(
            OpenProcess=Mock(return_value=123),
            WaitForSingleObject=Mock(return_value=0),
            CloseHandle=Mock(),
            GetLastError=Mock(),
        )
        with patch.object(probe.ctypes, "windll", SimpleNamespace(kernel32=kernel), create=True):
            self.assertTrue(probe.process_exited(27036))
        kernel.OpenProcess.assert_called_once_with(0x00101000, 0, 27036)
        kernel.WaitForSingleObject.assert_called_once_with(123, 0)
        kernel.CloseHandle.assert_called_once_with(123)

    def test_query_failure_is_not_absence(self) -> None:
        kernel = SimpleNamespace(
            OpenProcess=Mock(return_value=0),
            WaitForSingleObject=Mock(),
            CloseHandle=Mock(),
            GetLastError=Mock(return_value=5),
        )
        with patch.object(probe.ctypes, "windll", SimpleNamespace(kernel32=kernel), create=True):
            self.assertFalse(probe.process_exited(27036))
        kernel.GetLastError.return_value = 87
        with patch.object(probe.ctypes, "windll", SimpleNamespace(kernel32=kernel), create=True):
            self.assertTrue(probe.process_exited(27036))

    def test_completed_wrapper_uses_held_process_handle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = root / "cpu-parent"
            record = root / "cpu-launch.json"
            wrapper = Mock(pid=33712)
            wrapper.wait.return_value = 0
            wrapper.poll.return_value = 0
            argv = ["probe", "--row", str(row), "--record", str(record),
                    "--timeout-seconds", "900", "--", "python", "wrapper.py"]
            with (patch.object(sys, "argv", argv),
                  patch.object(probe.sys, "platform", "win32"),
                  patch.object(probe.subprocess, "Popen", return_value=wrapper),
                  patch.object(probe, "native_pids", return_value={27036}),
                  patch.object(probe, "process_exited", return_value=True) as exited):
                self.assertEqual(probe.main(), 0)
            proof = json.loads(record.read_text(encoding="utf-8"))
            self.assertTrue(proof["wrapper_exited"])
            self.assertTrue(proof["native_exited"])
            self.assertTrue(proof["safe_to_continue"])
            exited.assert_called_once_with(27036)


class ContextDiagnosticValidationTests(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_evidence(self, root: Path, *, timed_out: bool) -> None:
        sha = "a" * 40
        row = root / "cpu-parent"
        row.mkdir()
        self.write_json(root / "provenance.json", {
            "checked_out_sha": sha, "clean_tree": True,
            "rc4_base_sha": "7b7730a1a06a231ce337352dbc5b525eb3f8cc78",
        })
        self.write_json(root / "snapshot-metadata.json", {"all_metadata_qualified": True})
        self.write_json(root / "gpu-reservation.json", {
            "gpu_index": 0, "gpu_uuid": "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
        })
        self.write_json(root / "gpu-recheck.json", {
            "selected_gpu_uuid": "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
            "selected_gpu_compute_processes": [],
        })
        self.write_json(root / "cpu-launch.json", {
            "safe_to_continue": True, "native_pids": [137], "timeout_seconds": 1500,
            "command": ["python", "wrapper.py", "--cases", "context_64",
                        "--diagnostic-timeout-seconds", "1200"],
            "returncode": 1 if timed_out else 0,
        })
        self.write_json(row / "receipt.json", {
            "status": "timed_out" if timed_out else "completed",
            "runtime": {"head_sha": sha},
            "model": {
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "id": "candle-cpu-qwen38-parent-context64-probe",
                "inventory_before": {"files": [{"path": "weights.safetensors"}], "inventory_sha256": "frozen"},
                "inventory_after": {"inventory_sha256": None if timed_out else "frozen"},
            },
            "command": {"candle_device": "cpu", "load_profile": "candle-dense-cpu"},
            "process": {
                "process_id": 137, "run_id": "one-run", "diagnostic_timeout_seconds": 1200,
                "exit_code": 1 if timed_out else 0,
                "child_tree_cleanup": {
                    "root_reaped": True, "tree_termination_requested": True,
                } if timed_out else None,
            },
        })
        (row / "progress-rss.jsonl").write_text('{"process_id":137,"bytes":100}\n', encoding="utf-8")
        (row / "stderr.log").write_text(json.dumps({
            "kind": "comparison_stage_v1", "stage": "load", "event": "start",
            "elapsed_seconds": 1, "run_id": "one-run", "process_id": 137,
        }) + "\n", encoding="utf-8")
        if not timed_out:
            self.write_json(row / "provider.json", {
                "status": "completed", "case_ids": ["context_64"],
                "cases": [{
                    "case_id": "context_64", "oracle": {"kind": "exact", "value": "NEBULA-47"},
                    "status": "completed", "evidence_complete": True,
                    "stream_contract_passed": True, "quality_passed": True,
                    "output": {"text": " NEBULA-47\n"},
                }],
            })
        files = [path for path in row.iterdir() if path.is_file() and path.name != "receipt.json"]
        self.write_json(row / "artifact-manifest.json", {
            "files": [{
                "path": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            } for path in files],
        })
        self.write_json(row / "seal.json", {
            "receipt_sha256": hashlib.sha256((row / "receipt.json").read_bytes()).hexdigest(),
            "artifact_manifest_sha256": hashlib.sha256((row / "artifact-manifest.json").read_bytes()).hexdigest(),
        })

    def validate(self, root: Path) -> dict:
        argv = ["probe", "validate", "--root", str(root), "--runtime-sha", "a" * 40]
        with (patch.object(sys, "argv", argv),
              patch.dict(probe.os.environ, {"SC23935_GPU_RESERVATION": str(root / "absent.json")}),
              redirect_stdout(io.StringIO())):
            result = probe.validate_main()
        verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0 if verification["diagnostic_valid"] else 1)
        return verification

    def test_completed_exact_context_case_is_diagnostic_and_case_pass(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_evidence(root, timed_out=False)
            result = self.validate(root)
            self.assertTrue(result["diagnostic_valid"])
            self.assertTrue(result["context_case_passed"])
            self.assertFalse(result["full_campaign_accepted"])

    def test_native_timeout_is_valid_diagnostic_but_not_case_pass(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_evidence(root, timed_out=True)
            result = self.validate(root)
            self.assertTrue(result["diagnostic_valid"])
            self.assertFalse(result["context_case_passed"])
            self.assertEqual(result["cpu_last_stage"], "load")

    def test_other_case_or_oracle_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_evidence(root, timed_out=False)
            provider_path = root / "cpu-parent/provider.json"
            provider = json.loads(provider_path.read_text(encoding="utf-8"))
            provider["cases"][0]["oracle"]["value"] = "different"
            self.write_json(provider_path, provider)
            manifest_path = root / "cpu-parent/artifact-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provider_entry = next(item for item in manifest["files"] if item["path"] == "provider.json")
            provider_entry["bytes"] = provider_path.stat().st_size
            provider_entry["sha256"] = hashlib.sha256(provider_path.read_bytes()).hexdigest()
            self.write_json(manifest_path, manifest)
            seal_path = root / "cpu-parent/seal.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.write_json(seal_path, seal)
            result = self.validate(root)
            self.assertFalse(result["diagnostic_valid"])
            self.assertFalse(result["context_case_passed"])
            self.assertIn("CPU context oracle changed", result["errors"])

    def test_duplicate_case_option_is_not_accepted(self) -> None:
        self.assertIsNone(probe.command_option(["--cases", "context_64", "--cases", "arithmetic"], "--cases"))


if __name__ == "__main__":
    unittest.main()
