"""Sanitized diagnostic contract: no argv/environment or model/acceptance work."""
import importlib.util
from pathlib import Path
import plistlib
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen21_host", ROOT / "scripts/ci/qwen21_host_census.py")
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


class HostCensusTests(unittest.TestCase):
    def test_only_requested_process_fields_and_rss_order(self):
        self.assertEqual(HOST.PS_FIELDS, "pid=,ppid=,uid=,rss=,comm=")
        rows = HOST.process_rows(b"12 1 501 200 /Applications/Some App\n4 1 0 900 kernel_task\n")
        self.assertEqual(rows[0], {"pid": 4, "ppid": 1, "uid": 0, "rssKiB": 900, "comm": "kernel_task"})
        self.assertEqual(rows[1]["comm"], "/Applications/Some App")
        with self.assertRaises(ValueError):
            HOST.process_rows(b"12 1 invalid 200 command\n")

    def test_driver_output_is_numeric_allowlist_only(self):
        raw = plistlib.dumps([{"SecretProperty": "credential", "PerformanceStatistics": {
            "In use system memory": 17, "Texture memory": 1.5, "Buffer memory": True,
            "Device memory used": -1, "System memory used": "secret",
            "unknown": 123, "Alloc system memory": float("inf"),
        }}])
        self.assertEqual(HOST.driver_rows(raw), [{"In use system memory": 17, "Texture memory": 1.5}])

    def test_missing_and_malformed_diagnostics_are_non_acceptance(self):
        commands = []
        def query(command):
            commands.append(command)
            if command[0] == "/bin/ps":
                return b"1 0 0 10 kernel_task\n", None
            if command[0] == "/usr/sbin/ioreg":
                return plistlib.dumps([]), None
            return None, "query unavailable"
        with patch.object(HOST, "query", query):
            receipt = HOST.collect()
        self.assertEqual(receipt["kind"], "HOST_DIAGNOSTIC_ONLY")
        self.assertIs(receipt["acceptanceEvidence"], False)
        self.assertEqual(receipt["processFields"], ["PID", "PPID", "UID", "RSS", "comm"])
        self.assertEqual(commands[0], ["/bin/ps", "-axo", "pid=,ppid=,uid=,rss=,comm="])
        self.assertEqual(len(commands), 6)
        self.assertIn("driverMetrics", receipt["errors"])
        self.assertNotIn("rawDriverRegistry", receipt)
        with patch.object(HOST, "query", return_value=(b"malformed", None)):
            receipt = HOST.collect()
        self.assertIn("processes", receipt["errors"])

    def test_query_bounds_time_output_and_never_retains_stderr(self):
        def run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 5)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            kwargs["stdout"].write(b"x" * (HOST.MAX_QUERY_BYTES + 1))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(HOST.subprocess, "run", run):
            self.assertEqual(HOST.query(["ps"]), (None, "query exceeded 2 MiB output bound"))
        with patch.object(HOST.subprocess, "run", side_effect=subprocess.TimeoutExpired("ps", 5)):
            self.assertEqual(HOST.query(["ps"]), (None, "TimeoutExpired"))

    def test_workflow_collects_before_native_build_and_weight_work(self):
        workflow = (ROOT / ".github/workflows/real-weights.yml").read_text(encoding="utf-8")
        workflow = workflow.split("  mlx-qwen-image-2-1:", 1)[1].split("\n  mlx-", 1)[0]
        command = 'python3.12 scripts/ci/qwen21_host_census.py --out "$QWEN_IMAGE_2_1_RENDER_OUT/host-census"'
        self.assertEqual(workflow.count(command), 1)
        start = workflow.index(command)
        for step in ["Build the Qwen-Image 2.1 MLX test binary", "Prove trained velocity survives adapter save and reload",
                     "Materialize and verify immutable snapshots", "Run the Qwen-Image 2.1 LoRA/LoKr real-weight gates"]:
            self.assertLess(start, workflow.index("- name: " + step))


if __name__ == "__main__":
    unittest.main()
