"""NEVER MERGE: regress the Windows PID proof used by the bounded probe."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
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


if __name__ == "__main__":
    unittest.main()
