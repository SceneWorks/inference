"""CPU checks for S0 admission and owned-child return/timeout handling."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

_spec = importlib.util.spec_from_file_location("veda_s0_cuda", Path(__file__).resolve().parents[1] / "release" / "veda_s0_cuda.py")
s0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s0)


class S0CudaTests(unittest.TestCase):
    def test_atomic_partial_receipt_persists_completed_phase(self):
        with tempfile.TemporaryDirectory() as out:
            output=Path(out)
            s0.persist_receipt(output,dict(complete=False,status="random_complete",results=[dict(returncode=0)]))
            import json
            self.assertEqual(json.loads((output/"receipt.json").read_text())["status"],"random_complete")
            self.assertFalse((output/"receipt.json.tmp").exists())

    def test_idle_requires_no_process_low_memory_and_low_utilization(self):
        device = dict(uuid="GPU-test", used_mib=100, utilization_percent=0)
        self.assertTrue(s0.idle(device, []))
        self.assertFalse(s0.idle(device, [["GPU-test", "123", "other"]]))
        self.assertFalse(s0.idle(dict(device, used_mib=2048), []))
        self.assertFalse(s0.idle(dict(device, utilization_percent=5), []))

    def test_graphics_only_wddm_baseline_is_reserved_but_compute_is_refused(self):
        desktop = dict(uuid="GPU-desktop",used_mib=1164,utilization_percent=0)
        self.assertTrue(s0.idle(desktop,[["GPU-desktop","1","[Insufficient Permissions]","G"]]))
        for kind in ("C","C+G","unknown",""):
            self.assertFalse(s0.idle(desktop,[["GPU-desktop","1","desktop-name-is-not-authority",kind]]))
        self.assertFalse(s0.idle(dict(desktop,used_mib=95244),[["GPU-desktop","1","graphics","G"]]))

    def test_typed_driver_inventory_never_guesses_process_type_from_name(self):
        xml="""<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes>
            <process_info><pid>1</pid><process_name>[Insufficient Permissions]</process_name><type>G</type></process_info>
            <process_info><pid>2</pid><process_name>explorer.exe</process_name><type>C+G</type></process_info>
            <process_info><pid>3</pid><process_name>ShellHost.exe</process_name></process_info>
        </processes></gpu></nvidia_smi_log>"""
        rows=s0.typed_processes(xml)
        self.assertEqual([r[3] for r in rows],["G","C+G","unknown"])
        self.assertFalse(s0.idle(dict(uuid="GPU-test",used_mib=1164,utilization_percent=0),rows))

    def test_owned_child_preserves_failure_returncode(self):
        job = Mock()
        child = Mock()
        child.wait.return_value = 17
        with tempfile.TemporaryDirectory() as out, patch.object(s0.subprocess, "Popen", return_value=child) as popen:
            result = s0.execute_owned("test", ["binary"], Path(out), time.monotonic()+30, lambda: job)
        self.assertEqual(result["returncode"], 17)
        self.assertFalse(result["timeout"])
        self.assertEqual(popen.call_args.kwargs["creationflags"], 4)
        job.assign_and_resume.assert_called_once_with(child)
        job.close.assert_called_once()

    def test_owned_child_timeout_terminates_job_and_is_not_success(self):
        job = Mock()
        child = Mock()
        child.wait.side_effect = subprocess.TimeoutExpired("binary", 1)
        with tempfile.TemporaryDirectory() as out, patch.object(s0.subprocess, "Popen", return_value=child):
            result = s0.execute_owned("test", ["binary"], Path(out), time.monotonic()+30, lambda: job)
        self.assertIsNone(result["returncode"])
        self.assertTrue(result["timeout"])
        job.terminate_and_reap.assert_called_once_with(child, 10)
        job.close.assert_called_once()

    def test_failed_job_assignment_kills_suspended_child(self):
        job = Mock()
        job.assign_and_resume.side_effect = RuntimeError("cannot own")
        child = Mock()
        with tempfile.TemporaryDirectory() as out, patch.object(s0.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "cannot own"):
                s0.execute_owned("test", ["binary"], Path(out), time.monotonic()+30, lambda: job)
        child.kill.assert_called_once()
        job.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
