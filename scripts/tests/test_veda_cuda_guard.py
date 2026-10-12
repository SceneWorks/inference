"""CPU checks for preserved S0 admission and owned-child return/timeout handling."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

_spec = importlib.util.spec_from_file_location("veda_cuda_guard", Path(__file__).resolve().parents[1] / "release" / "veda_cuda_guard.py")
s0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s0)


class VedaCudaGuardTests(unittest.TestCase):

    def test_expired_inventory_deadline_does_not_launch_driver_probe(self):
        with patch.object(s0.subprocess, "check_output") as probe:
            with self.assertRaisesRegex(TimeoutError, "sampling deadline"):
                s0.command(["nvidia-smi"], deadline=time.monotonic() - 1)
            probe.assert_not_called()

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

    def test_wddm_process_types_require_stable_sampling(self):
        desktop = dict(uuid="GPU-desktop",used_mib=1164,utilization_percent=0)
        self.assertTrue(s0.idle(desktop,[["GPU-desktop","1","[Insufficient Permissions]","G"]]))
        self.assertTrue(s0.idle(desktop,[["GPU-desktop","1","[Insufficient Permissions]","C+G"]]))
        for kind in ("C","unknown",""):
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


    def test_driver_uuid_pid_mismatch_and_untyped_arrival_are_refused(self):
        gpu_csv = "0, GPU-a, A, 596.36, 97887, 1176, 0\n1, GPU-b, B, 596.36, 97887, 95244, 0"
        xml = """<nvidia_smi_log><gpu><uuid>GPU-a</uuid><processes>
            <process_info><pid>1</pid><process_name>unknown</process_name><type>C+G</type></process_info>
            </processes></gpu><gpu><uuid>GPU-b</uuid><processes/></gpu></nvidia_smi_log>"""
        with patch.object(s0, "command", side_effect=[gpu_csv, xml, "GPU-b, 1, opaque"]):
            with self.assertRaisesRegex(RuntimeError, "UUID/PID differ"):
                s0.inventory()
        with patch.object(s0, "command", side_effect=[gpu_csv, xml, "GPU-a, 2, opaque"]):
            devices, processes = s0.inventory()
        self.assertFalse(s0.idle(devices[0], processes))
        self.assertIn(["GPU-a", "2", "opaque", "unknown"], processes)

    def test_stable_baseline_and_phase_recheck_record_five_typed_snapshots(self):
        device = dict(uuid="GPU-test", used_mib=1164, utilization_percent=0)
        processes = [["GPU-test", "1", "permission-opaque", "C+G"]]
        with tempfile.TemporaryDirectory() as out, patch.object(s0, "inventory", return_value=([device], processes)) as inventory, patch.object(s0.time, "sleep"):
            checked, baseline = s0.sample_baseline(device, Path(out), "claimed-device")
            s0.verify_idle(device, Path(out), "random", baseline)
            self.assertEqual(checked, device)
            self.assertEqual(inventory.call_count, 10)
            self.assertEqual(baseline["process_identities"], [("GPU-test", "1", "C+G")])
            self.assertEqual(len(json.loads((Path(out) / "random-samples.json").read_text())), 5)

    def test_baseline_refuses_activity_identity_changes_and_rising_memory(self):
        device = dict(uuid="GPU-test", used_mib=1164, utilization_percent=0)
        row = [["GPU-test", "1", "opaque", "C+G"]]
        baseline = dict(uuid="GPU-test", reserved_baseline_mib=1164,
                        process_identities=[("GPU-test", "1", "C+G")])
        changed = [
            (dict(device, used_mib=1165), row),
            (dict(device, utilization_percent=5), row),
            (device, [["GPU-test", "1", "opaque", "C"]]),
            (device, [["GPU-test", "2", "opaque", "C+G"]]),
            (device, [["GPU-test", "1", "opaque", "unknown"]]),
            (dict(device, uuid="GPU-other"), row),
        ]
        for item in changed:
            with self.subTest(item=item), tempfile.TemporaryDirectory() as out, patch.object(s0, "inventory", side_effect=[([device], row), ([item[0]], item[1])]), patch.object(s0.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    s0.sample_baseline(device, Path(out), "random", baseline)
                self.assertEqual(len(json.loads((Path(out) / "random-samples.json").read_text())), 2)
        with tempfile.TemporaryDirectory() as out, patch.object(s0, "inventory", side_effect=[([dict(device, used_mib=1163)], row), ([device], row)]), patch.object(s0.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "baseline changed"):
                s0.sample_baseline(device, Path(out), "random", baseline)

    def test_claim_holds_host_global_uuid_lock_during_sampling(self):
        device = dict(index=1, uuid="GPU-test", used_mib=1164, utilization_percent=0)
        row = [["GPU-test", "1", "opaque", "C+G"]]
        fake_lock = Mock()
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), patch.dict(sys.modules, {"msvcrt": fake_lock}), patch.object(s0, "inventory", return_value=([device], row)), patch.object(s0.time, "sleep"):
            admitted, handle, baseline = s0.claim_device(Path(out))
            self.assertEqual(admitted["uuid"], device["uuid"])
            self.assertEqual(baseline["reserved_baseline_mib"], 1164)
            self.assertEqual(fake_lock.locking.call_count, 1)
            self.assertFalse(handle.closed)
            handle.close()

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
