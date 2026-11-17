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

    def test_non_ascii_inventory_and_receipts_are_utf8_with_lf(self):
        xml = "<nvidia_smi_log>\n<gpu><uuid>GPU-test</uuid><product_name>☃</product_name><processes/></gpu>\n</nvidia_smi_log>\n"
        csv = "0, GPU-test, GPU ☃, 596.36, 97887, 100, 97787, 0"
        with tempfile.TemporaryDirectory() as temp, patch.object(s0, "command", side_effect=[csv, xml, ""]):
            out = Path(temp)
            devices, _ = s0.inventory(out, "unicode")
            self.assertEqual(devices[0]["name"], "GPU ☃")
            self.assertEqual((out / "unicode-nvidia-smi.xml").read_bytes(), xml.encode("utf-8"))
            s0.persist_receipt(out, {"devices": devices})
            data = (out / "receipt.json").read_bytes()
            self.assertNotIn(b"\r\n", data)
            self.assertEqual(json.loads(data.decode("utf-8"))["devices"], devices)

    def test_expired_inventory_deadline_does_not_launch_driver_probe(self):
        with patch.object(s0.subprocess, "check_output") as probe:
            with self.assertRaisesRegex(TimeoutError, "sampling deadline"):
                s0.command(["nvidia-smi"], deadline=time.monotonic() - 1)
            probe.assert_not_called()

    def test_guard_vendor_is_exactly_the_authenticated_canonical_component(self):
        from scripts.ci import model_proof_cuda_primitives as primitives
        root = Path(__file__).resolve().parents[2]
        self.assertEqual((root / 'scripts/release/veda_cuda_guard.py').read_bytes(),
                         (root / 'scripts/ci/vendor/veda_cuda_guard.py').read_bytes())
        primitives.component('veda_cuda_guard.py', primitives.GUARD_SHA)

    def test_synced_vendor_preserves_legacy_positional_sampling_and_receipt_shape(self):
        from scripts.ci import model_proof_cuda_primitives as primitives
        vendor = primitives.component('veda_cuda_guard.py', primitives.GUARD_SHA)
        card = dict(uuid='GPU-test', used_mib=1164, utilization_percent=0)
        rows = [['GPU-test', '1', 'MOCK', 'C+G']]
        baseline = dict(uuid='GPU-test', reserved_baseline_mib=1164,
                        process_identities=[('GPU-test', '1', 'C+G')])
        with tempfile.TemporaryDirectory() as temporary, patch.object(vendor, 'inventory', return_value=([card], rows)), \
                patch.object(vendor.time, 'sleep'):
            checked, observed = vendor.sample_baseline(card, Path(temporary), 'legacy', baseline)
            vendor.verify_idle(card, Path(temporary), 'legacy-release', baseline)
            self.assertEqual(checked, card)
            self.assertEqual(observed, baseline)
            samples = json.loads((Path(temporary) / 'legacy-samples.json').read_text(encoding='utf-8'))
            self.assertEqual(len(samples), 5)
            self.assertEqual(set(samples[0]), {'sample_number', 'elapsed_seconds', 'devices', 'processes', 'selected_used_mib'})
            self.assertEqual(samples[0]['processes'], rows)
        with tempfile.TemporaryDirectory() as temporary, patch.object(vendor, 'inventory', return_value=(
                [dict(card, utilization_percent=5)], rows)), patch.object(vendor.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                vendor.sample_baseline(card, Path(temporary), 'legacy', baseline)

    def test_strict_inventory_still_requires_typed_census(self):
        row = "1, GPU-test, MOCK, 596.36, 97887, 100, 97787, 0"
        with patch.object(s0, 'command', side_effect=[row, TimeoutError('MOCK slow census')]) as probe:
            with self.assertRaisesRegex(TimeoutError, 'slow census'):
                s0.inventory(deadline=10)
            self.assertEqual(probe.call_args_list[1].args, (["nvidia-smi", "-q", "-x"], 10))

    def test_selected_inventory_never_runs_optional_host_process_census(self):
        uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
        row = f"1, {uuid}, GPU, 596.36, 97887, 5247, 92192, [N/A]"
        def driver(args, deadline):
            self.assertIn("--id=" + uuid, args)
            self.assertIn("--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", args)
            # A full XML/process census could exhaust the unchanged sampling window.
            # It must never be requested by this selected metadata reader.
            self.assertNotIn("-x", args)
            return row
        with patch.object(s0, "command", side_effect=driver) as probe:
            devices, processes = s0.inventory(deadline=10, selected_uuid=uuid)
        probe.assert_called_once()
        self.assertEqual(devices[0]["free_mib"], 92192)
        self.assertIsNone(devices[0]["utilization_percent"])
        self.assertEqual(devices[0]["process_inventory_status"], "unknown/not-collected-diagnostic-only")
        self.assertEqual(processes, [])

    def test_selected_inventory_refuses_missing_wrong_duplicate_or_invalid_primary_metadata(self):
        uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
        row = f"1, {uuid}, GPU, 596.36, 97887, 5247, 92192, 0"
        for response in ("", row.replace(uuid, "GPU-other"), row + "\n" + row,
                         row.replace("92192", "[N/A]"), row.replace("92192", "nan"),
                         row.replace("97887", "inf"), TimeoutError("MOCK primary deadline"),
                         subprocess.TimeoutExpired("nvidia-smi", 5)):
            with self.subTest(response=response), patch.object(s0, "command", side_effect=[response]):
                with self.assertRaises((ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired)):
                    s0.inventory(selected_uuid=uuid)
        with patch.object(s0, "command") as probe:
            with self.assertRaisesRegex(ValueError, "invalid selected"):
                s0.inventory(selected_uuid="1")
            probe.assert_not_called()

    def test_atomic_partial_receipt_persists_completed_phase(self):
        with tempfile.TemporaryDirectory() as out:
            output=Path(out)
            s0.persist_receipt(output,dict(complete=False,status="random_complete",results=[dict(returncode=0)]))
            import json
            self.assertEqual(json.loads((output/"receipt.json").read_text(encoding="utf-8"))["status"],"random_complete")
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
        gpu_csv = "0, GPU-a, A, 596.36, 97887, 1176, 96711, 0\n1, GPU-b, B, 596.36, 97887, 95244, 2643, 0"
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
            self.assertEqual(len(json.loads((Path(out) / "random-samples.json").read_text(encoding="utf-8"))), 5)

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
                self.assertEqual(len(json.loads((Path(out) / "random-samples.json").read_text(encoding="utf-8"))), 2)
        with tempfile.TemporaryDirectory() as out, patch.object(s0, "inventory", side_effect=[([dict(device, used_mib=1163)], row), ([device], row)]), patch.object(s0.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "baseline changed"):
                s0.sample_baseline(device, Path(out), "random", baseline)

    def test_claim_holds_host_global_uuid_lock_during_sampling(self):
        device = dict(index=1, uuid="GPU-test", total_mib=97887, used_mib=1164, free_mib=96723, utilization_percent=0)
        row = [["GPU-test", "1", "opaque", "C+G"]]
        for kwargs in (dict(required_free_mib=4096), dict(strict=True)):
            fake_lock = Mock()
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), \
                    patch.dict(sys.modules, {"msvcrt": fake_lock}), patch.object(s0, "inventory", return_value=([device], row)), \
                    patch.object(s0.time, "sleep"):
                admitted, handle, baseline = s0.claim_device(Path(out), **kwargs)
                self.assertEqual(admitted["uuid"], device["uuid"])
                self.assertEqual(baseline["reserved_baseline_mib"], 1164)
                self.assertEqual(fake_lock.locking.call_count, 1)
                self.assertFalse(handle.closed)
                handle.close()

    def test_default_claim_admits_on_headroom_despite_foreign_compute_display_and_resident_model(self):
        # The owner's desktop app holds a CUDA context and a loaded model; the card drives a display.
        quiet = dict(index=1, uuid="GPU-test", total_mib=97887, used_mib=40000, free_mib=57887, utilization_percent=12)
        rows = [["GPU-test", "7", "desktop-app", "C"], ["GPU-test", "8", "dwm", "C+G"], ["GPU-test", "9", "x", "unknown"]]
        self.assertFalse(s0.idle(quiet, rows))
        self.assertTrue(s0.has_headroom(quiet, 4096))
        jitter = [dict(quiet, used_mib=40000 + delta, free_mib=57887 - delta, utilization_percent=util)
                  for delta, util in ((0, 12), (200, 25), (40, 10), (120, 20), (100, 15))]
        fake_lock = Mock()
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), \
                patch.dict(sys.modules, {"msvcrt": fake_lock}), \
                patch.object(s0, "inventory", side_effect=[([quiet], rows)] + [([card], rows) for card in jitter]), \
                patch.object(s0.time, "sleep"):
            admitted, handle, _ = s0.claim_device(Path(out), 4096)
            self.assertEqual(admitted["uuid"], "GPU-test")
            handle.close()
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), \
                patch.dict(sys.modules, {"msvcrt": Mock()}), patch.object(s0, "inventory", return_value=([quiet], rows)):
            with self.assertRaisesRegex(RuntimeError, "no unclaimed idle"):
                s0.claim_device(Path(out), strict=True)

    def test_default_claim_requires_a_quiet_stable_window(self):
        card = dict(index=1, uuid="GPU-test", total_mib=97887, used_mib=20000, free_mib=77887, utilization_percent=5)
        cases = {
            "utilization 31%": [dict(card, utilization_percent=31)],
            "utilization moved 16 points": [dict(card, utilization_percent=5), dict(card, utilization_percent=21)],
            "memory.used moved 257 MiB": [card, dict(card, used_mib=20257, free_mib=77630)],
            "memory.used grew 129 MiB": [card] * 4 + [dict(card, used_mib=20129, free_mib=77758)],
        }
        for reason, window in cases.items():
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as out, \
                    patch.dict(os.environ, {"PUBLIC": out}), patch.dict(sys.modules, {"msvcrt": Mock()}), \
                    patch.object(s0, "inventory", side_effect=[([card], [])] + [([item], []) for item in window]), \
                    patch.object(s0.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, reason):
                    s0.claim_device(Path(out), 4096)

    def test_default_claim_refuses_only_insufficient_or_unstable_headroom_with_reason(self):
        card = dict(index=1, uuid="GPU-test", total_mib=97887, used_mib=1164, free_mib=96723, utilization_percent=0)
        short = dict(card, used_mib=95000, free_mib=2887)
        for value in (short, dict(card, free_mib=None), dict(card, used_mib=99999)):
            self.assertFalse(s0.has_headroom(value, 4096))
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), \
                patch.dict(sys.modules, {"msvcrt": Mock()}), patch.object(s0, "inventory", return_value=([short], [])):
            with self.assertRaisesRegex(RuntimeError, "2887 MiB free < 4096 MiB required"):
                s0.claim_device(Path(out), 4096)
        # Headroom lost during the five locked samples releases the lock and refuses.
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}), \
                patch.dict(sys.modules, {"msvcrt": Mock()}), \
                patch.object(s0, "inventory", side_effect=[([card], []), ([card], []), ([short], [])]), \
                patch.object(s0.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "no unclaimed physical CUDA device with stable headroom"):
                s0.claim_device(Path(out), 4096)
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"PUBLIC": out}):
            with self.assertRaisesRegex(ValueError, "required_free_mib"):
                s0.claim_device(Path(out))

    def test_in_session_headroom_recheck_tolerates_foreign_growth_activity_and_context_changes(self):
        card = dict(index=1, uuid="GPU-test", total_mib=97887, used_mib=1164, free_mib=96723, utilization_percent=0)
        grown = dict(card, used_mib=30000, free_mib=67887, utilization_percent=90)
        baseline = dict(uuid="GPU-test", reserved_baseline_mib=1164, process_identities=[])
        snapshots = [([card], []), ([grown], [["GPU-test", "5", "app", "C"]])] * 3
        with tempfile.TemporaryDirectory() as out, patch.object(s0, "inventory", side_effect=snapshots[:5]), \
                patch.object(s0.time, "sleep"):
            s0.verify_headroom(card, Path(out), "phase", baseline, 4096)
            self.assertEqual(len(json.loads((Path(out) / "phase-samples.json").read_text(encoding="utf-8"))), 5)

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
