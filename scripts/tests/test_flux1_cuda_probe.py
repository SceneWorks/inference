"""Metadata-only CUDA proof checks; no model or device execution."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import media_campaign_windows as windows
from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
from scripts.ci.wan_adapter_tier2_fixture import observe_windows_host
from scripts.media_campaign_supervisor import SupervisionError, SystemProbe


UUID = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
MIB = 1024**2


class FakeSystem:
    gpu_usage_basis = "global-used-upper-bound"  # exact owned PID/UUID WDDM N/A row

    def __init__(self, global_used=200 * MIB):
        self.global_used = global_used

    def gpu_free_and_tree_bytes(self, owner):
        return 1000 * MIB, self.global_used


class PairCudaProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cuda-pool.json"
        policy = SimpleNamespace(cuda_device_uuid=UUID, child_gpu_cap_bytes=256 * MIB)
        self.probe = PairCudaProbe.__new__(PairCudaProbe)
        self.probe.policy = policy
        self.probe.proof_path = self.path
        self.probe.system = FakeSystem()
        self.probe.gpu_usage_basis = "owned-process-attributed"
        self.owner = SimpleNamespace(members=lambda: {123})

    def write(self, **changes):
        row = {"pid": 123, "cudaDeviceUuid": UUID, "poolUsedBytes": 70 * MIB,
               "poolUsedHighBytes": 80 * MIB,
               "poolReservedBytes": 100 * MIB, "poolReservedHighBytes": 120 * MIB}
        row.update(changes)
        self.path.write_text(json.dumps(row), encoding="utf-8")

    def test_old_head_wddm_na_remains_unattributed_until_live_child_proof(self):
        # Old head's 1000 ms poll sees either no child or only the WDDM
        # global upper bound; strict require_gpu_sample cannot accept either.
        self.assertEqual(self.probe.gpu_free_and_tree_bytes(self.owner)[1], 200 * MIB)
        self.assertEqual(self.probe.gpu_usage_basis, "global-used-upper-bound")
        self.write()
        self.assertEqual(self.probe.gpu_free_and_tree_bytes(self.owner)[1], 120 * MIB)
        self.assertEqual(self.probe.gpu_usage_basis, "owned-process-attributed")

    def test_opt_in_host_peaks_reuse_distinct_existing_windows_job_counters(self):
        self.probe.windows_footprint_peaks = {}
        for at, working, private, job in ((1, 500, 300, 450), (2, 450, 600, 550), (3, 400, 550, 700)):
            self.probe.system.last_windows_footprint_details = {
                "sampledProcessCount": 1,
                "processes": [{"pid": 123, "workingSetBytes": working, "privateUsageBytes": private}],
                "jobPeakMemoryUsedBytes": job,
            }
            self.probe.system.tree_footprint = lambda _owner: max(working, private, job)
            with patch("scripts.ci.flux1_pair.cuda_probe.time.time_ns", return_value=at):
                self.assertEqual(self.probe.tree_footprint(self.owner), max(working, private, job))
        self.assertEqual(self.probe.windows_footprint_peaks, {
            "pid": 123, "sample_count": 3, "pid_sample_count": 3, "working_set": {"bytes": 500, "at_unix_ns": 1},
            "private_usage": {"bytes": 600, "at_unix_ns": 2}, "job_peak": {"bytes": 700, "at_unix_ns": 3},
        })
        self.probe.windows_footprint_peaks = None
        self.assertEqual(self.probe.tree_footprint(self.owner), 700)
        self.assertIsNone(self.probe.windows_footprint_peaks)

    def test_exiting_pid_leaves_authentic_job_peak_for_guard_and_diagnostic_replay(self):
        job_peaks = iter((450, 800))

        def query_job(_handle, _info_class, output, _size, _returned):
            output._obj.PeakJobMemoryUsed = next(job_peaks)
            return True

        def memory_info(_process, output, _size):
            output._obj.WorkingSetSize = 500
            output._obj.PrivateUsage = 300
            return True

        job = windows.WindowsJob.__new__(windows.WindowsJob)
        job.handle = 1
        job.members = Mock(side_effect=({123}, {123}, set()))
        job.kernel = SimpleNamespace(OpenProcess=Mock(side_effect=(1, None)), CloseHandle=Mock(),
                                     QueryInformationJobObject=query_job)
        job.psapi = SimpleNamespace(GetProcessMemoryInfo=memory_info)
        self.probe.system = SystemProbe.__new__(SystemProbe)
        self.probe.system.policy = SimpleNamespace(backend="windows-cuda")
        self.probe.windows_footprint_peaks = {}
        samples = []
        for at in (1, 2):
            with patch("scripts.ci.flux1_pair.cuda_probe.time.time_ns", return_value=at):
                samples.append({"sample_kind": "process", "peak_bytes": self.probe.tree_footprint(job)})
        host = self.probe.windows_footprint_peaks
        self.assertEqual(job.last_footprint_details["processes"], [])
        self.assertEqual([sample["peak_bytes"] for sample in samples], [500, 800])
        self.assertEqual(host, {"pid": 123, "sample_count": 2, "pid_sample_count": 1,
                                "working_set": {"bytes": 500, "at_unix_ns": 1},
                                "private_usage": {"bytes": 300, "at_unix_ns": 1},
                                "job_peak": {"bytes": 800, "at_unix_ns": 2}})
        self.assertEqual(observe_windows_host({"pid": 123, "peak_host_bytes": 800, "samples": samples,
                                               "windows_host_peaks": host}), host)

    def test_rejects_wrong_or_absent_owner_uuid_and_zero_or_unknown_counters(self):
        for change in ({"pid": 456}, {"cudaDeviceUuid": "GPU-00000000-0000-0000-0000-000000000000"},
                       {"poolUsedBytes": 0}, {"poolUsedHighBytes": 0},
                       {"poolReservedBytes": 0}, {"poolReservedHighBytes": 0},
                       {"poolReservedBytes": None}, {"poolReservedBytes": 130 * MIB},
                       {"poolUsedBytes": 110 * MIB},
                       {"poolUsedHighBytes": 130 * MIB}, {"poolReservedHighBytes": 300 * MIB}):
            with self.subTest(change=change):
                self.write(**change)
                with self.assertRaises(SupervisionError):
                    self.probe.gpu_free_and_tree_bytes(self.owner)
        self.write()
        self.owner = SimpleNamespace(members=lambda: set())
        with self.assertRaises(SupervisionError):
            self.probe.gpu_free_and_tree_bytes(self.owner)

    def test_unrelated_global_use_only_tightens_cap_and_cannot_supply_attribution(self):
        self.write()
        self.probe.system.global_used = 300 * MIB
        with self.assertRaises(SupervisionError) as raised:
            self.probe.gpu_free_and_tree_bytes(self.owner)
        self.assertEqual(raised.exception.reason, "device-memory")
        self.probe.system.global_used = 50 * MIB
        with self.assertRaises(SupervisionError):
            self.probe.gpu_free_and_tree_bytes(self.owner)

    def test_rejects_linked_proof(self):
        target = Path(self.tmp.name) / "target.json"
        target.write_text("{}", encoding="utf-8")
        self.path.symlink_to(target)
        with self.assertRaises(SupervisionError):
            self.probe.gpu_free_and_tree_bytes(self.owner)

    def test_rejects_malformed_proof_without_defaulting_to_zero(self):
        for content in ('null', '{', '{"pid": 123}', 'x' * 4097):
            with self.subTest(content=content[:16]):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaises(SupervisionError):
                    self.probe.gpu_free_and_tree_bytes(self.owner)

    def test_rejects_path_replacement_between_identity_and_open(self):
        self.write()
        replacement = Path(self.tmp.name) / "replacement.json"
        replacement.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        actual_open = os.open

        def swap_then_open(path, flags):
            os.replace(replacement, self.path)
            return actual_open(path, flags)

        with patch("scripts.ci.flux1_pair.cuda_probe.os.open", side_effect=swap_then_open):
            with self.assertRaises(SupervisionError):
                self.probe.gpu_free_and_tree_bytes(self.owner)

    def test_rejects_symlink_swap_between_identity_and_open(self):
        self.write()
        target = Path(self.tmp.name) / "target.json"
        target.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        actual_open = os.open

        def swap_then_open(path, flags):
            self.path.unlink()
            self.path.symlink_to(target)
            return actual_open(path, flags)

        with patch("scripts.ci.flux1_pair.cuda_probe.os.open", side_effect=swap_then_open):
            with self.assertRaises(SupervisionError):
                self.probe.gpu_free_and_tree_bytes(self.owner)


if __name__ == "__main__":
    unittest.main()
