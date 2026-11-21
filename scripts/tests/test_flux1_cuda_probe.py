"""Metadata-only CUDA proof checks; no model or device execution."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import wddm_process_memory as wddm
from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
from scripts.media_campaign_supervisor import SupervisionError


UUID = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
MIB = 1024**2


class FakeSystem:
    gpu_usage_basis = "global-used-upper-bound"  # exact owned PID/UUID WDDM N/A row

    def __init__(self, global_used=200 * MIB):
        self.global_used = global_used

    def gpu_free_and_tree_bytes(self, owner):
        self.last_global_used_upper_bound_bytes = self.global_used
        return 1000 * MIB, self.global_used


class PairCudaProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cuda-pool.json"
        policy = SimpleNamespace(cuda_device_uuid=UUID, child_gpu_cap_bytes=256 * MIB)
        with patch("scripts.ci.flux1_pair.cuda_probe.SystemProbe", return_value=FakeSystem()):
            self.probe = PairCudaProbe(policy, self.path)
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

    def test_pdh_collects_on_failed_nvidia_tick_between_two_valid_samples(self):
        adapter = {"cuda_uuid": UUID, "luid": "luid_0x00000000_0x00000001", "node_mask": 1,
                   "physical_node": 0, "counter": wddm.COUNTER}
        self.owner.member_identities = lambda pids: {"processes": [
            {"pid": 123, "status": "owned-identity-resolved", "createdFileTime": 11}]}
        for middle, complete in ((None, False), ([21, 150], True)):
            with self.subTest(middle=middle), patch.object(wddm, "_query") as query:
                system = FakeSystem()
                with patch("scripts.ci.flux1_pair.cuda_probe.SystemProbe", return_value=system):
                    probe = PairCudaProbe(self.probe.policy, self.path, dedicated_adapter=adapter)
                query.return_value.__enter__.return_value.side_effect = [[20, 120], middle, [22, 130]]
                probe.gpu_free_and_tree_bytes(self.owner)
                with patch.object(system, "gpu_free_and_tree_bytes", side_effect=SupervisionError("probe-failure", "NVIDIA timeout")):
                    with self.assertRaisesRegex(SupervisionError, "NVIDIA timeout"):
                        probe.gpu_free_and_tree_bytes(self.owner)
                probe.gpu_free_and_tree_bytes(self.owner)
                self.assertEqual(probe.completed_process_gpu_sample(123), complete)
                record = probe.measurement()["wddm_dedicated"]
                self.assertEqual(record["missing_active_polls"], int(not complete))
                self.assertEqual(wddm.sampled_peak(record, pid=123, adapter=adapter), 150 if complete else None)

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
