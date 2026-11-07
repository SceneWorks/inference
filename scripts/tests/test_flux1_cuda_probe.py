"""Metadata-only CUDA proof checks; no model or device execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
from scripts.media_campaign_supervisor import SupervisionError


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
        row = {"pid": 123, "cudaDeviceUuid": UUID, "poolUsedHighBytes": 80 * MIB,
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

    def test_rejects_wrong_or_absent_owner_uuid_and_zero_or_unknown_counters(self):
        for change in ({"pid": 456}, {"cudaDeviceUuid": "GPU-00000000-0000-0000-0000-000000000000"},
                       {"poolUsedHighBytes": 0}, {"poolReservedHighBytes": 0},
                       {"poolReservedBytes": None}, {"poolReservedBytes": 130 * MIB},
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


if __name__ == "__main__":
    unittest.main()
