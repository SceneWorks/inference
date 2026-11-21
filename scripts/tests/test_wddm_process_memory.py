"""Windows byte-gauge and CUDA identity contracts, with no native/model execution."""
from contextlib import contextmanager
import ctypes
from unittest import TestCase
from unittest.mock import MagicMock, patch
import uuid

from scripts.ci import wddm_process_memory as wddm


GPU = "GPU-00000011-2222-3333-4444-555555555555"
LUID = "luid_0x44444444_0x22222222"
ADAPTER = {"cuda_uuid": GPU, "luid": LUID, "physical_node": 0, "node_mask": 1, "counter": wddm.COUNTER}
PID = 17


def owner():
    job = MagicMock()
    job.members.return_value = {PID}
    job.member_identities.return_value = {"processes": [{"pid": PID, "status": "owned-identity-resolved",
                                                         "createdFileTime": 999}]}
    return job


class WindowsDedicatedMemoryTests(TestCase):
    def test_cuda_mapping_checks_every_visible_uuid_and_converts_luid_halves(self):
        def run(uuids, mask):
            def cuda(name, *args):
                if name == "cuDeviceGetCount":
                    args[0]._obj.value = len(uuids)
                elif name == "cuDeviceGet":
                    args[0]._obj.value = args[1] + 7
                elif name == "cuDeviceGetUuid_v2":
                    args[0][:] = uuid.UUID(uuids[args[1].value - 7][4:]).bytes
                elif name == "cuDeviceGetLuid":
                    args[0][:] = bytes.fromhex("2222222244444444")
                    args[1]._obj.value = mask
                else:
                    self.assertEqual(name, "cuInit")
                return 0

            with patch.object(ctypes, "WinDLL", create=True), patch.object(wddm, "_function",
                    side_effect=lambda dll, name, types: lambda *args: cuda(name, *args)):
                return wddm._adapter(GPU)

        other = "GPU-00000022-2222-3333-4444-555555555555"
        self.assertEqual(run([other, GPU], 1), ADAPTER)
        for devices, mask in (([other], 1), ([GPU, GPU], 1), ([GPU], 0), ([GPU], 3)):
            with self.assertRaises(ValueError):
                run(devices, mask)

    def test_targeted_exact_query_rejects_duplicate_and_closes_on_failure(self):
        api = MagicMock()
        api.open.return_value = api.add.return_value = api.collect.return_value = api.close.return_value = 0
        api.read.side_effect = [[10, 512], [10, 12]]
        with patch.object(wddm, "_Pdh", return_value=api), self.assertRaisesRegex(OSError, "ambiguous"):
            with wddm._query(ADAPTER, PID) as read:
                read()
        paths = [call.args[1] for call in api.add.call_args_list]
        prefix = rf"\GPU Process Memory(pid_{PID}_{LUID}_phys_0"
        self.assertEqual(paths, [prefix + suffix + r")\Dedicated Usage" for suffix in ("", "#1")])
        api.close.assert_called_once()

    def test_preflight_requires_selected_adapter_but_allows_an_unregistered_process(self):
        with patch.object(wddm.os, "name", "nt"), patch.object(wddm, "_adapter", return_value=ADAPTER):
            with patch.object(wddm, "_query") as query:
                query.return_value.__enter__.return_value.side_effect = [[10, 0], None]
                self.assertEqual(wddm.preflight(GPU), ADAPTER)
                self.assertEqual(query.call_args_list[0].args, (ADAPTER,))
                self.assertEqual(query.call_args_list[1].args, (ADAPTER, wddm.os.getpid()))
            with patch.object(wddm, "_query") as query, self.assertRaisesRegex(ValueError, "selected adapter"):
                query.return_value.__enter__.return_value.return_value = None
                wddm.preflight(GPU)

    def test_live_poll_is_owned_on_both_sides_and_delayed_registration_never_becomes_zero(self):
        job = owner()
        @contextmanager
        def query(adapter, pid):
            self.assertEqual((adapter, pid), (ADAPTER, PID))
            yield iter([None, [20, 12], [20, 12], [30, 24], None]).__next__

        with patch.object(wddm, "_query", side_effect=query):
            probe = wddm.DedicatedSamples(ADAPTER, GPU)
            probe.poll(job, cuda_active=False)
            for _ in range(3):
                probe.poll(job, cuda_active=True)
            row = probe.record
            self.assertEqual(row["samples"], [[20, 12], [30, 24]])
            self.assertEqual(wddm.sampled_peak(row, pid=PID, adapter=ADAPTER), 24)
            probe.poll(job, cuda_active=True)
            self.assertIsNone(wddm.sampled_peak(row, pid=PID, adapter=ADAPTER))
            probe.close()

    def test_owned_job_pid_turnover_descendants_and_receipt_tampering(self):
        row = dict(adapter=ADAPTER, pid=PID, created_filetime=111, samples=[[1, 3], [2, 5]],
                   missing_active_polls=0, error=None)
        for change in ({"pid": PID + 1}, {"adapter": dict(ADAPTER, physical_node=1)},
                       {"samples": [[2, 7], [1, 5]]}, {"samples": [[1, True]]}, {"missing_active_polls": True}):
            with self.assertRaises(ValueError):
                wddm.sampled_peak(dict(row, **change), pid=PID, adapter=ADAPTER)
        for change in ({"samples": [[1, 3]]}, {"error": "ambiguous instance"},
                       {"samples": [[1, 0], [2, 0]]}, {"missing_active_polls": 1}):
            self.assertIsNone(wddm.sampled_peak(dict(row, **change), pid=PID, adapter=ADAPTER))

        job = owner()
        with patch.object(wddm, "_query") as query:
            query.return_value.__enter__.return_value.return_value = [22, 60]
            job.members.side_effect = [{PID}, {PID, 18}]
            probe = wddm.DedicatedSamples(ADAPTER, GPU)
            probe.poll(job, cuda_active=True)
            self.assertRegex(probe.record["error"], "identity changed")
            self.assertFalse(probe.record["samples"])
            probe.close()
            job.members.side_effect = None
            with patch.object(wddm, "_creation_time", side_effect=[111, 112]):
                probe = wddm.DedicatedSamples(ADAPTER, GPU)
                probe.poll(job, cuda_active=True)
                self.assertRegex(probe.record["error"], "identity changed")
                self.assertFalse(probe.record["samples"])
                probe.close()

    def test_verify_rejects_malformed_or_cross_device_preflight(self):
        for change in ({"cuda_uuid": GPU[:-1] + "1"}, {"luid": "other"}, {"physical_node": True},
                       {"physical_node": 1}, {"node_mask": 0}, {"counter": "Shared Usage"}):
            with self.assertRaises(ValueError):
                wddm.validate_adapter(dict(ADAPTER, **change), GPU)

    def test_raw_gauge_requires_known_pdh_status_type_and_timestamp(self):
        # Build only the mocked API surface, with the SDK structure layout.
        api = wddm._Pdh.__new__(wddm._Pdh)
        def read(status, kind, stamp, value):
            def fill(handle, native_type, native_row):
                native_type._obj.value = kind
                row = native_row._obj
                row.status = status
                row.stamp_high, row.stamp_low = divmod(stamp, 1 << 32)
                row.first = value
                return 0
            api.raw = fill
            return api.read(None)
        self.assertIsNone(read(wddm.NO_INSTANCE, 0, 0, 0))
        self.assertEqual(read(1, wddm.LARGE_RAWCOUNT, 0x100000003, 123), [0x100000003, 123])
        for status, kind, stamp, value in ((2, wddm.LARGE_RAWCOUNT, 10, 2), (0, 0, 10, 2),
                                           (0, wddm.LARGE_RAWCOUNT, 0, 2), (0, wddm.LARGE_RAWCOUNT, 10, -1)):
            with self.assertRaises(OSError):
                read(status, kind, stamp, value)
