"""Windows byte-gauge and CUDA identity contracts, with no native/model execution."""
from contextlib import contextmanager
import ctypes
import os
import subprocess
import sys
from unittest import TestCase, skipUnless
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
    job.root_identity.return_value = (999, False)
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

    def test_actual_instance_array_controls_uniqueness_even_if_scalar_index_aliases(self):
        api = MagicMock()
        api.open.return_value = api.add.return_value = api.collect.return_value = api.close.return_value = 0
        name = f"pid_{PID}_{LUID}_phys_0"
        sample = [10, 512]
        api.read.return_value = sample
        with patch.object(wddm, "_Pdh", return_value=api):
            diagnostic = {}
            with wddm._query(ADAPTER, PID, diagnostic=diagnostic) as read:
                api.selected_rows.return_value = [(name, sample)]
                self.assertEqual(read(), sample)
                self.assertEqual(diagnostic["enumerated_count"], 1)
                self.assertEqual(diagnostic["scalar"], diagnostic["scalar_index_one"])
                for duplicate in (name, name + "#1"):
                    api.selected_rows.return_value = [(name, sample), (duplicate, sample)]
                    with self.assertRaisesRegex(OSError, "ambiguous"):
                        read()
                api.selected_rows.return_value = [(name + "#1", sample)]
                with self.assertRaisesRegex(OSError, "ambiguous"):
                    read()
                api.selected_rows.return_value = []
                self.assertIsNone(read())
        paths = [call.args[1] for call in api.add.call_args_list]
        self.assertEqual(paths, [rf"\GPU Process Memory({i})\Dedicated Usage" for i in ("*", name, name + "#1")])
        api.close.assert_called_once()

    def test_native_array_two_pass_retry_reads_only_exact_pid_adapter_node(self):
        name = f"pid_{PID}_{LUID}_phys_0"
        names = [name.replace("pid_17", "pid_171"), name.replace("phys_0", "phys_1"),
                 name.replace("22222222", "33333333"), name, name + "#1"]
        strings = [ctypes.create_unicode_buffer(n) for n in names]
        calls = []
        def fill(counter, size, count, buf):
            calls.append(buf is None)
            if buf is None:
                size._obj.value = ctypes.sizeof(wddm._RawItem) * len(names)
                return wddm.MORE_DATA
            if len(calls) == 2:
                return wddm.MORE_DATA
            rows = (wddm._RawItem * len(names)).from_buffer(buf)
            count._obj.value = len(rows)
            for i, row in enumerate(rows):
                row.name = ctypes.cast(strings[i], ctypes.c_wchar_p)
                row.raw.status = 0 if i >= 3 else 999
                row.raw.stamp_low = 10
                row.raw.first = 100 + i
            return 0
        api = wddm._Pdh.__new__(wddm._Pdh)
        api.array = fill
        self.assertEqual(api.selected_rows(None, name), [(name, [10, 103]), (name + "#1", [10, 104])])
        self.assertEqual(calls, [True, False, True, False])

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

    def test_both_counters_bracket_each_read_with_assigned_handle_even_when_image_lookup_fails(self):
        from scripts.media_campaign_windows import WindowsJob
        job = WindowsJob.__new__(WindowsJob)
        job.kernel = MagicMock()
        job._root = MagicMock(pid=PID)
        job._root._handle = 12345
        job._root.poll.return_value = None
        job.members = MagicMock(return_value={PID})
        job.kernel.GetProcessId.return_value = PID
        def times(handle, created, *_):
            self.assertEqual(handle, job._root._handle)
            created._obj.dwHighDateTime = 2
            created._obj.dwLowDateTime = 1
            return True
        job.kernel.GetProcessTimes.side_effect = times
        for shared in (False, True):
            with self.subTest(shared=shared), patch.object(wddm, "_query") as query:
                job.kernel.reset_mock()
                # In the former identity path a missing image or OpenProcess
                # result poisoned the record even with the original handle live.
                job.kernel.OpenProcess.return_value = 0
                job.kernel.QueryFullProcessImageNameW.return_value = 0
                query.return_value.__enter__.return_value.side_effect = [[10, 64], [11, 72]]
                probe = wddm.DedicatedSamples(ADAPTER, GPU, shared=shared)
                probe.poll(job, cuda_active=True)
                probe.poll(job, cuda_active=True)
                self.assertEqual(probe.record["created_filetime"], 2 << 32 | 1)
                self.assertEqual(wddm.sampled_peak(probe.record, pid=PID, adapter=ADAPTER), 72)
                self.assertEqual(job.kernel.GetProcessTimes.call_count, 4)
                self.assertEqual([c.args for c in job.kernel.GetProcessId.call_args_list],
                                 [(job._root._handle,)] * 4)
                job.kernel.OpenProcess.assert_not_called()
                job.kernel.QueryFullProcessImageNameW.assert_not_called()
                probe.close()

        job.kernel.GetProcessId.return_value = PID + 1
        with self.assertRaisesRegex(RuntimeError, "different PID"):
            job.root_identity(PID)
        job.kernel.GetProcessId.return_value = PID
        job.kernel.GetProcessTimes.side_effect = None
        job.kernel.GetProcessTimes.return_value = False
        with patch("scripts.media_campaign_windows._failed", return_value=RuntimeError("times unavailable")):
            probe = wddm.DedicatedSamples(ADAPTER, GPU)
            probe.poll(job, cuda_active=True)
            self.assertEqual(probe.record["error"], "times unavailable")
            self.assertIsNone(wddm.sampled_peak(probe.record, pid=PID, adapter=ADAPTER))

    def test_root_exit_requires_the_assigned_process_handle_and_an_empty_job(self):
        from scripts.media_campaign_windows import WindowsJob
        job = WindowsJob.__new__(WindowsJob)
        job.members = MagicMock()
        job._root = MagicMock(pid=PID)
        for pid, status, members, exited in ((PID, 0, set(), True), (PID, None, set(), False),
                                             (PID + 1, 0, set(), False), (PID, 0, {PID + 1}, False)):
            job._root.poll.return_value = status
            job.members.return_value = members
            self.assertEqual(job.exited_root_and_empty(pid), exited)

    def test_final_poll_requires_original_root_exit_and_empty_job_and_discards_unbracketed_data(self):
        for stage in ("before-open", "after-read", "during-recheck", "before-poll"):
            for confirmed in (True, False):
                with self.subTest(stage=stage, confirmed=confirmed), patch.object(wddm, "_query") as query:
                    job = owner()
                    job.exited_root_and_empty.return_value = confirmed
                    reader = query.return_value.__enter__.return_value
                    reader.side_effect = [[10, 64], [11, 72], [12, 9999]]
                    probe = wddm.DedicatedSamples(ADAPTER, GPU)
                    probe.poll(job, cuda_active=True)
                    probe.poll(job, cuda_active=True)
                    resolved = job.root_identity.return_value
                    departed = (999, True)
                    if stage == "before-open":
                        job.root_identity.return_value = departed
                    elif stage == "after-read":
                        job.members.side_effect = [{PID}, set()]
                    elif stage == "during-recheck":
                        job.root_identity.side_effect = [resolved, departed]
                    else:
                        job.members.return_value = set()
                    probe.poll(job, cuda_active=True)
                    job.exited_root_and_empty.assert_called_once_with(PID)
                    self.assertEqual(probe.record["samples"], [[10, 64], [11, 72]])
                    self.assertEqual(wddm.sampled_peak(probe.record, pid=PID, adapter=ADAPTER),
                                     72 if confirmed else None)
                    probe.close()

    def test_later_exit_cannot_clear_live_alias_changed_creation_time_or_earlier_missing_gauge(self):
        for problem in ("foreign-handle", "reused-pid", "descendant", "gap"):
            with self.subTest(problem=problem), patch.object(wddm, "_query") as query:
                job = owner()
                job.exited_root_and_empty.return_value = True
                query.return_value.__enter__.return_value.side_effect = [[10, 64], [11, 72], None]
                probe = wddm.DedicatedSamples(ADAPTER, GPU)
                probe.poll(job, cuda_active=True)
                probe.poll(job, cuda_active=True)
                if problem == "foreign-handle":
                    job.root_identity.side_effect = ValueError("retained handle PID differs")
                elif problem == "reused-pid":
                    job.root_identity.return_value = (1111, True)
                elif problem == "descendant":
                    job.members.return_value = {PID + 1}
                probe.poll(job, cuda_active=True)
                job.members.return_value = set()
                probe.poll(job, cuda_active=True)
                self.assertIsNone(wddm.sampled_peak(probe.record, pid=PID, adapter=ADAPTER))
                probe.close()

    @skipUnless(os.name == "nt", "requires a real Windows Job and original process handle")
    def test_native_exit_between_job_snapshot_and_identity_open(self):
        from scripts.media_campaign_windows import WindowsJob
        job = WindowsJob()
        child = None
        try:
            child = subprocess.Popen([sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read(1)"],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=0x00000004)
            job.assign_and_resume(child)
            self.assertFalse(job.exited_root_and_empty(child.pid))
            with patch.object(wddm, "_query") as query:
                query.return_value.__enter__.return_value.side_effect = [[10, 64], [11, 72]]
                probe = wddm.DedicatedSamples(ADAPTER, GPU)
                self.addCleanup(probe.close)
                probe.poll(job, cuda_active=True)
                probe.poll(job, cuda_active=True)
                created, exited = job.root_identity(child.pid)
                self.assertGreater(created, 0)
                self.assertFalse(exited)
                with self.assertRaisesRegex(RuntimeError, "not the assigned Job root"):
                    job.root_identity(child.pid + 1)
                identify = job.root_identity
                def exit_before_open(pid):
                    child.stdin.close()
                    self.assertEqual(child.wait(timeout=5), 0)
                    return identify(pid)
                with patch.object(job, "root_identity", side_effect=exit_before_open), \
                     patch.object(job, "member_identities", side_effect=AssertionError("reopened root PID")):
                    probe.poll(job, cuda_active=True)
                self.assertEqual(identify(child.pid), (created, True))
                self.assertTrue(job.exited_root_and_empty(child.pid))
                self.assertFalse(job.exited_root_and_empty(child.pid + 1))
                self.assertEqual(wddm.sampled_peak(probe.record, pid=child.pid, adapter=ADAPTER), 72)
        finally:
            job.close()
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                child.stdin.close()

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
