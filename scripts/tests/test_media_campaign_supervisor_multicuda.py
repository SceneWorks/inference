"""Model-free multi-CUDA supervision: fake probes/process ownership, no driver or GPU query."""
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import media_campaign_supervisor as safety

UUIDS = ("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff", "GPU-11111111-1111-1111-1111-111111111111", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


class Timeline:
    def __init__(self, root_exit=2, empty=7, escaped=True):
        self.tick = 0
        self.root_exit, self.empty, self.escaped = root_exit, empty, escaped
        self.trace, self.stopped = [], []
        self.spawned = False
        self.child = SimpleNamespace(pid=1000, poll=self.poll, wait=self.wait)

    def poll(self):
        return 0 if self.tick >= self.root_exit else None

    def wait(self, timeout=None):
        return 0

    def clock(self):
        return self.tick * .01

    def sleep(self, _seconds):
        self.tick += 1

    def snapshot(self, _owner):
        if self.tick >= self.empty:
            return set(), set()
        if self.tick >= self.root_exit:
            return (set() if self.escaped else {1001}), {1001}
        return {1000}, {1000}

    def spawn(self, *_args, **_kwargs):
        self.spawned = True
        return self.child


class DeviceProbe:
    def __init__(self, policy, timeline, role, breach=None, use=64):
        self.policy, self.timeline, self.role = policy, timeline, role
        self.breach, self.use = breach, use
        self.owners = []

    def host_admission(self):
        return 100000, None

    def tree_footprint(self, owner):
        self.timeline.trace.append((self.role, "rss", self.timeline.tick))
        return 2048

    def active_failure(self, phase):
        if not self.breach:
            return None
        when, kind = self.breach
        if when == "before" and phase == "before":
            return kind
        if phase == "sample" and ((when == "during" and self.timeline.tick < self.timeline.root_exit)
                                  or (when == "after" and self.timeline.tick >= self.timeline.root_exit)):
            return kind
        if phase == "sample" and when == "alternating" and self.timeline.tick % 2 == self.role % 2:
            return kind
        return None

    def gpu_free(self):
        self.timeline.trace.append((self.role, "preflight", self.timeline.tick))
        failure = self.active_failure("before")
        if failure == "probe":
            raise safety.SupervisionError("probe-failure", f"CPU fixture role {self.role} unavailable")
        if failure == "reserve":
            return self.policy.gpu_free_reserve_bytes + self.policy.child_gpu_cap_bytes - 1
        return 100000

    def gpu_free_and_tree_bytes(self, owner):
        self.timeline.trace.append((self.role, "sample", self.timeline.tick))
        self.owners.append(owner)
        failure = self.active_failure("sample")
        if failure == "probe":
            raise safety.SupervisionError("probe-failure", f"CPU fixture role {self.role} sample unavailable")
        free = self.policy.gpu_free_reserve_bytes - 1 if failure == "reserve" else 100000
        used = self.policy.child_gpu_cap_bytes + 1 if failure == "cap" else self.use
        return free, used


class MultiCudaSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = 0
        self.policies = []
        for role, uuid in enumerate(UUIDS):
            data = {"schemaVersion": 1, "backend": "linux-cuda", "deadlineSeconds": 2,
                    "pollMillis": 10, "termGraceMillis": 20,
                    "hostFreeReserveBytes": 100, "childFootprintCapBytes": 4096,
                    "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 100000,
                    "cudaDeviceUuid": uuid, "gpuFreeReserveBytes": 100 * (role + 1),
                    "childGpuCapBytes": 1000 * (role + 1)}
            path = self.root / f"policy-{role}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.policies.append(safety.load_policy(path))

    def run_fake(self, timeline=None, probes=None, policies=None, **kwargs):
        timeline = timeline or Timeline()
        policies = self.policies[:2] if policies is None else policies
        probes = [DeviceProbe(policy, timeline, role) for role, policy in enumerate(policies)] if probes is None else probes
        self.sequence += 1
        directory = self.root / str(self.sequence)
        directory.mkdir()
        is_windows = policies[0].backend == "windows-cuda"
        timeline.job_assignments, timeline.job_closes = [], []
        def close_job():
            timeline.job_closes.append(True)
        timeline.job = SimpleNamespace(
            members=lambda: set.union(*timeline.snapshot(None)),
            assign_and_resume=lambda child: timeline.job_assignments.append(child.pid),
            terminate_and_reap=lambda child, grace: timeline.stopped.append((child.pid, grace)),
            close=close_job)
        with patch.object(safety.os, "name", "nt" if is_windows else os.name), \
             patch.object(safety.windows, "WindowsJob", return_value=timeline.job), \
             patch.object(safety.subprocess, "Popen", side_effect=timeline.spawn), \
             patch.object(safety.PosixTree, "snapshot", lambda owner: timeline.snapshot(owner)), \
             patch.object(safety, "_stop_tree", side_effect=lambda child, grace, known: timeline.stopped.append((child.pid, grace, set(known)))), \
             patch.object(safety.time, "sleep", side_effect=timeline.sleep), \
             patch.object(safety.time, "time_ns", return_value=12345), \
             patch.object(safety, "SystemProbe", side_effect=AssertionError("no real probes/device discovery")):
            return safety.run_guarded(
                ["CPU-fixture-never-executed"], cwd=self.root, env={}, policy=policies[0],
                stdout_path=directory / "out", stderr_path=directory / "err", probe=probes[0],
                clock=timeline.clock, additional_cuda_policies=tuple(policies[1:]),
                additional_cuda_probes=tuple(probes[1:]), **kwargs)

    def test_single_device_failure_retains_sample_and_usage_basis(self):
        for basis in ('owned-process-attributed', 'global-used-upper-bound'):
            for breach in ('cap', 'reserve'):
                with self.subTest(basis=basis, breach=breach):
                    policy = dataclasses.replace(self.policies[0], backend='windows-cuda')
                    timeline = Timeline()
                    probe = DeviceProbe(policy, timeline, 0, breach=('during', breach), use=64)
                    probe.gpu_usage_basis = basis
                    with self.assertRaises(safety.SupervisionError) as caught:
                        self.run_fake(timeline, [probe], [policy], require_gpu_sample=False)
                    error = caught.exception
                    self.assertEqual(error.reason, 'device-memory')
                    sample = error.watchdog_device_memory
                    self.assertEqual(sample['cudaDeviceUuid'], policy.cuda_device_uuid)
                    self.assertEqual(sample['gpuUsageBasis'], basis)
                    self.assertEqual(sample['capExceeded'], breach == 'cap')
                    self.assertEqual(sample['reserveBreached'], breach == 'reserve')
                    self.assertEqual(sample['childGpuCapBytes'], policy.child_gpu_cap_bytes)
                    self.assertEqual(sample['gpuFreeReserveBytes'], policy.gpu_free_reserve_bytes)
                    self.assertEqual(sample['atNs'], 12345)
                    self.assertGreaterEqual(sample['elapsedSeconds'], 0)
                    key = 'ownedGpuBytes' if basis == 'owned-process-attributed' else 'globalUsedUpperBoundBytes'
                    peak = 'sampledOwnedPeakBytes' if basis == 'owned-process-attributed' else 'sampledGlobalUpperBoundPeakBytes'
                    self.assertEqual(sample[key], policy.child_gpu_cap_bytes + 1 if breach == 'cap' else 64)
                    self.assertEqual(sample[peak], sample[key])
                    other = 'globalUsedUpperBoundBytes' if key == 'ownedGpuBytes' else 'ownedGpuBytes'
                    self.assertIsNone(sample[other])
                    self.assertEqual(len(timeline.stopped), 1)
                    path = self.root / f'failure-{basis}-{breach}.json'
                    safety.write_unaccepted_record(path, kind='test', coordinate='single', error=error)
                    saved = json.loads(path.read_text())
                    self.assertFalse(saved['accepted'])
                    self.assertEqual(saved['watchdogDeviceMemory'], sample)

    def test_role_order_independent_caps_and_estimates_are_preserved(self):
        timeline = Timeline()
        probes = [DeviceProbe(p, timeline, role, use=500 + role * 750) for role, p in enumerate(self.policies)]
        result = self.run_fake(timeline, probes, self.policies, static_floor_gpu_bytes=500,
                               gpu_estimate_source="CPU primary estimate")
        self.assertIsInstance(result, safety.RunResult)
        self.assertIsInstance(result, safety.MultiCudaRunResult)
        self.assertEqual([row["cudaDeviceUuid"] for row in result.cuda_devices], list(UUIDS))
        self.assertNotEqual(list(UUIDS), sorted(UUIDS))
        self.assertEqual([row["role"] for row in result.cuda_devices], [0, 1, 2])
        self.assertEqual([row["peakGpuBytes"] for row in result.cuda_devices], [500, 1250, 2000])
        for index, row in enumerate(result.cuda_devices):
            expected = 500 if index == 0 else self.policies[index].child_gpu_cap_bytes
            self.assertEqual(row["admission"]["gpuEstimateBytes"], expected)
            self.assertEqual(row["admission"]["gpuFreeReserveBytes"], self.policies[index].gpu_free_reserve_bytes)
            self.assertEqual(row["admission"]["childGpuCapBytes"], self.policies[index].child_gpu_cap_bytes)
            safety.validate_admission(row["admission"], policy_sha256=self.policies[index].sha256)
            self.assertTrue(row["samples"])
            self.assertTrue(any(sample["rootExited"] for sample in row["samples"]))
        self.assertEqual(result.peak_gpu_bytes, 500)
        owners = [owner for probe in probes for owner in probe.owners]
        self.assertTrue(all(owner is owners[0] for owner in owners))
        self.assertEqual(owners[0].cuda_device_uuids, UUIDS)
        self.assertEqual(timeline.stopped, [])

    def test_reserve_and_probe_failures_before_spawn_on_each_role(self):
        for role in [0, 1]:
            for kind in ["reserve", "probe"]:
                with self.subTest(role=role, kind=kind):
                    timeline = Timeline()
                    probes = [DeviceProbe(p, timeline, index, ("before", kind) if index == role else None)
                              for index, p in enumerate(self.policies[:2])]
                    with self.assertRaises(safety.SupervisionError) as caught:
                        self.run_fake(timeline, probes)
                    self.assertEqual(caught.exception.reason, "preflight-memory" if kind == "reserve" else "probe-failure")
                    self.assertIsNone(caught.exception.pid)
                    self.assertFalse(timeline.spawned)
                    self.assertEqual(timeline.stopped, [])
                    self.assertIn(UUIDS[role], caught.exception.detail)
                    self.assertIsNone(caught.exception.cuda_devices[role]["peakGpuBytes"])

    def test_each_role_reserve_cap_and_sustained_probe_failure_during_and_after_root_exit(self):
        for role in [0, 1]:
            for phase in ["during", "after"]:
                for kind in ["reserve", "cap", "probe"]:
                    with self.subTest(role=role, phase=phase, kind=kind):
                        timeline = Timeline(root_exit=6 if phase == "during" else 1, empty=20)
                        probes = [DeviceProbe(p, timeline, index, (phase, kind) if index == role else None)
                                  for index, p in enumerate(self.policies[:2])]
                        with self.assertRaises(safety.SupervisionError) as caught:
                            self.run_fake(timeline, probes)
                        self.assertEqual(caught.exception.reason, "probe-failure" if kind == "probe" else "device-memory")
                        self.assertEqual(caught.exception.pid, 1000)
                        self.assertIn(UUIDS[role], caught.exception.detail)
                        self.assertEqual(len(timeline.stopped), 1)
                        self.assertEqual(timeline.stopped[0][0], 1000)
                        if phase == "after":
                            self.assertGreaterEqual(timeline.tick, timeline.root_exit)
                            self.assertLess(timeline.tick, timeline.empty)

    def test_transient_primary_failure_does_not_skip_secondary_or_share_failure_counter(self):
        timeline = Timeline(root_exit=2, empty=12)
        probes = [DeviceProbe(p, timeline, role, ("alternating", "probe")) for role, p in enumerate(self.policies[:2])]
        result = self.run_fake(timeline, probes)
        for tick in range(timeline.empty):
            self.assertEqual({role for role, kind, at in timeline.trace if kind == "sample" and at == tick}, {0, 1})
        self.assertTrue(all(row["peakGpuBytes"] == 64 for row in result.cuda_devices))
        self.assertEqual(timeline.stopped, [])

    def test_fast_exit_is_per_uuid_unknown_not_zero(self):
        timeline = Timeline(root_exit=0, empty=0)
        result = self.run_fake(timeline, require_gpu_sample=False)
        self.assertIsNone(result.peak_gpu_bytes)
        for row in result.cuda_devices:
            self.assertIsNone(row["peakGpuBytes"])
            self.assertIsNotNone(row["peakGpuUnknownReason"])
            self.assertEqual(row["samples"], ())
            self.assertEqual(row["admission"]["gpuAvailableBytes"], 100000)

    def test_sampled_zero_is_not_fabricated_positive_or_unknown_peak_zero(self):
        timeline = Timeline()
        probes = [DeviceProbe(p, timeline, role, use=0) for role, p in enumerate(self.policies[:2])]
        result = self.run_fake(timeline, probes, require_gpu_sample=False)
        self.assertTrue(all(row["peakGpuBytes"] is None for row in result.cuda_devices))
        self.assertTrue(all(row["samples"][0]["ownedGpuBytes"] == 0 for row in result.cuda_devices))

    def test_unaccepted_record_adds_multi_device_evidence_only_for_multi(self):
        timeline = Timeline()
        probes = [DeviceProbe(p, timeline, role, ("before", "reserve") if role == 1 else None)
                  for role, p in enumerate(self.policies[:2])]
        with self.assertRaises(safety.SupervisionError) as caught:
            self.run_fake(timeline, probes)
        output = self.root / "multi-unaccepted.json"
        safety.write_unaccepted_record(output, kind="CPU-only", coordinate="pair", error=caught.exception)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(record["accepted"])
        self.assertEqual([row["cudaDeviceUuid"] for row in record["cudaDevices"]], list(UUIDS[:2]))
        self.assertIsNone(record["cudaDevices"][1]["peakGpuBytes"])

    def test_single_result_constructor_and_serialization_remain_legacy(self):
        primary = self.policies[0]
        timeline = Timeline(root_exit=2, empty=2)
        probe = DeviceProbe(primary, timeline, 0)
        del probe.policy  # legacy injected probes need no multi-device identity protocol
        result = self.run_fake(timeline, [probe], [primary])
        names = [field.name for field in dataclasses.fields(safety.RunResult)]
        self.assertEqual(names, ["pid", "returncode", "peak_host_bytes", "peak_gpu_bytes", "host_free_at_launch",
                                 "gpu_free_at_launch", "elapsed_seconds", "samples", "admission"])
        self.assertIs(type(result), safety.RunResult)
        self.assertEqual(list(dataclasses.asdict(result)), names)
        self.assertFalse(hasattr(result, "cuda_devices"))
        self.assertEqual(dataclasses.asdict(safety.RunResult(*(getattr(result, name) for name in names))), dataclasses.asdict(result))

    def test_windows_global_upper_bound_enforces_cap_without_faking_owned_peak(self):
        data = json.loads(self.policies[0].canonical_bytes)
        data['backend'] = 'windows-cuda'
        data['childGpuCapBytes'] = 1000
        path = self.root / 'wddm-policy.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        policy = safety.load_policy(path)
        class UpperBoundProbe(DeviceProbe):
            def gpu_free_and_tree_bytes(self, owner):
                self.gpu_usage_basis = 'global-used-upper-bound'
                return super().gpu_free_and_tree_bytes(owner)

        timeline = Timeline()
        probe = UpperBoundProbe(policy, timeline, 0, use=900)
        result = self.run_fake(timeline, [probe], [policy], require_gpu_sample=False)
        self.assertIsNone(result.peak_gpu_bytes)
        bounds = [sample for sample in result.samples if sample.get('phase') == 'cuda-global-used-upper-bound-sample']
        self.assertTrue(bounds)
        self.assertTrue(all(sample['globalUsedUpperBoundBytes'] == 900 for sample in bounds))
        self.assertTrue(all('ownedGpuBytes' not in sample for sample in bounds))
        self.assertEqual(timeline.stopped, [])

        timeline = Timeline()
        probe = UpperBoundProbe(policy, timeline, 0, use=1001)
        with self.assertRaisesRegex(safety.SupervisionError, 'device-memory'):
            self.run_fake(timeline, [probe], [policy], require_gpu_sample=False)
        self.assertEqual(len(timeline.stopped), 1)

        timeline = Timeline()
        probe = UpperBoundProbe(policy, timeline, 0, use=900)
        with self.assertRaisesRegex(safety.SupervisionError, 'no attributable GPU-memory sample'):
            self.run_fake(timeline, [probe], [policy])

    def test_legacy_unaccepted_schema_has_no_multi_fields(self):
        error = safety.SupervisionError("preflight-memory", "CPU fixture")
        path = self.root / "legacy.json"
        safety.write_unaccepted_record(path, kind="CPU-only", coordinate="single", error=error)
        self.assertNotIn("cudaDevices", json.loads(path.read_text(encoding="utf-8")))

    def test_invalid_pairs_and_shared_settings_reject_before_any_probe_or_spawn(self):
        primary, secondary = self.policies[:2]
        for field in ["backend", "deadline_seconds", "poll_millis", "term_grace_millis", "host_free_reserve_bytes",
                      "child_footprint_cap_bytes", "stdout_cap_bytes", "stderr_cap_bytes", "event_cap_bytes"]:
            value = "windows-cuda" if field == "backend" else getattr(secondary, field) + 1
            changed = dataclasses.replace(secondary, **{field: value})
            with self.subTest(field=field):
                timeline = Timeline()
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-policy"):
                    self.run_fake(timeline, policies=[primary, changed])
                self.assertFalse(timeline.spawned)
        invalid = [dataclasses.replace(secondary, backend="linux-cpu"),
                   dataclasses.replace(secondary, cuda_device_uuid=None),
                   dataclasses.replace(secondary, cuda_device_uuid=primary.cuda_device_uuid),
                   dataclasses.replace(secondary, cuda_device_uuid=primary.cuda_device_uuid.upper())]
        for changed in invalid:
            with self.subTest(uuid=changed.cuda_device_uuid, backend=changed.backend):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-policy"):
                    self.run_fake(policies=[primary, changed])
        timeline = Timeline()
        for probes in [[None, DeviceProbe(secondary, timeline, 1)],
                       [DeviceProbe(primary, timeline, 0), None],
                       [DeviceProbe(secondary, timeline, 0), DeviceProbe(secondary, timeline, 1)]]:
            with self.subTest(probes=probes):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-policy"):
                    self.run_fake(timeline, probes)
        self.assertFalse(timeline.spawned)

    def test_tuple_count_type_mismatch_and_unguarded_secondary_reject(self):
        primary, secondary = self.policies[:2]
        probe = DeviceProbe(primary, Timeline(), 0)
        for policies, probes in [((secondary,), ()), ((), (probe,)), ([secondary], (probe,)), ((), [])]:
            with self.subTest(policies=policies, probes=probes):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-policy"):
                    safety.run_guarded(["never"], cwd=self.root, env={}, policy=primary, probe=probe,
                                       stdout_path=self.root / "out", stderr_path=self.root / "err",
                                       additional_cuda_policies=policies, additional_cuda_probes=probes)
        with self.assertRaisesRegex(safety.SupervisionError, "unguarded-policy"):
            self.run_fake(policies=[primary, dataclasses.replace(secondary, child_gpu_cap_bytes=0)])

    def windows_policies(self):
        result = []
        for role, policy in enumerate(self.policies[:2]):
            data = json.loads(policy.canonical_bytes)
            data["backend"] = "windows-cuda"
            path = self.root / f"windows-policy-{role}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result.append(safety.load_policy(path))
        return result

    def test_windows_job_is_shared_and_secondary_draining_failure_uses_existing_job_cleanup(self):
        policies = self.windows_policies()
        timeline = Timeline(root_exit=1, empty=10)
        probes = [DeviceProbe(p, timeline, role, ("after", "cap") if role == 1 else None)
                  for role, p in enumerate(policies)]
        with self.assertRaisesRegex(safety.SupervisionError, "device-memory") as caught:
            self.run_fake(timeline, probes, policies)
        self.assertEqual(caught.exception.pid, 1000)
        self.assertEqual(timeline.job_assignments, [1000])
        self.assertEqual(timeline.job_closes, [True])
        self.assertEqual(timeline.stopped, [(1000, policies[0].term_grace_millis / 1000)])
        self.assertTrue(all(owner is timeline.job for probe in probes for owner in probe.owners))
        self.assertEqual(timeline.job.cuda_device_uuids, UUIDS[:2])

    def test_windows_requires_attributable_samples_for_each_uuid_unless_explicitly_disabled(self):
        policies = self.windows_policies()
        timeline = Timeline()
        probes = [DeviceProbe(p, timeline, role, use=0 if role == 1 else 64)
                  for role, p in enumerate(policies)]
        with self.assertRaisesRegex(safety.SupervisionError, "no attributable") as caught:
            self.run_fake(timeline, probes, policies)
        self.assertIn(UUIDS[1], caught.exception.detail)
        self.assertIsNone(caught.exception.cuda_devices[1]["peakGpuBytes"])
        timeline = Timeline(root_exit=0, empty=0)
        result = self.run_fake(timeline, policies=policies, require_gpu_sample=False)
        self.assertTrue(all(row["peakGpuBytes"] is None for row in result.cuda_devices))
        self.assertEqual(timeline.job_assignments, [1000])
        self.assertEqual(timeline.job_closes, [True])
        self.assertEqual(timeline.stopped, [])

    def test_primary_host_guard_and_shared_event_cap_still_abort_the_same_owned_tree(self):
        timeline = Timeline()
        probes = [DeviceProbe(p, timeline, role) for role, p in enumerate(self.policies[:2])]
        probes[0].host_admission = lambda: (0 if timeline.spawned else 100000, None)
        with self.assertRaisesRegex(safety.SupervisionError, "host-memory"):
            self.run_fake(timeline, probes)
        self.assertEqual(len(timeline.stopped), 1)
        policies = [dataclasses.replace(policy, event_cap_bytes=400) for policy in self.policies[:2]]
        with self.assertRaisesRegex(safety.SupervisionError, "event-cap"):
            self.run_fake(policies=policies)

    def test_shared_deadline_is_not_multiplied_by_device_count(self):
        for phase in ["preflight", "sample"]:
            with self.subTest(phase=phase):
                timeline = Timeline(empty=1000)
                probes = [DeviceProbe(p, timeline, role) for role, p in enumerate(self.policies[:2])]
                if phase == "preflight":
                    original = probes[0].gpu_free
                    def delayed_free():
                        value = original()
                        timeline.tick = 201
                        return value
                    probes[0].gpu_free = delayed_free
                else:
                    original = probes[0].gpu_free_and_tree_bytes
                    def delayed_sample(owner):
                        value = original(owner)
                        timeline.tick = 201
                        return value
                    probes[0].gpu_free_and_tree_bytes = delayed_sample
                with self.assertRaisesRegex(safety.SupervisionError, "deadline"):
                    self.run_fake(timeline, probes)
                self.assertEqual(timeline.spawned, phase == "sample")
                self.assertEqual(len(timeline.stopped), 1 if phase == "sample" else 0)

    def test_non_cuda_or_missing_primary_uuid_cannot_acquire_secondary_monitoring(self):
        primary, secondary = self.policies[:2]
        for changed in [dataclasses.replace(primary, backend="linux-cpu"),
                        dataclasses.replace(primary, cuda_device_uuid=None)]:
            timeline = Timeline()
            with self.assertRaisesRegex(safety.SupervisionError, "invalid-policy"):
                self.run_fake(timeline, policies=[changed, secondary])
            self.assertFalse(timeline.spawned)

    def test_system_probe_attributes_each_explicit_uuid_and_still_refuses_outside_set(self):
        owner = SimpleNamespace(members=lambda: {1000}, cuda_device_uuids=UUIDS[:2])
        usage = f"1000, 32, {UUIDS[0]}\n1000, 64, {UUIDS[1]}\n999, 999, {UUIDS[2]}\n"
        def bounded(argv):
            return usage if "--query-compute-apps=pid,used_gpu_memory,gpu_uuid" in argv else f"{UUIDS[0]}, 8192\n{UUIDS[1]}, 4096\n"
        with patch.object(safety, "_bounded_output", side_effect=bounded):
            for role, expected in [(0, 32), (1, 64)]:
                probe = object.__new__(safety.SystemProbe)
                probe.policy, probe._smi = self.policies[role], "CPU-fixture-not-executed"
                self.assertEqual(probe.gpu_free_and_tree_bytes(owner)[1], expected * 1024**2)
            owner.cuda_device_uuids = (UUIDS[0],)
            with self.assertRaisesRegex(safety.SupervisionError, "unexpected GPU"):
                probe.gpu_free_and_tree_bytes(owner)


if __name__ == "__main__":
    unittest.main()
