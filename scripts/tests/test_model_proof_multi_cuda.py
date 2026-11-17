"""Explicit source CUDA0/1 resource tests; CPU stubs only, no driver/model/lock-host queries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_proof_multi_cuda as multi
from scripts.ci import model_proof_cuda_primitives as primitives

ROLE0 = 'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff'
ROLE1 = 'GPU-00000000-0000-0000-0000-000000000001'
OTHER = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'


def request():
    common = {'schemaVersion': 1, 'backend': 'windows-cuda', 'deadlineSeconds': 5, 'pollMillis': 10,
              'termGraceMillis': 50, 'hostFreeReserveBytes': 100, 'childFootprintCapBytes': 10000,
              'stdoutCapBytes': 10000, 'stderrCapBytes': 10000, 'eventCapBytes': 10000}
    return {'selection': 'candle-llm/shard/' + multi.CASE['fqn'],
            'policy': {**common, 'cudaDeviceUuid': ROLE0, 'gpuFreeReserveBytes': 200, 'childGpuCapBytes': 400},
            'cuda_device_uuids': [ROLE0, ROLE1],
            'additional_cuda_policies': [{**common, 'cudaDeviceUuid': ROLE1, 'gpuFreeReserveBytes': 300, 'childGpuCapBytes': 500}]}


def row():
    return {**multi.CASE, 'backend': 'candle-cuda', 'cargo_features': ['model-inference-proof', 'cuda'],
            'required_cuda_ordinals': [0, 1]}


class MultiCudaResources(unittest.TestCase):
    def test_request_requires_explicit_distinct_ordered_uuid_policies_not_discovery_or_gpu1_fallback(self):
        value = request()
        self.assertEqual(multi.validate_request(value, 'candle-cuda'), (ROLE0, ROLE1))
        self.assertEqual(multi.bind_rows((row(),), value, 'candle-cuda'), (ROLE0, ROLE1))
        mutations = [lambda value: value.pop('additional_cuda_policies'),
                     lambda value: value.update(cuda_device_uuids=[ROLE0, ROLE0]),
                     lambda value: value.update(cuda_device_uuids=[ROLE1, ROLE0]),
                     lambda value: value['additional_cuda_policies'][0].update(cudaDeviceUuid=OTHER),
                     lambda value: value['additional_cuda_policies'][0].update(hostFreeReserveBytes=101),
                     lambda value: value['additional_cuda_policies'][0].pop('childGpuCapBytes')]
        for mutate in mutations:
            value = request(); mutate(value)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                multi.validate_request(value, 'candle-cuda')
        with self.assertRaises(ValueError):
            multi.validate_request(request(), 'candle-cpu')
        with self.assertRaises(ValueError):
            multi.bind_rows((row(),), {'selection': 'case', 'policy': request()['policy']}, 'candle-cuda')
        with self.assertRaises(ValueError):
            multi.bind_rows(({**row(), 'fqn': 'unrelated::case'},), request(), 'candle-cuda')
        missing = row(); missing.pop('required_cuda_ordinals')
        with self.assertRaises(ValueError):
            multi.bind_rows((missing,), request(), 'candle-cuda')

    def _claim(self, temporary, *, swapped=False, fail_post=None, raise_body=False):
        directory = Path(temporary).resolve()
        environment = {'PUBLIC': str(directory), 'CUDA_VISIBLE_DEVICES': ROLE0 + ',' + ROLE1,
                       'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        original = dict(environment)
        locked = set(); post = []; yielded = []
        def locking(fd, operation, count):
            if operation == 1:
                locked.add(fd)
            else:
                locked.discard(fd)
        locker = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
        devices = [{'uuid': ROLE0, 'index': 7}, {'uuid': ROLE1, 'index': 3}]
        def inventory(*_args):
            return devices, []
        admitted = {}
        def baseline(device, out, phase, required_mib):
            self.assertTrue(locked)
            admitted[device['uuid']] = required_mib
            return device, {'uuid': device['uuid'], 'MOCK': True}
        def verify(device, out, phase, before, **options):
            # Release checks identity/counters only; foreign tenants and their contexts may change.
            self.assertEqual(options['require_stable_contexts'], False)
            self.assertTrue(options['permit_bounded_growth'])
            self.assertTrue(options['idle_check']({'total_mib': 10, 'used_mib': 9, 'free_mib': 1, 'utilization_percent': 100},
                                                  [[device['uuid'], '1', 'app', 'C']]))
            post.append(device['uuid'])
            self.assertEqual(len(locked), 2, 'both leases must remain held through every post-idle attempt')
            if device['uuid'] == fail_post:
                raise RuntimeError('MOCK post-idle failure')
        real_guard = primitives.component('veda_cuda_guard.py', primitives.GUARD_SHA)
        fake_guard = SimpleNamespace(command=lambda *_: None, inventory=inventory, has_headroom=lambda *_: True,
                                     counters_valid=real_guard.counters_valid,
                                     sample_headroom=baseline, sample_baseline=verify)
        original_resolve = primitives.resolve_binding
        def resolve(*args, **kwargs):
            self.assertEqual(len(locked), 2)
            def run(argv, **options):
                self.assertEqual(options['env'], environment)
                actual = [ROLE1, ROLE0] if swapped else [ROLE0, ROLE1]
                packet = {'schema_version': 1, 'context_created': False,
                    'abi_source': {'crate': 'cudarc', 'version': '0.19.7', 'checksum': '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'},
                    'visibility': {'CUDA_VISIBLE_DEVICES': environment['CUDA_VISIBLE_DEVICES'], 'CUDA_DEVICE_ORDER': environment['CUDA_DEVICE_ORDER']},
                    'devices': [{'ordinal': index, 'uuid': uuid} for index, uuid in enumerate(actual)]}
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(packet))
            return original_resolve(*args, **kwargs, run=run)
        policies = tuple(SimpleNamespace(cuda_device_uuid=uuid, deadline_seconds=5,
                                         gpu_free_reserve_bytes=reserve, child_gpu_cap_bytes=cap)
                         for uuid, reserve, cap in ((ROLE0, 2 << 30, 4 << 30), (ROLE1, 3 << 30, 5 << 30)))
        with patch.object(multi, 'os', SimpleNamespace(name='nt', fstat=os.fstat)), \
                patch.dict(sys.modules, {'msvcrt': locker}), \
                patch.object(primitives, 'component', return_value=fake_guard), \
                patch.object(primitives, 'resolve_binding', side_effect=resolve):
            try:
                with multi.execution_claim(row(), policies, request(), directory, environment) as record:
                    yielded.append(record)
                    self.assertEqual(record['approved_uuid_roles'], [ROLE0, ROLE1])
                    self.assertEqual(record['lock_order'], [ROLE1, ROLE0])
                    self.assertEqual([entry['uuid'] for entry in record['actual_ordinal_binding']['mapping']], [ROLE0, ROLE1])
                    self.assertFalse(record['post_execution_verified'])
                    if raise_body:
                        raise ValueError('MOCK bounded GPU2 cap violation after owned-tree drain')
            except (ValueError, RuntimeError) as error:
                return yielded, post, locked, str(error), environment == original
        # Each UUID is admitted on its own policy's reserve + child cap, in MiB.
        self.assertEqual(admitted, {ROLE0: 6144, ROLE1: 8192})
        return yielded, post, locked, None, environment == original

    def test_sorted_lock_order_never_changes_source_ordinal_roles_and_every_uuid_postverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            records, post, locked, error, unchanged = self._claim(temporary)
            self.assertIsNone(error)
            self.assertTrue(records[0]['post_execution_verified'])
            self.assertEqual(set(post), {ROLE0, ROLE1})
            self.assertFalse(locked)
            self.assertTrue(unchanged)

    def test_swapped_driver_roles_reject_before_model_and_no_fallback_or_leaked_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            records, post, locked, error, unchanged = self._claim(temporary, swapped=True)
            self.assertEqual(records, [])
            self.assertIn('ordered CUDA0/1 UUID mapping', error)
            self.assertEqual(set(post), {ROLE0, ROLE1})
            self.assertFalse(locked)
            self.assertTrue(unchanged)

    def test_secondary_postidle_failure_cannot_seal_and_still_attempts_both_devices(self):
        for failed in (ROLE0, ROLE1):
            with self.subTest(uuid=failed), tempfile.TemporaryDirectory() as temporary:
                records, post, locked, error, _ = self._claim(temporary, fail_post=failed)
                self.assertIn('not every approved CUDA UUID', error)
                self.assertFalse(records[0]['post_execution_verified'])
                self.assertEqual(set(post), {ROLE0, ROLE1})
                self.assertFalse(locked)

    def test_bounded_body_failure_keeps_case_unaccepted_and_releases_both_after_postchecks(self):
        with tempfile.TemporaryDirectory() as temporary:
            records, post, locked, error, _ = self._claim(temporary, raise_body=True)
            self.assertIn('GPU2 cap violation', error)
            self.assertEqual(set(post), {ROLE0, ROLE1})
            self.assertFalse(locked)
            self.assertTrue(records[0]['post_execution_verified'], 'resource cleanup is not body/model acceptance')

    def test_claim_waits_under_the_lock_for_headroom_and_quiet_and_logs_the_real_reason(self):
        guard = primitives.component('veda_cuda_guard.py', primitives.GUARD_SHA)
        quiet = {'uuid': ROLE0, 'index': 1, 'total_mib': 97887, 'used_mib': 40000, 'free_mib': 57887, 'utilization_percent': 10}
        short = dict(quiet, used_mib=96000, free_mib=1887)
        busy = dict(quiet, utilization_percent=80)
        broken = dict(quiet, free_mib=None)
        rows = [[ROLE0, '7', 'desktop-app', 'C'], [ROLE0, '8', 'x', 'unknown']]
        # lock held -> short -> invalid counters -> busy window -> quiet window
        snapshots = iter([([quiet], rows), ([short], rows), ([broken], rows), ([quiet], rows), ([busy], rows),
                          ([quiet], rows)] + [([quiet], rows)] * 5)
        lock_calls = []
        def locking(_fd, operation, _count):
            lock_calls.append(operation)
            if len(lock_calls) == 1:
                raise OSError('MOCK held by another CI job')
        locker = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
        clock = iter(range(0, 10**6, 100))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(guard, 'inventory', side_effect=lambda *_args, **_kw: next(snapshots)), \
                patch.object(guard.time, 'sleep'), patch('sys.stderr') as stderr:
            checked, handles, _, _ = primitives.claim_uuids(
                [ROLE0], guard, Path(temporary), 10**9, Path(temporary).resolve(), locker,
                {ROLE0: 4 << 30}, clock=lambda: next(clock), sleep=lambda _s: None)
            for handle in handles:
                handle.close()
        self.assertEqual(checked[0]['uuid'], ROLE0)
        lines = [line for line in ''.join(call.args[0] for call in stderr.write.call_args_list).splitlines() if line]
        self.assertEqual(len(lines), 4)
        self.assertIn('lock is held by another CI job', lines[0])
        self.assertIn('1887 MiB free < 4096 MiB required', lines[1])
        self.assertIn('invalid memory counters', lines[2])
        self.assertIn('utilization 80%', lines[3])
        # Every refusal after the lock is taken releases it again (lock, unlock pairs), then the
        # admitted claim keeps its final lock.
        self.assertEqual(lock_calls, [1, 1, 2, 1, 2, 1, 2, 1])

    def test_wait_notice_is_rate_limited(self):
        clock = iter([0, 10, 59, 60, 61])
        state = {}
        with patch('sys.stderr') as stderr:
            for _ in range(5):
                primitives.wait_notice(state, 'MOCK reason', clock=lambda: next(clock))
        self.assertEqual(sum('MOCK reason' in call.args[0] for call in stderr.write.call_args_list), 2)

if __name__ == '__main__':
    unittest.main()
