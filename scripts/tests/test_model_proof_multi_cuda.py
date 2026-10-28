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
        def baseline(device, out, phase):
            self.assertTrue(locked)
            return device, {'uuid': device['uuid'], 'MOCK': True}
        def verify(device, out, phase, before):
            post.append(device['uuid'])
            self.assertEqual(len(locked), 2, 'both leases must remain held through every post-idle attempt')
            if device['uuid'] == fail_post:
                raise RuntimeError('MOCK post-idle failure')
        fake_guard = SimpleNamespace(command=lambda *_: None, inventory=inventory, idle=lambda *_: True,
                                     sample_baseline=baseline, verify_idle=verify)
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
        policies = tuple(SimpleNamespace(cuda_device_uuid=uuid, deadline_seconds=5) for uuid in (ROLE0, ROLE1))
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


if __name__ == '__main__':
    unittest.main()
