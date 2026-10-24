"""Exact child CUDA visibility binding with CPU stubs only; never load/query a real driver."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_proof_resources as resources
from scripts.ci import windows_cuda_metadata as metadata
from scripts.release import veda_cuda_guard as guard

APPROVED = 'GPU-12345678-1234-5678-1234-567812345678'
OTHER = 'GPU-87654321-4321-8765-4321-876543218765'


def case():
    source = 'crates/media/candle-gen/candle-gen-iris/tests/train_distributed.rs'
    return {'package': 'candle-gen-iris', 'target': 'integration', 'target_kind': 'test',
            'fqn': 'train_distributed::a_rank_without_its_cuda_device_is_refused_unless_sharing_is_opted_in',
            'source': source, 'backend': 'candle-cuda', 'cargo_features': ['model-inference-proof', 'cuda'],
            'child_cuda_visibility': {'source_identity': source + '::sharing-test/child-env-override',
                                      'visible_devices': '0', 'device_order': 'inherit', 'ordinal': 0}}


def packet(environment, selected=APPROVED):
    return {'schema_version': 1, 'context_created': False,
            'abi_source': {'crate': 'cudarc', 'version': '0.19.7', 'checksum': resources.CUDA_METADATA_ABI_CHECKSUM},
            'visibility': {'CUDA_VISIBLE_DEVICES': environment.get('CUDA_VISIBLE_DEVICES'),
                           'CUDA_DEVICE_ORDER': environment.get('CUDA_DEVICE_ORDER')},
            'devices': [{'ordinal': 0, 'uuid': selected}]}


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
    def __call__(self, *args):
        return self.implementation(*args)


class FakeDriver:
    def __init__(self, raw_uuid):
        self.calls = []
        def init(flags):
            self.calls.append(('cuInit', flags)); return 0
        def count(pointer):
            self.calls.append(('cuDeviceGetCount',))
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 1
            return 0
        def device(pointer, ordinal):
            self.calls.append(('cuDeviceGet', ordinal))
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 31
            return 0
        def identifier(pointer, device):
            self.calls.append(('cuDeviceGetUuid', device))
            ctypes.memmove(pointer, raw_uuid, 16)
            return 0
        for name, function in (('cuInit', init), ('cuDeviceGetCount', count), ('cuDeviceGet', device), ('cuDeviceGetUuid', identifier)):
            setattr(self, name, FakeFunction(function))


class ChildCudaVisibility(unittest.TestCase):
    def test_primitive_is_exact_verified_app_bytes_and_reads_all16_uuid_bytes_with_cpu_abi_stub(self):
        raw = (ROOT / 'scripts/ci/windows_cuda_metadata.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), resources.CUDA_METADATA_HELPER_SHA256)
        binary = bytes.fromhex('12003400560078009a00bc00de00f000')
        driver = FakeDriver(binary)
        result = metadata.visible_devices(library=driver)
        self.assertIs(result['context_created'], False)
        self.assertEqual(result['devices'], [{'ordinal': 0, 'uuid': 'GPU-' + str(uuid.UUID(bytes=binary))}])
        self.assertEqual([call[0] for call in driver.calls], ['cuInit', 'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetUuid'])
        self.assertEqual(driver.cuInit.argtypes, [ctypes.c_uint])
        self.assertIs(driver.cuDeviceGetUuid.restype, ctypes.c_uint32)

    def test_exact_effective_child_env_is_fresh_and_parent_not_retargeted_including_unset_order(self):
        for order in (None, 'PCI_BUS_ID', 'FASTEST_FIRST'):
            environment = {'CUDA_VISIBLE_DEVICES': APPROVED, 'OTHER': 'preserved'}
            if order is not None:
                environment['CUDA_DEVICE_ORDER'] = order
            original = dict(environment)
            def run(argv, **kwargs):
                self.assertEqual(argv[:2], [sys.executable, '-I'])
                self.assertTrue(argv[-1].endswith('windows_cuda_metadata.py'))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
                self.assertEqual(kwargs['env'].get('CUDA_DEVICE_ORDER'), order)
                self.assertTrue(kwargs['check'])
                self.assertLessEqual(kwargs['timeout'], 30)
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(packet(kwargs['env'])))
            with self.subTest(order=order), patch.object(resources.subprocess, 'run', side_effect=run) as child:
                result = resources.child_visibility_binding(case(), environment, APPROVED, 5)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(result['metadata']['devices'][0]['uuid'], APPROVED)
                self.assertEqual(environment, original)

    def test_unknown_scope_flags_ordinals_or_missing_binding_are_rejected(self):
        mutations = [lambda row: row.pop('child_cuda_visibility'),
                     lambda row: row['child_cuda_visibility'].update(visible_devices='1'),
                     lambda row: row['child_cuda_visibility'].update(device_order='PCI_BUS_ID'),
                     lambda row: row['child_cuda_visibility'].update(ordinal=True),
                     lambda row: row['child_cuda_visibility'].update(arbitrary_environment={'ALLOW_GPU': '1'}),
                     lambda row: row.update(fqn='unrelated::case'),
                     lambda row: row.update(backend='candle-cpu'),
                     lambda row: row.update(cargo_features=['model-inference-proof'])]
        for mutate in mutations:
            row = case(); mutate(row)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                resources.validate_child_visibility(row)
        resources.validate_child_visibility({'package': 'legacy-other'})

    def test_missing_ambiguous_wrong_uuid_context_abi_visibility_and_failed_probe_never_fall_back(self):
        environment = {'CUDA_VISIBLE_DEVICES': APPROVED, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        child_env = dict(environment, CUDA_VISIBLE_DEVICES='0')
        mutations = [lambda value: value.update(devices=[]),
                     lambda value: value['devices'].append(copy.deepcopy(value['devices'][0])),
                     lambda value: value['devices'][0].update(uuid=OTHER),
                     lambda value: value['devices'][0].update(ordinal=True),
                     lambda value: value.update(context_created=True),
                     lambda value: value.update(schema_version=True),
                     lambda value: value['abi_source'].update(checksum='unverified'),
                     lambda value: value['visibility'].update(CUDA_VISIBLE_DEVICES=APPROVED),
                     lambda value: value['visibility'].update(CUDA_DEVICE_ORDER=None)]
        for mutate in mutations:
            value = packet(child_env); mutate(value)
            with self.subTest(mutation=mutate), patch.object(resources.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(value))):
                with self.assertRaises(ValueError):
                    resources.child_visibility_binding(case(), environment, APPROVED, 5)
        with patch.object(resources.subprocess, 'run', side_effect=subprocess.TimeoutExpired('MOCK metadata only', 1)):
            with self.assertRaises(ValueError):
                resources.child_visibility_binding(case(), environment, APPROVED, 5)
        with patch.object(resources.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                resources.child_visibility_binding(case(), environment, APPROVED, 0)
            run.assert_not_called()

    def test_original_ordinal0_can_map_nvml_index1_but_must_match_exact_locked_uuid(self):
        for selected_uuid in (APPROVED, OTHER):
            with self.subTest(driver_uuid=selected_uuid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                environment = {'PUBLIC': str(root), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': APPROVED,
                               'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
                original = dict(environment)
                locked = []
                def locking(_descriptor, operation, _count):
                    locked[:] = [operation == 1]
                fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
                fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
                nvml_device = {'uuid': APPROVED, 'index': 1}
                policy = SimpleNamespace(cuda_device_uuid=APPROVED, deadline_seconds=5, poll_millis=1)
                def run(argv, **kwargs):
                    self.assertEqual(locked, [True], 'mapping must happen INSIDE approved UUID lease')
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
                    return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(packet(kwargs['env'], selected_uuid)))
                with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                        patch.object(guard, 'inventory', return_value=([nvml_device], [])), \
                        patch.object(guard, 'idle', return_value=True), \
                        patch.object(guard, 'sample_baseline', return_value=(nvml_device, {'MOCK': True})) as baseline, \
                        patch.object(guard, 'verify_idle') as post_idle, \
                        patch.object(resources.subprocess, 'run', side_effect=run):
                    if selected_uuid == APPROVED:
                        with resources.execution_claim('candle-cuda', policy, root, environment, constraints=case()) as receipt:
                            self.assertEqual(locked, [True])
                            self.assertEqual(receipt['physical_index'], 1)
                            self.assertEqual(receipt['child_visibility_binding']['metadata']['devices'][0]['ordinal'], 0)
                        self.assertTrue(receipt['post_execution_verified'])
                        baseline.assert_called_once()
                        post_idle.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError, 'other than the approved leased device'):
                            with resources.execution_claim('candle-cuda', policy, root, environment, constraints=case()):
                                self.fail('wrong UUID must never reach model launch')
                        baseline.assert_not_called()
                        post_idle.assert_not_called()
                self.assertEqual(locked, [False])
                self.assertEqual(environment, original)


if __name__ == '__main__':
    unittest.main()
