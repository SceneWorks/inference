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
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_proof_resources as resources
from scripts.ci import windows_cuda_metadata as metadata
from scripts.release import veda_cuda_guard as guard

APPROVED = 'GPU-12345678-1234-5678-1234-567812345678'
OTHER = 'GPU-87654321-4321-8765-4321-876543218765'
GPU_POLICY = SimpleNamespace(cuda_device_uuid=APPROVED, deadline_seconds=5, poll_millis=1,
                             gpu_free_reserve_bytes=16 * 1024**3, child_gpu_cap_bytes=64 * 1024**3)


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
                nvml_device = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887,
                               'used_mib': 1884, 'free_mib': 96003, 'utilization_percent': 0}
                policy = GPU_POLICY
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

    def test_rejected_fresh_claim_records_bounded_driver_predicates_without_process_names(self):
        # CI_IMPACT: fake Windows lock and inventory only; no GPU, process, model, or Cargo.
        for card, processes, expected in (
            ({'uuid': APPROVED, 'index': 1, 'total_mib': 98304, 'used_mib': 16385,
              'free_mib': 81919, 'utilization_percent': 0},
             [[APPROVED, '123', 'private.exe', 'C']], 'free_memory_above_policy_headroom'),
            ({'uuid': APPROVED, 'index': 1, 'total_mib': 98304, 'used_mib': 1024,
              'free_mib': 97280, 'utilization_percent': 0},
             [[APPROVED, '123', 'private.exe', 'unknown']], 'selected_process_types_allowed'),
            ({'uuid': OTHER, 'index': 1, 'total_mib': 98304, 'used_mib': 1024,
              'free_mib': 97280, 'utilization_percent': 0}, [], 'unique_uuid'),
        ):
            with self.subTest(predicate=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                environment = {'PUBLIC': str(root), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': APPROVED}
                policy = GPU_POLICY
                locks = []
                def locking(_descriptor, operation, _count):
                    locks.append(operation)
                fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
                fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
                with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                        patch.object(guard, 'inventory', return_value=([card], processes)) as inventory, \
                        patch.object(guard, 'sample_baseline') as baseline:
                    with self.assertRaises(resources.CudaClaimRejected) as caught:
                        with resources.execution_claim('candle-cuda', policy, root, environment):
                            self.fail('refused claim must never execute a body')
                inventory.assert_called_once_with(root, 'model-proof-claimed-device')
                baseline.assert_not_called()
                self.assertEqual(locks, [1, 2])
                diagnostic = caught.exception.diagnostic
                self.assertFalse(diagnostic['predicates'][expected])
                self.assertEqual(diagnostic['requested_uuid'], APPROVED)
                self.assertNotIn('private.exe', json.dumps(diagnostic))
                self.assertLess(len(json.dumps(diagnostic)), 4096)

        many = [[APPROVED, str(i), 'private.exe', 'C'] for i in range(100)]
        diagnostic = resources.cuda_claim_diagnostic(APPROVED, [card | {'uuid': APPROVED}], many, GPU_POLICY)
        self.assertEqual(diagnostic['selected_process_count'], 100)
        self.assertEqual(len(diagnostic['selected_processes']), 64)
        self.assertTrue(diagnostic['process_list_truncated'])

    def test_release_refusal_reports_only_existing_bounded_sample_predicates(self):
        # CI_IMPACT: synthetic saved driver rows only; no GPU, model or process execution.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        rows = [[APPROVED, '123', 'private-process-name', 'C']]
        baseline = {'process_identities': [(APPROVED, '123', 'C')]}
        samples = [
            {'sample_number': 1, 'devices': [dict(card, utilization_percent=1)], 'processes': rows},
            {'sample_number': 2, 'devices': [dict(card, used_mib=15968, free_mib=81919)], 'processes': rows},
            {'sample_number': 3, 'devices': [card], 'processes': [[APPROVED, '124', 'other-private-name', 'C']]},
            {'sample_number': 4, 'devices': [card], 'processes': rows},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / 'model-proof-release-samples.json'
            path.write_text(json.dumps(samples), encoding='utf-8')
            facts = resources.cuda_release_diagnostic(output, APPROVED, 1, GPU_POLICY, baseline)
            self.assertEqual(facts['status'], 'retained-existing-samples')
            self.assertEqual(facts['sample_count'], 4)
            self.assertFalse(facts['samples'][0]['predicates']['selected_gpu_zero_utilization'])
            self.assertFalse(facts['samples'][1]['predicates']['free_memory_above_policy_headroom'])
            self.assertFalse(facts['samples'][2]['context_identities_match_admission'])
            self.assertTrue(facts['samples'][3]['predicates']['proof_policy_headroom_and_inactive_contexts'])
            self.assertNotIn('private-process-name', json.dumps(facts))
            self.assertNotIn('other-private-name', json.dumps(facts))
            path.write_bytes(b'x' * (512 * 1024 + 1))
            self.assertEqual(resources.cuda_release_diagnostic(output, APPROVED, 1, GPU_POLICY, baseline),
                             {'status': 'missing-or-over-budget'})
            path.unlink()
            path.symlink_to(output / 'missing')
            self.assertEqual(resources.cuda_release_diagnostic(output, APPROVED, 1, GPU_POLICY, baseline),
                             {'status': 'missing-or-over-budget'})

    def test_release_refusal_keeps_unsealed_claim_and_existing_sample_diagnostic(self):
        # CI_IMPACT: mocked physical controller only; no driver, GPU, model or process execution.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        rows = [[APPROVED, '123', 'private-process-name', 'C']]
        baseline = {'uuid': APPROVED, 'reserved_baseline_mib': 1884,
                    'process_identities': [(APPROVED, '123', 'C')]}
        def refuse(_card, output, _phase, _baseline, **_kwargs):
            (output / 'model-proof-release-samples.json').write_text(json.dumps([
                {'sample_number': 1, 'devices': [dict(card, utilization_percent=1)], 'processes': rows},
            ]), encoding='utf-8')
            raise RuntimeError('physical GPU occupied during model-proof-release')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner'}
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
            with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                    patch.object(guard, 'inventory', return_value=([card], rows)), \
                    patch.object(guard, 'sample_baseline', return_value=(card, baseline)), \
                    patch.object(guard, 'verify_idle', side_effect=refuse), patch.object(guard.time, 'sleep'):
                with self.assertRaises(resources.CudaReleaseRejected) as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, output, environment,
                                                   constraints={'required_cuda_physical_index': 1}) as claim:
                        self.assertFalse(claim['post_execution_verified'])
            self.assertFalse(claim['post_execution_verified'])
            self.assertEqual(caught.exception.diagnostic['sample_count'], 1)
            self.assertEqual(caught.exception.diagnostic['settle_attempt_count'], 5)
            self.assertFalse(caught.exception.diagnostic['samples'][0]['predicates']['selected_gpu_zero_utilization'])
            self.assertNotIn('private-process-name', json.dumps(caught.exception.diagnostic))

    def test_transient_selected_gpu_utilization_requires_five_clean_admission_and_release_samples(self):
        # CI_IMPACT: mocked SMI snapshots prove bounded settle without learned work or device queries.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        warm = dict(card, utilization_percent=2)
        rows = [[APPROVED, '123', 'private-process-name', 'C']]
        snapshots = [([warm], rows)] + [([card], rows)] * 5 + [([warm], rows)] + [([card], rows)] * 5
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner'}
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
            with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                    patch.object(guard, 'inventory', side_effect=snapshots) as inventory, patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', GPU_POLICY, output, environment,
                                               constraints={'required_cuda_physical_index': 1}) as claim:
                    self.assertFalse(claim['post_execution_verified'])
            self.assertEqual(inventory.call_count, 12)
            self.assertTrue(claim['post_execution_verified'])
            self.assertEqual(claim['physical_index'], 1)
            self.assertEqual(claim['device_uuid'], APPROVED)
            self.assertEqual(len(json.loads((output / 'model-proof-admission-samples.json').read_text())), 5)
            self.assertEqual(len(json.loads((output / 'model-proof-release-samples.json').read_text())), 5)

    def test_persistent_selected_gpu_utilization_rejects_after_bounded_retries(self):
        # CI_IMPACT: one controlled precheck plus five synthetic failed samples; never launches model work.
        warm = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 2}
        rows = [[APPROVED, '123', 'private-process-name', 'C']]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner'}
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
            with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                    patch.object(guard, 'inventory', return_value=([warm], rows)) as inventory, \
                    patch.object(guard.time, 'sleep'):
                with self.assertRaises(resources.CudaClaimRejected) as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, output, environment,
                                                   constraints={'required_cuda_physical_index': 1}):
                        self.fail('persistent GPU activity must never execute model body')
            self.assertEqual(inventory.call_count, 6)
            self.assertEqual(caught.exception.diagnostic['admission_samples']['sample_count'], 1)
            self.assertEqual(caught.exception.diagnostic['admission_samples']['settle_attempt_count'], 5)
            self.assertTrue(caught.exception.diagnostic['admission_samples']['samples'][0]['transient_utilization_only'])

    def test_settle_deadline_expires_without_additional_driver_probe(self):
        # CI_IMPACT: pure deadline control; no driver, CUDA or model work.
        probe = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'settle deadline expired'):
                resources.settled_cuda_samples(probe, Path(temporary), 'model-proof-admission',
                                               APPROVED, 1, GPU_POLICY, {'process_identities': []}, 0)
        probe.assert_not_called()

    def test_settle_never_retries_capacity_loss_or_unknown_context(self):
        # CI_IMPACT: synthetic already-sampled refusals; no GPU or model work.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        baseline = {'process_identities': [(APPROVED, '123', 'C')]}
        for changed_card, rows in (
            (dict(card, used_mib=15968, free_mib=81919), [[APPROVED, '123', 'opaque', 'C']]),
            (card, [[APPROVED, '123', 'opaque', 'unknown']]),
        ):
            with self.subTest(card=changed_card, rows=rows), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                def refuse():
                    (output / 'model-proof-release-samples.json').write_text(json.dumps([
                        {'sample_number': 1, 'devices': [changed_card], 'processes': rows},
                    ]), encoding='utf-8')
                    raise RuntimeError('physical GPU occupied during model-proof-release')
                probe = Mock(side_effect=refuse)
                with self.assertRaisesRegex(RuntimeError, 'physical GPU occupied'):
                    resources.settled_cuda_samples(probe, output, 'model-proof-release', APPROVED, 1,
                                                   GPU_POLICY, baseline, 10**18)
                probe.assert_called_once()

    def test_proof_admits_stable_inactive_gpu1_context_while_gpu0_is_busy(self):
        # CI_IMPACT: fake driver rows only; no CUDA, GPU query, model, or process execution.
        gpu0 = {'uuid': OTHER, 'index': 0, 'total_mib': 97887, 'used_mib': 70000,
                'free_mib': 27887, 'utilization_percent': 90}
        gpu1 = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        rows = [[OTHER, '987', 'SceneWorks on GPU0', 'C'],
                [APPROVED, '123', 'context only', 'C'],
                [APPROVED, '124', 'second inactive context', 'C'],
                [APPROVED, '456', 'desktop', 'C+G']]
        self.assertFalse(guard.idle(gpu1, rows), 'default release guard remains strict')
        self.assertTrue(resources.proof_cuda_idle(gpu1, rows, GPU_POLICY))
        baseline = {'uuid': APPROVED, 'reserved_baseline_mib': 1884,
                    'process_identities': resources.proof_cuda_contexts(APPROVED, rows)}
        samples = [gpu1 | {'used_mib': used, 'free_mib': 97887 - used}
                   for used in (1884, 1885, 1883, 1886, 1884)]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(guard, 'inventory', side_effect=[([gpu0, selected], rows) for selected in samples]) as inventory, \
                patch.object(guard.time, 'sleep'):
            checked, observed = guard.sample_baseline(gpu1, Path(temporary), 'MOCK', baseline,
                                                      idle_check=lambda card, current: resources.proof_cuda_idle(card, current, GPU_POLICY),
                                                      identity_key=resources.proof_cuda_contexts,
                                                      permit_bounded_growth=True)
            self.assertEqual(checked, samples[-1])
            self.assertEqual(observed, baseline)
            self.assertEqual(observed['reserved_baseline_mib'], 1886)
            self.assertEqual(inventory.call_count, 5)

        for bad_gpu1, bad_rows in (
            (gpu1 | {'utilization_percent': 1}, rows),
            (gpu1 | {'used_mib': 16385, 'free_mib': 81919}, rows),
            (gpu1, rows + [[APPROVED, '789', 'opaque', 'unknown']]),
            (gpu1, rows + [[APPROVED, '', 'unidentified', 'C']]),
            (gpu1 | {'free_mib': None}, rows),
        ):
            self.assertFalse(resources.proof_cuda_idle(bad_gpu1, bad_rows, GPU_POLICY))

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(guard, 'inventory', side_effect=[([gpu0, gpu1], rows),
                             ([gpu0, gpu1], [row if row[1] != '123' else [APPROVED, '789', 'new context', 'C']
                                            for row in rows])]), \
                patch.object(guard.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'baseline changed'):
                guard.sample_baseline(gpu1, Path(temporary), 'MOCK', baseline,
                                      idle_check=lambda card, current: resources.proof_cuda_idle(card, current, GPU_POLICY),
                                      identity_key=resources.proof_cuda_contexts,
                                      permit_bounded_growth=True)

        for changed in ([gpu0], [gpu0, gpu1, gpu1]):
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(guard, 'inventory', return_value=(changed, rows)), \
                    patch.object(guard.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    guard.sample_baseline(gpu1, Path(temporary), 'MOCK', baseline,
                                          idle_check=lambda card, current: resources.proof_cuda_idle(card, current, GPU_POLICY),
                                          identity_key=resources.proof_cuda_contexts,
                                          permit_bounded_growth=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            environment = {'PUBLIC': str(root), 'RUNNER_NAME': 'MOCK runner'}
            locks = []
            def locking(_descriptor, operation, _count):
                locks.append(operation)
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
            with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                    patch.object(guard, 'inventory', return_value=([gpu0, gpu1], rows)) as inventory, \
                    patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', GPU_POLICY, root, environment,
                                               constraints={'required_cuda_physical_index': 1}) as receipt:
                    self.assertEqual(receipt['physical_index'], 1)
                    self.assertFalse(receipt['post_execution_verified'])
                self.assertTrue(receipt['post_execution_verified'])
                self.assertEqual(receipt['admission_baseline']['reserved_baseline_mib'], 1884)
                self.assertEqual(inventory.call_count, 11)  # claim + five admission + five cleanup
                self.assertEqual(locks, [1, 2])


if __name__ == '__main__':
    unittest.main()
