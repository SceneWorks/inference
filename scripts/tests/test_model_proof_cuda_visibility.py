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
from scripts.ci import model_inference_proofs as proof
from scripts.ci import windows_cuda_metadata as metadata
from scripts.release import veda_cuda_guard as guard

APPROVED = 'GPU-12345678-1234-5678-1234-567812345678'
OTHER = 'GPU-87654321-4321-8765-4321-876543218765'
GPU_POLICY = SimpleNamespace(cuda_device_uuid=APPROVED, deadline_seconds=5, poll_millis=1,
                             gpu_free_reserve_bytes=16 * 1024**3, child_gpu_cap_bytes=64 * 1024**3,
                             # A job that states its 64 GiB device estimate: the card needs 80 GiB free.
                             claim_gpu_estimate=('test-stated-estimate', 64 * 1024**3))


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
                        patch.object(guard.time, 'sleep'), \
                        patch.object(resources.subprocess, 'run', side_effect=run):
                    if selected_uuid == APPROVED:
                        with resources.execution_claim('candle-cuda', policy, root, environment, constraints=case()) as receipt:
                            self.assertEqual(locked, [True])
                            self.assertEqual(receipt['physical_index'], 1)
                            self.assertEqual(receipt['child_visibility_binding']['metadata']['devices'][0]['ordinal'], 0)
                        self.assertTrue(receipt['post_execution_verified'])
                        self.assertTrue((root / 'model-proof-admission-samples.json').is_file())
                        self.assertTrue((root / 'model-proof-release-samples.json').is_file())
                    else:
                        with self.assertRaisesRegex(ValueError, 'other than the approved leased device'):
                            with resources.execution_claim('candle-cuda', policy, root, environment, constraints=case()):
                                self.fail('wrong UUID must never reach model launch')
                        self.assertFalse((root / 'model-proof-admission-samples.json').exists())
                self.assertEqual(locked, [False])
                self.assertEqual(environment, original)

    def _fake_windows(self, locks=None):
        def locking(_descriptor, operation, _count):
            if locks is not None:
                locks.append(operation)
        return (patch.object(resources, 'os', SimpleNamespace(name='nt', fstat=os.fstat)),
                patch.dict(sys.modules, {'msvcrt': SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)}))

    def test_default_claim_admits_foreign_compute_unknown_contexts_and_resident_model(self):
        # CI_IMPACT: fake Windows lock and inventory only. The owner's desktop app holds a
        # compute context and a loaded model on a display GPU; headroom alone decides admission.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 98304, 'used_mib': 12000,
                'free_mib': 86304, 'utilization_percent': 12}
        rows = [[APPROVED, '123', 'private.exe', 'C'], [APPROVED, '124', 'opaque', 'unknown'],
                [APPROVED, '125', 'dwm.exe', 'C+G']]
        self.assertFalse(guard.idle(card, rows))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            locks = []
            os_patch, lock_patch = self._fake_windows(locks)
            with os_patch, lock_patch, patch.object(guard, 'inventory', return_value=([card], rows)) as inventory, \
                    patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', GPU_POLICY, root,
                                               {'PUBLIC': str(root), 'RUNNER_NAME': 'MOCK runner'}) as receipt:
                    self.assertEqual(receipt['activity_profile'], resources.SHARED_CUDA_HEADROOM_ACTIVITY)
                    self.assertEqual(receipt['admission_utilization_percent'], [12] * 5)
                    self.assertFalse(receipt['isolated_benchmark_claim'])
            self.assertTrue(receipt['post_execution_verified'])
            self.assertTrue(receipt['lease_released'])
            self.assertEqual(locks, [1, 2])
            self.assertEqual(inventory.call_count, 11)  # first poll + five admission + five release
            # Only the exact selected card's counters are read; no host process census.
            self.assertEqual({call.kwargs.get('selected_uuid') for call in inventory.call_args_list}, {APPROVED})
            self.assertNotIn('private.exe', json.dumps(receipt))

    def test_default_claim_refuses_only_missing_headroom_or_identity_and_never_waits_silently(self):
        # CI_IMPACT: fake Windows lock and inventory only; no GPU, process, model, or Cargo.
        short = {'uuid': APPROVED, 'index': 1, 'total_mib': 98304, 'used_mib': 30000,
                 'free_mib': 68304 - 16384 * 2, 'utilization_percent': 0}
        policy = SimpleNamespace(**{**vars(GPU_POLICY), 'deadline_seconds': 0.05, 'poll_millis': 10})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            locks = []
            os_patch, lock_patch = self._fake_windows(locks)
            now = [0.0]
            def advance(seconds):
                now[0] += seconds
            # Deterministic clock: each refused poll sleeps 10 ms of a 50 ms claim deadline.
            with os_patch, lock_patch, patch.object(guard, 'inventory', return_value=([short], [])), \
                    patch.object(resources.time, 'monotonic', side_effect=lambda: now[0]), \
                    patch.object(resources.time, 'sleep', side_effect=advance), \
                    patch.object(guard, 'sample_baseline') as baseline, patch('sys.stderr') as stderr:
                with self.assertRaises(resources.CudaClaimRejected) as caught:
                    with resources.execution_claim('candle-cuda', policy, root, {'PUBLIC': str(root)}):
                        self.fail('refused claim must never execute a body')
            baseline.assert_not_called()
            # The UUID lock is released between refused polls so other CI jobs can use the card.
            self.assertGreaterEqual(len(locks), 4)
            self.assertEqual(locks, [1, 2] * (len(locks) // 2))
            diagnostic = caught.exception.diagnostic
            self.assertFalse(diagnostic['predicates']['free_memory_above_policy_headroom'])
            self.assertIn('original CUDA claim deadline expired', diagnostic['admission_error'])
            self.assertGreaterEqual(diagnostic['admission_wait']['headroom_shortfalls'], 1)
            self.assertLess(len(json.dumps(diagnostic)), 4096)
            printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn('CUDA admission waiting: ' + APPROVED + ': 35536 MiB free < 81920 MiB required', printed)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            os_patch, lock_patch = self._fake_windows()
            other = dict(short, uuid=OTHER, free_mib=97280)
            with os_patch, lock_patch, patch.object(guard, 'inventory', return_value=([other], [])) as inventory:
                with self.assertRaises(resources.CudaClaimRejected) as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, root, {'PUBLIC': str(root)}):
                        self.fail('missing UUID must never execute a body')
            inventory.assert_called_once()
            self.assertFalse(caught.exception.diagnostic['predicates']['unique_uuid'])

        many = [[APPROVED, str(i), 'private.exe', 'C'] for i in range(100)]
        diagnostic = resources.cuda_claim_diagnostic(APPROVED, [short], many, GPU_POLICY)
        self.assertEqual(diagnostic['selected_process_count'], 100)
        self.assertEqual(len(diagnostic['selected_processes']), 64)
        self.assertTrue(diagnostic['process_list_truncated'])
        self.assertNotIn('private.exe', json.dumps(diagnostic))

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
        def refuse(_card, output, _phase, _baseline, **_kwargs):
            (output / 'model-proof-release-samples.json').write_text(json.dumps([
                {'sample_number': 1, 'devices': [dict(card, total_mib=48000)], 'processes': rows},
            ]), encoding='utf-8')
            raise RuntimeError('physical GPU occupied during model-proof-release')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            os_patch, lock_patch = self._fake_windows()
            with os_patch, lock_patch, patch.object(guard, 'inventory', return_value=([card], rows)), \
                    patch.object(guard, 'verify_idle', side_effect=refuse), patch.object(guard.time, 'sleep'):
                with self.assertRaises(resources.CudaReleaseRejected) as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, output, {'PUBLIC': str(output)},
                                                   constraints={'required_cuda_physical_index': 1}) as claim:
                        self.assertFalse(claim['post_execution_verified'])
            self.assertFalse(claim['post_execution_verified'])
            self.assertTrue(claim['lease_released'])
            diagnostic = caught.exception.diagnostic
            self.assertEqual(diagnostic['sample_count'], 1)
            self.assertEqual(diagnostic['settle_attempt_count'], 1)
            self.assertFalse(diagnostic['samples'][0]['post_owned_exit_primary_metadata_valid'])
            self.assertNotIn('private-process-name', json.dumps(diagnostic))

    def _release_with(self, release_inventory):
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        calls = []
        def inventory(out, label, deadline=None, *, selected_uuid=None):
            if not label.startswith('model-proof-release'):
                return [card], []
            calls.append(deadline - guard.time.monotonic() if deadline is not None else None)
            return release_inventory(card, len(calls))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        os_patch, lock_patch = self._fake_windows()
        with os_patch, lock_patch, patch.object(guard, 'inventory', side_effect=inventory), \
                patch.object(guard.time, 'sleep'):
            with resources.execution_claim('candle-cuda', GPU_POLICY, output, {'PUBLIC': str(output)},
                                           constraints={'required_cuda_physical_index': 1}) as claim:
                pass
        return claim, calls

    def test_release_query_timeout_after_a_valid_sample_is_a_diagnostic_and_the_proof_is_kept(self):
        # CI_IMPACT: mocked nvidia-smi timeouts only (run 44431476698); no driver, GPU or model.
        def timeout_after_first(card, call):
            if call == 1:
                return [card], []
            raise subprocess.TimeoutExpired(['nvidia-smi'], resources.RELEASE_QUERY_SECONDS)
        claim, calls = self._release_with(timeout_after_first)
        self.assertTrue(claim['post_execution_verified'])
        self.assertTrue(claim['lease_released'])
        verification = claim['release_metadata_verification']
        self.assertEqual(verification['status'], 'inconclusive-driver-query-timing')
        self.assertIn('timed out', verification['error'])
        self.assertEqual(verification['diagnostic']['sample_count'], 1)
        self.assertNotIn('release_utilization_percent', claim)
        # One valid sample, then the bounded retry; each query had the fixed budget, not a leftover window.
        self.assertEqual(len(calls), 1 + resources.RELEASE_QUERY_ATTEMPTS)
        self.assertTrue(all(resources.RELEASE_QUERY_SECONDS - 1 < budget <= resources.RELEASE_QUERY_SECONDS
                            for budget in calls))

    def test_release_with_no_retained_sample_is_refused(self):
        # CI_IMPACT: mocked nvidia-smi only. A driver hung after the run checked nothing: no seal.
        def hung(_card, _call):
            raise subprocess.TimeoutExpired(['nvidia-smi'], resources.RELEASE_QUERY_SECONDS)
        with self.assertRaisesRegex(resources.CudaReleaseRejected, 'timed out'):
            self._release_with(hung)

    def test_only_timing_failures_with_valid_retained_metadata_are_inconclusive(self):
        valid = {'samples': [{'post_owned_exit_primary_metadata_valid': True}]}
        invalid = {'samples': [{'post_owned_exit_primary_metadata_valid': True},
                               {'post_owned_exit_primary_metadata_valid': False}]}
        overrun = RuntimeError(resources.release_window_overrun('model-proof-release'))
        self.assertTrue(resources.release_check_inconclusive(overrun, valid))
        self.assertTrue(resources.release_check_inconclusive(TimeoutError('slow'), valid))
        self.assertFalse(resources.release_check_inconclusive(overrun, invalid))
        for empty in ({'status': 'missing-or-malformed'}, {'samples': []}):
            with self.subTest(diagnostic=empty):
                self.assertFalse(resources.release_check_inconclusive(TimeoutError('slow'), empty))
        for error in (RuntimeError('supervisor-owned Windows Job completion is unverified; no release seal'),
                      RuntimeError('selected physical GPU UUID absent or ambiguous; no device fallback'),
                      OSError('driver failed')):
            with self.subTest(error=error):
                self.assertFalse(resources.release_check_inconclusive(error, valid))

    def test_release_query_retry_recovers_and_metadata_is_verified(self):
        # CI_IMPACT: mocked nvidia-smi only; one slow query, then five valid release samples.
        def flaky(card, call):
            if call == 1:
                raise subprocess.TimeoutExpired(['nvidia-smi'], resources.RELEASE_QUERY_SECONDS)
            return [card], []
        claim, calls = self._release_with(flaky)
        self.assertTrue(claim['post_execution_verified'])
        self.assertEqual(claim['release_metadata_verification'], {'status': 'verified'})
        self.assertEqual(claim['release_utilization_percent'], [0] * 5)
        self.assertEqual(len(calls), 6)

    def test_invalid_release_metadata_is_refused_by_the_guard_before_any_timeout(self):
        # CI_IMPACT: mocked nvidia-smi only; a card whose total changed is evidence, not a timeout.
        def swapped_then_timeout(card, call):
            if call == 1:
                return [dict(card, total_mib=48000, free_mib=46116)], []
            raise subprocess.TimeoutExpired(['nvidia-smi'], resources.RELEASE_QUERY_SECONDS)
        with self.assertRaisesRegex(resources.CudaReleaseRejected, 'occupied'):
            self._release_with(swapped_then_timeout)

    def test_a_timeout_after_a_retained_invalid_sample_is_refused_not_inconclusive(self):
        # CI_IMPACT: mocked guard only. The release diagnostic, not the error type, decides: a timing
        # error with any retained sample whose exact-card metadata is invalid still refuses.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        def retained_invalid_then_timeout(_card, output, _phase, _baseline, **_kwargs):
            (output / 'model-proof-release-samples.json').write_text(json.dumps([
                {'sample_number': 1, 'devices': [card], 'processes': []},
                {'sample_number': 2, 'devices': [dict(card, total_mib=48000)], 'processes': []},
            ]), encoding='utf-8')
            raise subprocess.TimeoutExpired(['nvidia-smi'], resources.RELEASE_QUERY_SECONDS)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            os_patch, lock_patch = self._fake_windows()
            with os_patch, lock_patch, patch.object(guard, 'inventory', return_value=([card], [])), \
                    patch.object(guard, 'verify_idle', side_effect=retained_invalid_then_timeout), \
                    patch.object(guard.time, 'sleep'):
                with self.assertRaisesRegex(resources.CudaReleaseRejected, 'timed out') as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, output, {'PUBLIC': str(output)},
                                                   constraints={'required_cuda_physical_index': 1}) as claim:
                        pass
            self.assertFalse(claim['post_execution_verified'])
            self.assertEqual([sample['post_owned_exit_primary_metadata_valid']
                              for sample in caught.exception.diagnostic['samples']], [True, False])

    def test_unquiet_window_restarts_sampling_after_releasing_the_lock(self):
        # CI_IMPACT: mocked SMI snapshots; another tenant's burst refuses, then a quiet window admits.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 5}
        burst = dict(card, utilization_percent=90)
        rows = [[APPROVED, '123', 'desktop-app', 'C']]
        snapshots = [([card], rows), ([card], rows), ([burst], rows), ([card], rows)] + [([card], rows)] * 10
        locks = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            os_patch, lock_patch = self._fake_windows(locks)
            with os_patch, lock_patch, patch.object(guard, 'inventory', side_effect=snapshots) as inventory, \
                    patch.object(guard.time, 'sleep'), patch('sys.stderr') as stderr:
                with resources.execution_claim('candle-cuda', GPU_POLICY, output, {'PUBLIC': str(output)}) as claim:
                    self.assertFalse(claim['post_execution_verified'])
            self.assertEqual(inventory.call_count, len(snapshots))
            self.assertTrue(claim['post_execution_verified'])
            self.assertEqual(claim['admission_wait']['stable_sample_restarts'], 1)
            self.assertEqual(locks, [1, 2, 1, 2])
            printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn('utilization 90% (> 30%)', printed)

    def test_headroom_loss_during_locked_sampling_waits_and_restarts_with_a_logged_reason(self):
        # CI_IMPACT: mocked SMI snapshots; foreign utilization swings are diagnostic only.
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        busy = dict(card, utilization_percent=10)
        short = dict(card, used_mib=60000, free_mib=37887)
        rows = [[APPROVED, '123', 'private-process-name', 'C']]
        snapshots = ([([card], rows)] + [([busy], rows)] * 2 + [([short], rows)]
                     + [([short], rows)] + [([busy], rows)] * 6 + [([card], rows)] * 5)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            os_patch, lock_patch = self._fake_windows()
            with os_patch, lock_patch, patch.object(guard, 'inventory', side_effect=snapshots) as inventory, \
                    patch.object(guard.time, 'sleep'), patch.object(resources.time, 'sleep'), patch('sys.stderr') as stderr:
                with resources.execution_claim('candle-cuda', GPU_POLICY, output, {'PUBLIC': str(output)},
                                               constraints={'required_cuda_physical_index': 1}) as claim:
                    self.assertFalse(claim['post_execution_verified'])
            self.assertEqual(inventory.call_count, len(snapshots))
            self.assertTrue(claim['post_execution_verified'])
            self.assertEqual(claim['admission_wait']['stable_sample_restarts'], 1)
            self.assertEqual(claim['admission_wait']['headroom_shortfalls'], 1)
            self.assertEqual(claim['admission_utilization_percent'], [10] * 5)
            printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn('37887 MiB free is below the requirement', printed)

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

    def test_explicit_gpu0_claim_ignores_busy_gpu1_but_rejects_wrong_physical_index(self):
        # CI_IMPACT: synthetic driver snapshots and a fake lock only; no GPU or model work.
        gpu0 = {'uuid': OTHER, 'index': 0, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 0}
        gpu1 = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 29510,
                'free_mib': 68377, 'utilization_percent': 100}
        policy = SimpleNamespace(**{**vars(GPU_POLICY), 'cuda_device_uuid': OTHER})
        rows = [[APPROVED, '987', 'busy GPU1', 'C'], [OTHER, '123', 'inactive GPU0 context', 'C']]
        fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
        fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
        for required_index, accepted in ((0, True), (1, False)):
            with self.subTest(required_index=required_index), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner',
                               'CUDA_VISIBLE_DEVICES': OTHER}
                with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                        patch.object(guard, 'inventory', return_value=([gpu0, gpu1], rows)) as inventory, \
                        patch.object(guard.time, 'sleep'):
                    if accepted:
                        with resources.execution_claim('candle-cuda', policy, output, environment,
                                                       constraints={'required_cuda_physical_index': required_index}) as receipt:
                            self.assertEqual(receipt['physical_index'], 0)
                            self.assertEqual(receipt['device_uuid'], OTHER)
                        self.assertTrue(receipt['post_execution_verified'])
                        self.assertEqual(inventory.call_count, 11)
                    else:
                        with self.assertRaises(resources.CudaClaimRejected) as caught:
                            with resources.execution_claim('candle-cuda', policy, output, environment,
                                                           constraints={'required_cuda_physical_index': required_index}):
                                self.fail('wrong physical index must never reach a model body')
                        self.assertEqual(caught.exception.diagnostic['admission_error'],
                                         'invalid selected primary identity/metadata')
                        inventory.assert_called_once()

    def test_iris_correctness_profile_admits_two_percent_and_seals_five_samples(self):
        # CI_IMPACT: four exact registry recipes, synthetic SMI rows; no model/GPU process.
        rows = proof.load_registry()
        for fqn in ('quant_real_weights::real_tiers_shrink_the_resident_weights',
                    'quant_real_weights::real_quantized_renders_track_the_bf16_render_at_512'):
            source = next(row for row in rows if row['package'] == 'candle-gen-iris' and row['fqn'] == fqn)
            for configuration, index, selected_uuid in (('source_selected_cuda', 1, APPROVED),
                                                        ('source_selected_cuda_gpu0', 0, OTHER)):
                with self.subTest(fqn=fqn, configuration=configuration), tempfile.TemporaryDirectory() as temporary:
                    case = proof.configured_case(source, configuration)
                    selected = {'uuid': selected_uuid, 'index': index, 'total_mib': 97887,
                                'used_mib': 1884, 'free_mib': 96003, 'utilization_percent': 2}
                    other = {'uuid': OTHER if index == 1 else APPROVED, 'index': 1 - index,
                             'total_mib': 97887, 'used_mib': 29510, 'free_mib': 68377, 'utilization_percent': 100}
                    processes = [[selected_uuid, '123', 'graphics-only', 'G'],
                                 [selected_uuid, '124', 'mixed-context', 'C+G'],
                                 [other['uuid'], '987', 'busy-other-card', 'C']]
                    policy = SimpleNamespace(**{**vars(GPU_POLICY), 'cuda_device_uuid': selected_uuid})
                    output = Path(temporary)
                    environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner',
                                   'CUDA_VISIBLE_DEVICES': selected_uuid}
                    fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
                    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
                    with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                            patch.object(guard, 'inventory', return_value=([selected, other], processes)) as inventory, \
                            patch.object(guard.time, 'sleep'):
                        with resources.execution_claim('candle-cuda', policy, output, environment, constraints=case) as receipt:
                            self.assertFalse(receipt['post_execution_verified'])
                            self.assertEqual(receipt['admission_utilization_percent'], [2] * 5)
                            # The declared Iris profile is still authenticated, but admits on headroom.
                            self.assertEqual(receipt['activity_profile'], resources.SHARED_CUDA_HEADROOM_ACTIVITY)
                            self.assertFalse(receipt['isolated_benchmark_claim'])
                        self.assertTrue(receipt['post_execution_verified'])
                        self.assertEqual(receipt['release_utilization_percent'], [2] * 5)
                        self.assertEqual(receipt['physical_index'], index)
                        self.assertEqual(inventory.call_count, 11)

    def test_iris_low_activity_still_refuses_compute_unknown_and_capacity(self):
        # CI_IMPACT: pure predicate negatives; no driver query or learned work.
        profile = resources.IRIS_CUDA_CORRECTNESS_ACTIVITY
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 2}
        graphics = [[APPROVED, '123', 'private', 'G']]
        self.assertTrue(resources.proof_cuda_idle(card, graphics, GPU_POLICY, activity_profile=profile))
        self.assertFalse(resources.proof_cuda_idle(card, graphics, GPU_POLICY))
        for invalid_card, invalid_processes in (
            (dict(card, utilization_percent=5), graphics),
            (dict(card, free_mib=81919, used_mib=15968), graphics),
            (card, [[APPROVED, '123', 'private', 'C']]),
            (card, [[APPROVED, '123', 'private', 'unknown']]),
            (card, [[APPROVED, '123', 'private']]),
            (card, []),
        ):
            with self.subTest(card=invalid_card, rows=invalid_processes):
                self.assertFalse(resources.proof_cuda_idle(invalid_card, invalid_processes, GPU_POLICY,
                                                           activity_profile=profile))
        self.assertTrue(resources.proof_cuda_idle(dict(card, utilization_percent=0),
                                                  [[APPROVED, '123', 'inactive', 'C']], GPU_POLICY,
                                                  activity_profile=profile))
        baseline = {'uuid': APPROVED, 'reserved_baseline_mib': 1884,
                    'process_identities': resources.proof_cuda_contexts(APPROVED, graphics,
                                                                          activity_profile=profile)}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(guard, 'inventory', return_value=([card], [[APPROVED, '124', 'changed', 'G']])):
            with self.assertRaisesRegex(RuntimeError, 'baseline changed'):
                guard.sample_baseline(card, Path(temporary), 'MOCK', baseline,
                                      idle_check=lambda current, current_rows: resources.proof_cuda_idle(
                                          current, current_rows, GPU_POLICY, activity_profile=profile),
                                      identity_key=lambda selected_uuid, current_rows: resources.proof_cuda_contexts(
                                          selected_uuid, current_rows, activity_profile=profile),
                                      permit_bounded_growth=True)

    def test_iris_profile_forgery_and_post_failure_remain_unaccepted(self):
        # CI_IMPACT: registry forgery and mocked release rows only; no device/model execution.
        row = next(row for row in proof.load_registry() if row['package'] == 'candle-gen-iris'
                   and row['fqn'] == 'quant_real_weights::real_tiers_shrink_the_resident_weights')
        configured = proof.configured_case(row, 'source_selected_cuda')
        self.assertEqual(resources.bound_cuda_activity_profile(configured), resources.SHARED_CUDA_HEADROOM_ACTIVITY)
        for forged in ({**configured, 'source': 'crates/media/candle-gen/candle-gen-iris/tests/other.rs'},
                       {**configured, 'fqn': 'other::proof'},
                       {**configured, 'selected_configuration': {**configured['selected_configuration'],
                                                                   'cuda_activity_profile': 'benchmark-isolated'}}):
            with self.assertRaises(ValueError):
                resources.bound_cuda_activity_profile(forged)
        wrong_card = {'uuid': APPROVED, 'index': 0, 'total_mib': 97887, 'used_mib': 1884,
                      'free_mib': 96003, 'utilization_percent': 2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': APPROVED}
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
            for inventory_cards, expected in (([wrong_card], ValueError),
                                               ([dict(wrong_card, uuid=OTHER, index=1)], resources.CudaClaimRejected)):
                with self.subTest(cards=inventory_cards), patch.object(resources, 'os', fake_windows_os), \
                        patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                        patch.object(guard, 'inventory', return_value=(inventory_cards, [])) as inventory:
                    with self.assertRaises(expected):
                        with resources.execution_claim('candle-cuda', GPU_POLICY, output, environment,
                                                       constraints=configured):
                            self.fail('wrong physical card must never reach the model body')
                    inventory.assert_called_once()
        card = {'uuid': APPROVED, 'index': 1, 'total_mib': 97887, 'used_mib': 1884,
                'free_mib': 96003, 'utilization_percent': 2}
        changed = dict(card, utilization_percent=5, total_mib=48000)  # card identity/counters changed
        processes = [[APPROVED, '123', 'private', 'G']]
        snapshots = [([card], processes)] * 6 + [([changed], processes)] * 5
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': APPROVED}
            fake_windows_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
            with patch.object(resources, 'os', fake_windows_os), patch.dict(sys.modules, {'msvcrt': fake_msvcrt}), \
                    patch.object(guard, 'inventory', side_effect=snapshots), patch.object(guard.time, 'sleep'):
                with self.assertRaises(resources.CudaReleaseRejected) as caught:
                    with resources.execution_claim('candle-cuda', GPU_POLICY, output, environment,
                                                   constraints=configured) as receipt:
                        self.assertFalse(receipt['post_execution_verified'])
            self.assertFalse(receipt['post_execution_verified'])
            self.assertEqual(receipt['admission_utilization_percent'], [2] * 5)
            self.assertNotIn('release_utilization_percent', receipt)
            self.assertEqual(caught.exception.diagnostic['activity_profile'], resources.SHARED_CUDA_HEADROOM_ACTIVITY)
            self.assertEqual(caught.exception.diagnostic['samples'][0]['selected_card']['utilization_percent'], 5)


if __name__ == '__main__':
    unittest.main()
