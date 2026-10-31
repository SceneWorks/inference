"""Shared CUDA admission contracts with synthetic snapshots only; no driver/model work."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof
from scripts.ci import model_proof_resources as resources
from scripts.release import veda_cuda_guard as guard

FQN = 'train::projection::tests::cuda_bf16_packed_qkvg_preserves_outputs_and_every_adapter_gradient'
UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'
OTHER = 'GPU-12345678-1234-5678-1234-567812345678'
PROFILE = resources.SHARED_CUDA_HEADROOM_ACTIVITY
POLICY = SimpleNamespace(cuda_device_uuid=UUID, deadline_seconds=1800, poll_millis=1,
                         child_gpu_cap_bytes=8 * 1024**3, gpu_free_reserve_bytes=2 * 1024**3)


class SharedHeadroom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next(row for row in proof.load_registry() if row['fqn'] == FQN)
        cls.configured = proof.configured_case(cls.source, 'fixed_cuda')

    def card(self, utilization=6, **changes):
        return dict({'uuid': UUID, 'index': 1, 'total_mib': 97887, 'used_mib': 4996,
                     'free_mib': 92891, 'utilization_percent': utilization}, **changes)

    def claim(self, snapshots, body=None, policy=POLICY, owned_completion=True, unlock_failure=False):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': UUID}
            fake_os = SimpleNamespace(name='nt', fstat=os.fstat)
            def lock(_fd, operation, _count):
                if operation == 2 and unlock_failure:
                    raise OSError('MOCK explicit UUID unlock failure')
            fake_lock = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lock)
            with patch.object(resources, 'os', fake_os), patch.dict(sys.modules, {'msvcrt': fake_lock}), \
                    patch.object(guard, 'inventory', side_effect=snapshots) as inventory, \
                    patch.object(proof, 'load_registry', return_value=(self.source,)), \
                    patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', policy, output, environment,
                                               constraints=self.configured) as receipt:
                    self.assertFalse(receipt['post_execution_verified'])
                    if owned_completion:
                        receipt['owned_execution_completion'] = {
                            'scope': 'supervisor-owned-windows-job', 'root_pid': 1234,
                            'root_returncode': 0, 'waited_returncode': 0,
                            'job_member_pids': [], 'owned_pids': []}
                    if body is not None:
                        body(receipt)
                self.assertTrue(receipt['post_execution_verified'])
                self.assertTrue(all(call.kwargs.get('selected_uuid') == UUID for call in inventory.call_args_list))
                return receipt, inventory.call_count

    def test_shared_activity_and_changing_foreign_contexts_are_diagnostic_in_both_phases(self):
        # CI_IMPACT: synthetic claim + five admission + five release snapshots, no real processes.
        utilization = [3, 4, 6, 100, 3, 6, 99, 5, 4, 3, 100]
        snapshots = [([self.card(value)], [[UUID, str(100 + index), 'MOCK', kind]])
                     for index, (value, kind) in enumerate(zip(utilization, ['C', 'C+G', 'G', 'C', 'unknown',
                                                                             'C', 'G', 'C+G', 'C', 'G', 'C']))]
        receipt, count = self.claim(snapshots)
        self.assertEqual(count, 11)
        self.assertFalse(receipt['isolated_benchmark_claim'])
        self.assertEqual(receipt['activity_profile'], PROFILE)
        self.assertEqual(receipt['admission_utilization_percent'], utilization[1:6])
        self.assertEqual(receipt['release_utilization_percent'], utilization[6:])
        for phase in ('admission', 'release'):
            diagnostic = receipt[phase + '_background_activity']
            self.assertEqual(diagnostic['sample_count'], 5)
            self.assertTrue(all(not item['context_identities_match_admission'] for item in diagnostic['samples']))
            self.assertTrue(all(item['predicates']['proof_shared_correctness_headroom'] for item in diagnostic['samples']))
            self.assertNotIn('selected_gpu_zero_utilization', json.dumps(diagnostic))

    def test_unknown_foreign_census_is_explicit_and_cannot_masquerade_as_no_processes(self):
        card = self.card(None, process_inventory_status='unknown/not-collected-diagnostic-only')
        receipt, count = self.claim([([card], [])] * 11)
        self.assertEqual(count, 11)
        for phase in ('admission', 'release'):
            self.assertEqual(receipt[phase + '_utilization_percent'], [None] * 5)
            for sample in receipt[phase + '_background_activity']['samples']:
                self.assertEqual(sample['process_inventory_status'], 'unknown/not-collected-diagnostic-only')
                self.assertIsNone(sample['selected_process_count'])
                self.assertIsNone(sample['selected_process_type_counts'])
                self.assertIsNone(sample['context_identities_match_admission'])

    def test_bound_profile_admits_without_slow_or_unavailable_census(self):
        # CI_IMPACT: command stub only; no driver, GPU context, model, or host lease.
        def driver(args, deadline):
            if '--id=' + UUID not in args or '-x' in args or any('query-compute-apps' in arg for arg in args):
                raise TimeoutError('MOCK optional host census exhausted sampling deadline')
            return f'1, {UUID}, MOCK, 596.36, 97887, 5247, 92192, 100'
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fake_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_lock = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_: None)
            with patch.object(resources, 'os', fake_os), patch.dict(sys.modules, {'msvcrt': fake_lock}), \
                    patch.object(guard, 'command', side_effect=driver) as probe, patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', POLICY, output,
                                               {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK'},
                                               constraints=self.configured) as receipt:
                    self.assertEqual(receipt['physical_index'], 1)
                    receipt['owned_execution_completion'] = {'scope': 'supervisor-owned-windows-job', 'root_pid': 1234,
                        'root_returncode': 0, 'waited_returncode': 0, 'job_member_pids': [], 'owned_pids': []}
                self.assertTrue(receipt['post_execution_verified'])
                self.assertEqual(probe.call_count, 11)
                self.assertEqual(receipt['admission_utilization_percent'], [100] * 5)
                self.assertIsNone(receipt['admission_background_activity']['samples'][0]['selected_process_count'])

    def test_primary_timeout_preserves_refusal_details_and_never_enters_body(self):
        for error in (TimeoutError('GPU inventory sampling deadline exhausted'),
                      subprocess.TimeoutExpired('nvidia-smi', 5)):
            valid = ([self.card()], [])
            with self.subTest(error=error), self.assertRaises(resources.CudaClaimRejected) as caught:
                self.claim([valid, valid, error], body=lambda _: self.fail('primary timeout reached body'))
            self.assertIn(str(error), caught.exception.diagnostic['admission_error'])
            self.assertEqual(caught.exception.diagnostic['admission_samples']['sample_count'], 1)
            with self.assertRaises(resources.CudaReleaseRejected):
                self.claim([valid] * 7 + [error])

    def test_memory_and_physical_identity_refusals_never_enter_body(self):
        # CI_IMPACT: pure snapshots exercise initial and later phase failures, not model computation.
        invalid = [self.card(uuid=OTHER), self.card(index=0), self.card(total_mib='97887'),
                   self.card(used_mib=-1), self.card(free_mib=97888), self.card(free_mib=10239, used_mib=87648),
                   self.card(used_mib=9000), self.card(free_mib=True),
                   self.card(total_mib=None), self.card(used_mib=float('nan')),
                   self.card(free_mib=float('inf'))]
        for card in invalid:
            with self.subTest(card=card), self.assertRaises((ValueError, RuntimeError)):
                self.claim([([card], [])], body=lambda _: self.fail('invalid admission reached body'))
        with self.assertRaisesRegex(ValueError, 'fixed physical-device restriction'):
            self.claim([], policy=SimpleNamespace(**{**vars(POLICY), 'cuda_device_uuid': OTHER}))

    def test_sampling_rechecks_identity_and_headroom_after_claim_and_after_body(self):
        for invalid in (self.card(index=0), self.card(uuid=OTHER), self.card(free_mib=float('inf'))):
            with self.subTest(phase='admission', card=invalid), self.assertRaises(resources.CudaClaimRejected):
                self.claim([([self.card()], []), ([invalid], [])],
                           body=lambda _: self.fail('bad sample reached body'))
            with self.subTest(phase='release', card=invalid), self.assertRaises(resources.CudaReleaseRejected):
                self.claim([([self.card()], [])] * 6 + [([invalid], [])])

    def test_actual_moss_release_shape_passes_after_owned_exit_without_readmitting_91gib(self):
        # CI_IMPACT: immutable audit scalars copied into synthetic snapshots; no NVML/NN calls.
        policy = SimpleNamespace(**{**vars(POLICY), 'child_gpu_cap_bytes': 91 * 1024**3})
        admission = self.card(0, used_mib=2186, free_mib=95253,
                              process_inventory_status='unknown/not-collected-diagnostic-only')
        release = self.card(100, used_mib=24531, free_mib=72908,
                            process_inventory_status='unknown/not-collected-diagnostic-only')
        self.assertTrue(resources.proof_cuda_idle(admission, [], policy, activity_profile=PROFILE))
        self.assertFalse(resources.proof_cuda_idle(release, [], policy, activity_profile=PROFILE))
        receipt, count = self.claim([([admission], [])] * 6 + [([release], [])] * 5, policy=policy)
        self.assertEqual(count, 11)
        self.assertTrue(receipt['lease_released'])
        self.assertTrue(receipt['post_execution_verified'])
        diagnostics = receipt['release_background_activity']
        self.assertTrue(diagnostics['admission_headroom_is_diagnostic_only'])
        self.assertTrue(all(sample['post_owned_exit_primary_metadata_valid'] for sample in diagnostics['samples']))
        self.assertTrue(all(not sample['predicates']['free_memory_above_policy_headroom'] for sample in diagnostics['samples']))
        with self.assertRaises(resources.CudaClaimRejected):
            self.claim([([release], [])], policy=policy,
                       body=lambda _: self.fail('release snapshot admitted new work'))
        for low_free in (self.card(100, used_mib=97439, free_mib=0),
                         self.card(None, used_mib=97887, free_mib=0)):
            self.claim([([admission], [])] * 6 + [([low_free], [])] * 5, policy=policy)

    def test_shared_release_preserves_counter_rounding_and_failed_closed_scope(self):
        valid = ([self.card()], [])
        for overflow in (1, 2):
            release = self.card(100, used_mib=24531, free_mib=97887-24531+overflow)
            self.claim([valid] * 6 + [([release], [])] * 5)
        release = self.card(100, used_mib=24531, free_mib=97887-24531+3)
        with self.assertRaises(resources.CudaReleaseRejected):
            self.claim([valid] * 6 + [([release], [])])
        held = []
        for completion in ({'scope':'supervisor-no-child-created','creation_attempted':False},
                           {'scope':'supervisor-owned-windows-job','root_pid':1234,'root_returncode':1,
                            'waited_returncode':1,'job_member_pids':[],'owned_pids':[]}):
            def closed(receipt, completion=completion):
                receipt['owned_execution_completion'] = completion
                held.append(receipt)
                raise RuntimeError('MOCK unsuccessful but owned closure proven')
            with self.assertRaisesRegex(RuntimeError, 'owned closure proven'):
                self.claim([valid] * 11, body=closed)
            self.assertTrue(held[-1]['post_execution_verified'])
            self.assertTrue(held[-1]['lease_released'])

    def test_shared_release_rejects_unproven_or_pending_owned_job_before_inventory(self):
        valid = ([self.card()], [])
        with self.assertRaisesRegex(resources.CudaReleaseRejected, 'completion is unverified'):
            self.claim([valid] * 6, owned_completion=False)
        for key, value in (('root_returncode', None), ('waited_returncode', None),
                           ('root_pid', 0), ('job_member_pids', [999]), ('owned_pids', [999]),
                           ('scope', 'host-process-census')):
            def pending(receipt, key=key, value=value):
                receipt['owned_execution_completion'][key] = value
            with self.subTest(key=key), self.assertRaises(resources.CudaReleaseRejected):
                self.claim([valid] * 6, body=pending)

    def test_shared_release_metadata_deadline_and_unlock_failures_never_seal(self):
        valid = ([self.card()], [])
        for card in (self.card(uuid=OTHER), self.card(index=0), self.card(total_mib=97886),
                     self.card(used_mib=-1), self.card(used_mib=float('nan')),
                     self.card(free_mib=float('inf')), self.card(total_mib=True)):
            with self.subTest(card=card), self.assertRaises(resources.CudaReleaseRejected):
                self.claim([valid] * 6 + [([card], [])])
        held = []
        with self.assertRaisesRegex(OSError, 'unlock failure'):
            self.claim([valid] * 11, body=lambda receipt: held.append(receipt), unlock_failure=True)
        self.assertFalse(held[0]['post_execution_verified'])
        self.assertFalse(held[0]['lease_released'])
        with patch.object(resources.time, 'monotonic', return_value=100.) as clock:
            def expire(receipt):
                held.append(receipt)
                clock.return_value = 100. + POLICY.deadline_seconds
            with self.assertRaisesRegex(resources.CudaReleaseRejected, 'deadline expired'):
                self.claim([valid] * 6, body=expire)
        self.assertFalse(held[-1]['post_execution_verified'])
        self.assertTrue(held[-1]['lease_released'])

    def test_legacy_profile_with_unselected_configuration_keeps_its_original_schema(self):
        legacy = {'activity_profile': resources.STRICT_CUDA_ACTIVITY, 'post_execution_verified': True}
        self.assertIsNone(resources.validate_shared_release_record(legacy, [], {'selected_configuration': None}))
        self.assertNotIn('owned_execution_completion', legacy)
        self.assertNotIn('lease_released', legacy)

    def test_release_seal_joins_exact_supervisor_owned_completion_and_lease(self):
        record, _ = self.claim([([self.card()], [])] * 11)
        body = {'argv': ['cargo', 'test'], 'supervision': {
            'pid': 1234, 'returncode': 0, 'owned_completion': record['owned_execution_completion']}}
        resources.validate_shared_release_record(record, [body], self.configured)
        for mutation in (lambda r, c: r.update(lease_released=False),
                         lambda r, c: r.update(post_execution_verified=False),
                         lambda r, c: r.update(activity_profile=resources.STRICT_CUDA_ACTIVITY),
                         lambda r, c: r.update(device_uuid=OTHER),
                         lambda r, c: r.update(physical_index=0),
                         lambda r, c: c[0]['supervision'].update(pid=999),
                         lambda r, c: c[0]['supervision'].update(owned_completion={}),
                         lambda r, c: r['owned_execution_completion'].update(owned_pids=[999])):
            invalid, commands = copy.deepcopy(record), copy.deepcopy([body])
            mutation(invalid, commands)
            with self.assertRaises(ValueError):
                resources.validate_shared_release_record(invalid, commands, self.configured)

    def test_body_failure_propagates_even_when_shared_cleanup_succeeds(self):
        def fail(_):
            raise RuntimeError('MOCK child failure')
        with self.assertRaisesRegex(RuntimeError, 'MOCK child failure'):
            self.claim([([self.card()], [])] * 11, body=fail)

    def test_later_headroom_or_index_loss_preserves_structured_refusal_in_both_phases(self):
        # CI_IMPACT: prior successful synthetic samples must not enter strict-profile retry analysis.
        valid = ([self.card()], [])
        for sample_number in (2, 4):
            for invalid in (self.card(index=0), self.card(free_mib=float('inf'))):
                bad = ([invalid], [])
                with self.subTest(phase='admission', sample=sample_number, card=invalid):
                    with self.assertRaises(resources.CudaClaimRejected) as caught:
                        self.claim([valid] * sample_number + [bad],
                                   body=lambda _: self.fail('failed admission reached body'))
                    diagnostic = caught.exception.diagnostic['admission_samples']
                    self.assertEqual(diagnostic['status'], 'retained-existing-samples')
                    self.assertEqual(diagnostic['sample_count'], sample_number)
                    self.assertEqual(diagnostic['settle_attempt_count'], 1)
                with self.subTest(phase='release', sample=sample_number, card=invalid):
                    with self.assertRaises(resources.CudaReleaseRejected) as caught:
                        self.claim([valid] * (5 + sample_number) + [bad])
                    diagnostic = caught.exception.diagnostic
                    self.assertEqual(diagnostic['status'], 'retained-existing-samples')
                    self.assertEqual(diagnostic['sample_count'], sample_number)
                    self.assertEqual(diagnostic['settle_attempt_count'], 1)

    def test_deadline_and_driver_failure_cannot_become_shared_admission_success(self):
        probe = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'deadline expired'):
                resources.settled_cuda_samples(probe, Path(temporary), 'model-proof-admission',
                                               UUID, 1, POLICY, {'process_identities': []}, 0,
                                               activity_profile=PROFILE)
        probe.assert_not_called()
        with self.assertRaises(resources.CudaClaimRejected):
            self.claim([([self.card()], []), RuntimeError('MOCK driver failure')],
                       body=lambda _: self.fail('failed probe reached body'))

    def test_shared_profile_still_requires_complete_successful_actual_device_body(self):
        # CI_IMPACT: output protocol text only, not a successful numerical fixture claim.
        lines = ['test result: ok. 1 passed; 0 failed; 0 ignored;',
                 'MODEL_INFERENCE_BODY_COMPLETED ' + FQN,
                 'MODEL_INFERENCE_DEVICE ' + FQN + ' candle-cuda']
        self.assertEqual(proof.verify_execution(self.configured, '\n'.join(lines))['body_completed'], 1)
        invalid = [lines[:index] + lines[index + 1:] for index in range(len(lines))]
        invalid += [[lines[0].replace('1 passed', '0 passed'), *lines[1:]],
                    [*lines[:2], lines[2].replace('candle-cuda', 'candle-cpu')],
                    [*lines, 'MODEL_INFERENCE_PREREQUISITE_MISSING ' + FQN + ' MOCK missing work']]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                proof.verify_execution(self.configured, '\n'.join(output))

    def test_source_registry_binding_is_reusable_but_cannot_be_forged(self):
        self.assertEqual(resources.bound_cuda_activity_profile(self.configured), PROFILE)
        # A future source-owned case uses the same schema; no runtime allowlist code change.
        future = copy.deepcopy(self.source)
        future['fqn'] = 'train::reviewed_fixture'
        future['configurations']['fixed_cuda']['source_identity'] = future['source'] + '::train::reviewed_fixture/fixed_cuda'
        configured = proof.configured_case(future)
        with patch.object(proof, 'load_registry', return_value=(future,)):
            self.assertEqual(resources.bound_cuda_activity_profile(configured), PROFILE)
        for key, value in (('source', 'other.rs'), ('fqn', 'tests::unregistered'),
                           ('required_cuda_device_uuid', OTHER), ('required_cuda_physical_index', 0)):
            forged = {**self.configured, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.bound_cuda_activity_profile(forged)
        for key in ('required_cuda_device_uuid',):
            row = copy.deepcopy(self.source)
            row.pop(key)
            with self.assertRaises(ValueError):
                proof.validate_configurations(row)
        for value in ('unknown', resources.IRIS_CUDA_CORRECTNESS_ACTIVITY):
            row = copy.deepcopy(self.source)
            row['configurations']['fixed_cuda']['cuda_activity_profile'] = value
            with self.assertRaises(ValueError):
                proof.validate_configurations(row)

    def test_all_bounded_cases_bind_source_owned_gpu1_headroom_without_implicit_selection(self):
        rows = [row for row in proof.load_registry() if 'bounded_acceleration' in row['fqn']]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(case=proof.identity(row)):
                configured = proof.configured_case(row, 'fixed_cuda')
                self.assertEqual(configured['required_cuda_device_uuid'], UUID)
                self.assertEqual(configured['required_cuda_physical_index'], 1)
                self.assertEqual(resources.bound_cuda_activity_profile(configured), PROFILE)
                if row['default_configuration'] is None:
                    with self.assertRaisesRegex(ValueError, 'explicit configuration required'):
                        proof.configured_case(row)

    def test_request_cannot_inject_profile_and_profiles_keep_distinct_activity_rules(self):
        request = {'selection': 'candle-gen/candle_gen/' + FQN, 'policy': {'backend': 'windows-cuda'},
                   'cuda_activity_profile': PROFILE}
        with self.assertRaises(ValueError):
            proof.request(json.dumps(request), 'candle-cuda')
        card = self.card(6)
        rows = [[UUID, '123', 'MOCK', 'C']]
        self.assertTrue(resources.proof_cuda_idle(card, rows, POLICY, activity_profile=PROFILE))
        self.assertFalse(resources.proof_cuda_idle(card, rows, POLICY))
        self.assertFalse(resources.proof_cuda_idle(card, rows, POLICY,
                                                 activity_profile=resources.IRIS_CUDA_CORRECTNESS_ACTIVITY))


if __name__ == '__main__':
    unittest.main()
