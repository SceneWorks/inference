"""Shared CUDA admission contracts with synthetic snapshots only; no driver/model work."""
import copy
import json
import os
from pathlib import Path
import sys
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

    def claim(self, snapshots, body=None, policy=POLICY):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': UUID}
            fake_os = SimpleNamespace(name='nt', fstat=os.fstat)
            fake_lock = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_: None)
            with patch.object(resources, 'os', fake_os), patch.dict(sys.modules, {'msvcrt': fake_lock}), \
                    patch.object(guard, 'inventory', side_effect=snapshots) as inventory, \
                    patch.object(guard.time, 'sleep'):
                with resources.execution_claim('candle-cuda', policy, output, environment,
                                               constraints=self.configured) as receipt:
                    self.assertFalse(receipt['post_execution_verified'])
                    if body is not None:
                        body(receipt)
                self.assertTrue(receipt['post_execution_verified'])
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

    def test_memory_and_physical_identity_refusals_never_enter_body(self):
        # CI_IMPACT: pure snapshots exercise initial and later phase failures, not model computation.
        invalid = [self.card(uuid=OTHER), self.card(index=0), self.card(total_mib='97887'),
                   self.card(used_mib=-1), self.card(free_mib=97888), self.card(free_mib=10239, used_mib=87648),
                   self.card(used_mib=9000), self.card(free_mib=True)]
        for card in invalid:
            with self.subTest(card=card), self.assertRaises((ValueError, RuntimeError)):
                self.claim([([card], [])], body=lambda _: self.fail('invalid admission reached body'))
        with self.assertRaisesRegex(ValueError, 'fixed physical-device restriction'):
            self.claim([], policy=SimpleNamespace(**{**vars(POLICY), 'cuda_device_uuid': OTHER}))

    def test_sampling_rechecks_identity_and_headroom_after_claim_and_after_body(self):
        for invalid in (self.card(index=0), self.card(uuid=OTHER), self.card(free_mib=10239, used_mib=87648)):
            with self.subTest(phase='admission', card=invalid), self.assertRaises(resources.CudaClaimRejected):
                self.claim([([self.card()], []), ([invalid], [])],
                           body=lambda _: self.fail('bad sample reached body'))
            with self.subTest(phase='release', card=invalid), self.assertRaises(resources.CudaReleaseRejected):
                self.claim([([self.card()], [])] * 6 + [([invalid], [])])

    def test_body_failure_propagates_even_when_shared_cleanup_succeeds(self):
        def fail(_):
            raise RuntimeError('MOCK child failure')
        with self.assertRaisesRegex(RuntimeError, 'MOCK child failure'):
            self.claim([([self.card()], [])] * 11, body=fail)

    def test_later_headroom_or_index_loss_preserves_structured_refusal_in_both_phases(self):
        # CI_IMPACT: prior successful synthetic samples must not enter strict-profile retry analysis.
        valid = ([self.card()], [])
        for sample_number in (2, 4):
            for invalid in (self.card(index=0), self.card(free_mib=10239, used_mib=87648)):
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
