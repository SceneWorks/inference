"""Two-approved-UUID runner composition with frozen bounded-supervisor ABI; no hardware/NN."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.ci import model_inference_proofs as proof
from scripts.tests.test_model_proof_multi_cuda import ROLE0, ROLE1, row, request


class MultiCudaRunner(unittest.TestCase):
    def collect(self, root, *, post_failure=False, secondary_cap_failure=False):
        case = row()
        case.update(fixture_identity='MOCK source-defined protocol fixture; NO actual model', configuration_source='MOCK source fixture',
                    output_contract='MOCK logs', execution_role='direct', expected_devices=['candle-cuda'],
                    input_contract={'fixture_kind': 'source-defined', 'source_identity': 'MOCK fixture', 'requirements': []})
        source = root / case['source']; source.parent.mkdir(parents=True); source.write_text('// MOCK not Rust execution', encoding="utf-8")
        value = request()
        observed = []
        records = []
        @contextmanager
        def physical(actual_row, policies, actual_request, output, environment):
            self.assertEqual(tuple(policy.cuda_device_uuid for policy in policies), (ROLE0, ROLE1))
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], ROLE0 + ',' + ROLE1)
            record = {'approved_uuid_roles': [ROLE0, ROLE1], 'lock_order': [ROLE1, ROLE0],
                'actual_ordinal_binding': {'mapping': [{'cuda_ordinal': 0, 'uuid': ROLE0}, {'cuda_ordinal': 1, 'uuid': ROLE1}],
                    'probe': {'schema_version': 1, 'context_created': False,
                        'abi_source': {'crate': 'cudarc', 'version': '0.19.7', 'checksum': '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'},
                        'visibility': {'CUDA_VISIBLE_DEVICES': ROLE0 + ',' + ROLE1, 'CUDA_DEVICE_ORDER': environment.get('CUDA_DEVICE_ORDER')}}},
                'post_execution_verified': False, 'post_idle_devices': []}
            records.append(record)
            try:
                yield record
            finally:
                record['post_idle_devices'] = [{'uuid': ROLE0, 'verified': True}, {'uuid': ROLE1, 'verified': not post_failure}]
                record['post_execution_verified'] = not post_failure
                if post_failure:
                    raise ValueError('MOCK secondary post-idle failed')
        def guarded(argv, **kwargs):
            discovery = '--list' in argv
            self.assertEqual(kwargs['reap_metadata_descendants'], discovery)
            observed.append(kwargs)
            text = (case['fqn'] + ': test\n\n1 test, 0 benchmarks\n') if discovery else (
                'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED ' + case['fqn'] + '\n'
                'MODEL_INFERENCE_DEVICE ' + case['fqn'] + ' candle-cuda\n')
            kwargs['stdout_path'].write_text(text, encoding="utf-8"); kwargs['stderr_path'].write_text('', encoding="utf-8"); kwargs['event_path'].write_text('{"MOCK":true}\n', encoding="utf-8")
            if discovery:
                self.assertNotIn('additional_cuda_policies', kwargs)
                return safety.RunResult(1, 0, 100, None, 1000000, None, 0.01, (), {'MOCK': True})
            policies = (kwargs['policy'], *kwargs['additional_cuda_policies'])
            probes = (kwargs['probe'], *kwargs['additional_cuda_probes'])
            self.assertEqual(tuple(policy.cuda_device_uuid for policy in policies), (ROLE0, ROLE1))
            self.assertEqual(tuple(probe.policy.cuda_device_uuid for probe in probes), (ROLE0, ROLE1))
            devices = tuple({'role': role, 'cudaDeviceUuid': policy.cuda_device_uuid, 'policySha256': policy.sha256,
                             'admission': {'MOCK': 'bounded cap-fallback estimate'}, 'samples': [],
                             'peakGpuBytes': None, 'peakGpuUnknownReason': 'MOCK process completed before attributed positive sample'}
                            for role, policy in enumerate(policies))
            if secondary_cap_failure:
                error = safety.SupervisionError('gpu-memory', 'MOCK role1 cap violation; owned tree drained')
                error.cuda_devices = devices
                raise error
            return safety.MultiCudaRunResult(1, 0, 100, None, 1000000, 1000000, 0.01, (), {'MOCK': True}, devices)
        environment = {'RUNNER_TEMP': str(root), 'RUNNER_OS': 'Windows', 'RUNNER_NAME': 'MOCK CUDA runner',
                       'GITHUB_EVENT_NAME': 'workflow_dispatch', 'MODEL_PROOF_RUNNER_ENVIRONMENT': 'self-hosted'}
        output = root / 'evidence'
        with patch.dict(os.environ, environment), patch.object(proof, 'ROOT', root), \
                patch.object(proof, 'repository_identity', return_value={'revision': '1' * 40, 'tree': '2' * 40}), \
                patch.object(proof, 'verified_toolchain', return_value={'MOCK': 'no compiler'}), \
                patch.object(proof, 'prepare_build', return_value={'MOCK': 'no build'}), \
                patch.object(proof.multi_cuda, 'execution_claim', side_effect=physical), \
                patch.object(safety, 'SystemProbe', side_effect=lambda policy: SimpleNamespace(policy=policy)), \
                patch.object(safety, 'run_guarded', side_effect=guarded):
            status = proof.run((case,), 'candle-cuda', value, '1' * 40, output)
        receipt = json.loads((output / 'receipt.json').read_text(encoding="utf-8"))
        verified = proof.verify_artifacts(output, root, value, '1' * 40, (case,))
        return status, receipt, verified, records, output, case, value

    def test_all_policies_and_explicit_matching_probes_reach_bounded_monitor_with_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, records, output, case, value = self.collect(Path(temporary).resolve())
            self.assertEqual(status, 0)
            self.assertTrue(receipt['accepted'])
            self.assertEqual(receipt['approved_cuda_uuid_roles'], [ROLE0, ROLE1])
            self.assertEqual([device['cudaDeviceUuid'] for device in receipt['cases'][0]['cuda_devices']], [ROLE0, ROLE1])
            self.assertTrue(all(device['peakGpuBytes'] is None for device in receipt['cases'][0]['cuda_devices']))
            self.assertTrue(records[0]['post_execution_verified'])
            self.assertFalse(verified['terminal_model_evidence'])
            self.assertFalse(verified['checkpoint_byte_integrity_attested'])
            receipt['cases'][0]['physical_execution']['post_idle_devices'][1]['verified'] = False
            (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
            (output / '00000/result.json').write_text(json.dumps(receipt['cases'][0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'every approved ordered role/lease/post-idle'):
                proof.verify_artifacts(output, Path(temporary).resolve(), value, '1' * 40, (case,))

    def test_missing_second_monitor_or_policy_tamper_cannot_be_uploaded_as_two_gpu_proof(self):
        for tamper in ('missing-monitor', 'wrong-policy', 'driver-context'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                _, receipt, _, _, output, case, value = self.collect(root)
                result = receipt['cases'][0]
                if tamper == 'missing-monitor':
                    result['cuda_devices'].pop()
                elif tamper == 'wrong-policy':
                    result['cuda_devices'][1]['policySha256'] = '0' * 64
                else:
                    result['physical_execution']['actual_ordinal_binding']['probe']['context_created'] = True
                (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
                (output / '00000/result.json').write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(ValueError):
                    proof.verify_artifacts(output, root, value, '1' * 40, (case,))

    def test_secondary_cap_or_postidle_failure_never_seals_accepted_even_when_body_logs_look_green(self):
        for cap_failure in (False, True):
            with self.subTest(cap_failure=cap_failure), tempfile.TemporaryDirectory() as temporary:
                status, receipt, verified, records, _, _, _ = self.collect(Path(temporary).resolve(),
                    post_failure=not cap_failure, secondary_cap_failure=cap_failure)
                self.assertNotEqual(status, 0)
                self.assertFalse(receipt['accepted'])
                self.assertFalse(verified['accepted_case_execution'])
                self.assertFalse(receipt['cases'][0]['accepted'])
                if cap_failure:
                    self.assertEqual(len(receipt['cases'][0]['cuda_devices']), 2)
                    self.assertEqual(receipt['cases'][0]['supervision_failure']['reason'], 'gpu-memory')
                else:
                    self.assertTrue(receipt['cases'][0]['body_proof_accepted'])
                    self.assertTrue(receipt['cases'][0]['resource_verification_pending'])


if __name__ == '__main__':
    unittest.main()
