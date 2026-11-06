"""CPU stubs only: exact allocator launch binding, lease lifetime and fault propagation."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ci import cuda_memory_envelope_proof as proof
from scripts.media_campaign_supervisor import RunResult

UUID = 'GPU-12345678-1234-5678-1234-567812345678'


def environment():
    return {'CUDA_VISIBLE_DEVICES': UUID, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'SCENEWORKS_GPU_ID': '1', 'SCENEWORKS_CUDA_PROOF_POOL_BYTES': '536870912'}


class AllocatorLauncherTests(unittest.TestCase):
    def test_exact_compiled_target_is_selected_and_ambiguous_or_other_targets_refuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'MOCK-integration.exe'
            executable.write_bytes(b'NOT AN EXECUTABLE')
            artifact = {'reason': 'compiler-artifact', 'executable': str(executable),
                        'target': {'name': 'integration', 'kind': ['test'],
                                   'src_path': str(proof.ROOT / 'crates/media/candle-gen/candle-gen/tests/main.rs')},
                        'profile': {'test': True}, 'features': ['cuda', 'testkit']}
            manifest = root / 'artifacts.jsonl'
            manifest.write_text(json.dumps(artifact) + '\n', encoding='utf-8')
            self.assertEqual(proof.executable_from_artifacts(manifest), executable.resolve())
            manifest.write_text((json.dumps(artifact) + '\n') * 2, encoding='utf-8')
            with self.assertRaises(ValueError):
                proof.executable_from_artifacts(manifest)
            artifact['target']['src_path'] = str(root / 'other-tests.rs')
            manifest.write_text(json.dumps(artifact) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                proof.executable_from_artifacts(manifest)

    def test_child_and_positive_evidence_are_inside_lease_and_release_is_required(self):
        self.exercise()

    def test_nonzero_child_and_supervisor_cancellation_propagate_after_release(self):
        self.exercise(returncode=19, error=RuntimeError)
        self.exercise(fault=KeyboardInterrupt('MOCK cancellation'), error=KeyboardInterrupt)

    def test_unverified_release_or_missing_enforcement_cannot_pass(self):
        self.exercise(verified=False, error=ValueError)
        self.exercise(valid_evidence=False, error=ValueError)

    def test_missing_or_false_recovery_witness_cannot_pass(self):
        self.exercise(recovery=None, error=ValueError)
        self.exercise(recovery=False, error=ValueError)

    def test_invalid_binding_refuses_before_any_claim_or_launch(self):
        for key, value in [('CUDA_VISIBLE_DEVICES', '0'), ('SCENEWORKS_GPU_ID', '0'),
                           ('SCENEWORKS_CUDA_PROOF_POOL_BYTES', '1')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                env = environment(); env[key] = value
                def unexpected(*args, **kwargs):
                    self.fail('invalid binding reached physical claim or child')
                with self.assertRaises(ValueError):
                    proof.run(Path(temporary), Path('MOCK.exe'), env, claim=unexpected, supervise=unexpected)

    def exercise(self, *, returncode=0, fault=None, error=None, verified=True, valid_evidence=True, recovery=True):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            states = []
            @contextmanager
            def claim(backend, policy, directory, env, **kwargs):
                self.assertEqual(backend, 'candle-cuda')
                self.assertEqual(kwargs['constraints'], {'required_cuda_physical_index': 1,
                                                        'required_cuda_device_uuid': UUID})
                self.assertEqual(kwargs['original_deadline_monotonic'], 1900)
                states.append('locked')
                record = {'post_execution_verified': False, 'MOCK': True}
                try:
                    yield record
                finally:
                    states.append('released')
                    record['post_execution_verified'] = verified
            def supervise(argv, **kwargs):
                self.assertEqual(states, ['locked'])
                self.assertEqual(argv[1:], [proof.FQN, '--exact', '--ignored', '--nocapture', '--test-threads=1'])
                self.assertEqual(kwargs['policy'].cuda_device_uuid, UUID)
                self.assertEqual(kwargs['policy'].deadline_seconds, 1790)
                if fault:
                    raise fault
                evidence = {'overCapRefused': valid_evidence, 'bothConstructors': True,
                            'cublasStreamResetRebound': True, 'capBytes': 536870912,
                            'reservedHighBytes': 536870912}
                if recovery is not None:
                    evidence['fragmentationRecoveryExercised'] = recovery
                kwargs['stdout_path'].write_text('test result: ok. 1 passed; 0 failed; 0 ignored\n'
                                                + 'CUDA_PROOF_POOL_EVIDENCE ' + json.dumps(evidence) + '\n', encoding='utf-8')
                return RunResult(123, returncode, 1, None, 2, None, 0.1, (), {'MOCK': True})
            ticks = iter([100, 110])
            if error:
                with self.assertRaises(error):
                    proof.run(output, Path('MOCK.exe'), environment(), claim=claim, supervise=supervise,
                              clock=lambda: next(ticks))
            else:
                self.assertTrue(proof.run(output, Path('MOCK.exe'), environment(), claim=claim,
                                          supervise=supervise, clock=lambda: next(ticks))['accepted'])
            self.assertEqual(states, ['locked', 'released'])
            receipt = json.loads((output / 'allocator-execution.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['accepted'], error is None)
            if error:
                self.assertIn('error', receipt)


if __name__ == '__main__':
    unittest.main()
