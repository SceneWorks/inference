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

    def test_zero_and_positive_recovery_counts_are_reported_honestly(self):
        self.exercise(recovery_count=0, recovery=False)
        self.exercise(recovery_count=2, recovery=True)

    def test_missing_malformed_or_inconsistent_recovery_observations_cannot_pass(self):
        for count in (None, -1, True, False, 0.5, "0"):
            with self.subTest(count=count):
                self.exercise(recovery_count=count, error=ValueError)
        for count, exercised in ((0, True), (1, False), (0, None), (0, 0), (1, 1)):
            with self.subTest(count=count, exercised=exercised):
                self.exercise(recovery_count=count, recovery=exercised, error=ValueError)

    def test_missing_or_out_of_cap_high_marks_cannot_pass(self):
        for field, value in (("heldBytes", None), ("heldBytes", 1),
                             ("usedHighBytes", None), ("usedHighBytes", 536870913),
                             ("reservedHighBytes", 536870913), ("reservedHighBytes", True),
                             ("usedHighBytes", 536870911.0)):
            with self.subTest(field=field, value=value):
                self.exercise(evidence_override={field: value}, error=ValueError)

    def test_invalid_binding_refuses_before_any_claim_or_launch(self):
        for key, value in [('CUDA_VISIBLE_DEVICES', '0'), ('SCENEWORKS_GPU_ID', '0'),
                           ('SCENEWORKS_CUDA_PROOF_POOL_BYTES', '1')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                env = environment(); env[key] = value
                def unexpected(*args, **kwargs):
                    self.fail('invalid binding reached physical claim or child')
                with self.assertRaises(ValueError):
                    proof.run(Path(temporary), Path('MOCK.exe'), env, claim=unexpected, supervise=unexpected)

    def exercise(self, *, returncode=0, fault=None, error=None, verified=True, valid_evidence=True, recovery_count=0, recovery=False, evidence_override=None):
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
                            'heldBytes': 469762048, 'usedHighBytes': 536870912,
                            'reservedHighBytes': 536870912}
                if recovery_count is not None:
                    evidence['successfulRecoveryCount'] = recovery_count
                if recovery is not None:
                    evidence['fragmentationRecoveryExercised'] = recovery
                evidence.update(evidence_override or {})
                kwargs['stdout_path'].write_text('test result: ok. 1 passed; 0 failed; 0 ignored\n'
                                                + 'CUDA_PROOF_POOL_EVIDENCE ' + json.dumps(evidence) + '\n', encoding='utf-8')
                return RunResult(123, returncode, 1, None, 2, None, 0.1, (), {'MOCK': True})
            ticks = iter([100, 110])
            if error:
                with self.assertRaises(error):
                    proof.run(output, Path('MOCK.exe'), environment(), claim=claim, supervise=supervise,
                              clock=lambda: next(ticks))
            else:
                result = proof.run(output, Path('MOCK.exe'), environment(), claim=claim,
                                   supervise=supervise, clock=lambda: next(ticks))
                self.assertTrue(result['accepted'])
                self.assertEqual(result['allocator_evidence']['successfulRecoveryCount'], recovery_count)
                self.assertIs(result['allocator_evidence']['fragmentationRecoveryExercised'], recovery)
            self.assertEqual(states, ['locked', 'released'])
            receipt = json.loads((output / 'allocator-execution.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['accepted'], error is None)
            if error:
                self.assertIn('error', receipt)


if __name__ == '__main__':
    unittest.main()
