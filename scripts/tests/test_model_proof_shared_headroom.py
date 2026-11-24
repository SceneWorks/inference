"""Shared CUDA admission contracts with synthetic snapshots only; no driver/model work."""
import copy
import contextlib
import dataclasses
import io
import itertools
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
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


class FakeClock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class SharedHeadroom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = next(row for row in proof.load_registry() if row['fqn'] == FQN)
        cls.configured = proof.configured_case(cls.source, 'fixed_cuda')

    def card(self, utilization=6, **changes):
        return dict({'uuid': UUID, 'index': 1, 'total_mib': 97887, 'used_mib': 4996,
                     'free_mib': 92891, 'utilization_percent': utilization}, **changes)

    def claim(self, snapshots, body=None, policy=POLICY, owned_completion=True, unlock_failure=False, clock=None,
              original_deadline_monotonic=None, require_owned_completion=True, lock_trace=None):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            environment = {'PUBLIC': str(output), 'RUNNER_NAME': 'MOCK runner', 'CUDA_VISIBLE_DEVICES': UUID}
            fake_os = SimpleNamespace(name='nt', fstat=os.fstat)
            def lock(_fd, operation, _count):
                if lock_trace is not None:
                    lock_trace.append(operation)
                if operation == 2 and unlock_failure:
                    raise OSError('MOCK explicit UUID unlock failure')
            fake_lock = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lock)
            with patch.object(resources, 'os', fake_os), patch.dict(sys.modules, {'msvcrt': fake_lock}), \
                    patch.object(guard, 'inventory', side_effect=snapshots) as inventory, \
                    patch.object(proof, 'load_registry', return_value=(self.source,)), \
                    (patch.object(resources.time, 'monotonic', side_effect=clock.monotonic) if clock else contextlib.nullcontext()), \
                    patch.object(guard.time, 'sleep', side_effect=clock.sleep if clock else None):
                with resources.execution_claim('candle-cuda', policy, output, environment,
                                               constraints=self.configured,
                                               original_deadline_monotonic=original_deadline_monotonic,
                                               require_owned_completion=require_owned_completion) as receipt:
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

    def test_a_card_another_process_holds_40_gb_of_admits_a_fitting_estimate_and_refuses_one_that_does_not(self):
        # SCAIL's guard: 6 GiB reserve, 84 GiB cap. Another process holds 38,147 MiB (40 GB),
        # leaving 59,740 MiB free on the 97,887 MiB card.
        shared = self.card(4, used_mib=38147, free_mib=59740)
        base = SimpleNamespace(**{**vars(POLICY), 'child_gpu_cap_bytes': 84 * 1024**3,
                                  'gpu_free_reserve_bytes': 6 * 1024**3, 'deadline_seconds': 20})
        fits = SimpleNamespace(**{**vars(base), 'claim_gpu_estimate': ('test-stated-estimate', 40 * 1024**3)})
        too_big = SimpleNamespace(**{**vars(base), 'claim_gpu_estimate': ('test-stated-estimate', 60 * 1024**3)})
        # No stated estimate: the job may fill its 84 GiB cap, so it needs cap plus reserve free.
        self.assertEqual(resources.claim_required_free_bytes(base), 90 * 1024**3)
        self.assertFalse(resources.proof_cuda_idle(shared, [], base, activity_profile=PROFILE))
        for policy in (fits,):
            self.assertTrue(resources.proof_cuda_idle(shared, [], policy, activity_profile=PROFILE))
            receipt, _ = self.claim([([shared], [])] * 11, policy=policy, clock=FakeClock())
            self.assertEqual(receipt['claim_gpu_estimate']['required_free_bytes'],
                             resources.claim_required_free_bytes(policy))
            self.assertEqual(receipt['admission_wait']['headroom_shortfalls'], 0)
        self.assertFalse(resources.proof_cuda_idle(shared, [], too_big, activity_profile=PROFILE))
        short = SimpleNamespace(**{**vars(too_big), 'deadline_seconds': .003})
        stderr = io.StringIO()
        with self.assertRaises(resources.CudaClaimRejected) as caught, contextlib.redirect_stderr(stderr):
            self.claim(itertools.repeat(([shared], [])), policy=short, clock=FakeClock(),
                       body=lambda _: self.fail('a card without estimate plus reserve free reached the body'))
        reason = caught.exception.diagnostic['admission_wait']['last_refusal']
        self.assertIn('59740 MiB free < 67584 MiB required', reason)
        self.assertIn('test-stated-estimate', reason)
        self.assertIn('59740 MiB free', stderr.getvalue())

    def test_a_claim_record_replays_under_its_recorded_estimate_or_the_legacy_cap(self):
        from scripts.media_campaign_supervisor import load_policy
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'policy.json'
            path.write_text(json.dumps({
                'schemaVersion': 1, 'backend': 'windows-cuda', 'deadlineSeconds': 2400, 'pollMillis': 250,
                'termGraceMillis': 2000, 'stdoutCapBytes': 4096, 'stderrCapBytes': 4096, 'eventCapBytes': 4096,
                'childFootprintCapBytes': 64 * 1024**3, 'hostFreeReserveBytes': 8 * 1024**3,
                'cudaDeviceUuid': UUID, 'childGpuCapBytes': 84 * 1024**3, 'gpuFreeReserveBytes': 6 * 1024**3}),
                encoding='utf-8')
            loaded = load_policy(path)
        legacy = resources.replay_claim_policy(loaded, {})
        self.assertEqual(resources.claim_required_free_bytes(legacy), 90 * 1024**3)
        stated = resources.claim_estimate_record(dataclasses.replace(
            loaded, claim_gpu_estimate=('scail2-plural-reviewed-device-peak-v1', 60 * 1024**3)))
        replay = resources.replay_claim_policy(loaded, {'claim_gpu_estimate': stated})
        self.assertEqual(resources.claim_required_free_bytes(replay), 66 * 1024**3)
        self.assertEqual(replay.sha256, loaded.sha256)
        for bad in (dict(stated, required_free_bytes=1), dict(stated, estimate_bytes=85 * 1024**3),
                    dict(stated, reserve_bytes=1), {'source': 'x'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resources.replay_claim_policy(loaded, {'claim_gpu_estimate': bad})

    def test_comparison_absolute_deadline_survives_slow_discovery_before_claim(self):
        # Metadata-only clock: global100+1000 budget, discovery ended402;
        # body claim must not refresh to402+998 despite its entry policy.
        clock=FakeClock();clock.now=402.
        policy=SimpleNamespace(**{**vars(POLICY),'deadline_seconds':998.})
        receipt,_=self.claim([([self.card()],[])]*11,policy=policy,clock=clock,
                            original_deadline_monotonic=1100.)
        self.assertEqual(receipt['claim_deadline_monotonic'],1100.)
        self.assertTrue(receipt['lease_released'])

    def test_shared_headroom_waits_before_body_and_restarts_five_samples_without_other_veto(self):
        # This job states a 91 GiB device estimate (its cap), so the card needs 93 GiB free.
        policy = SimpleNamespace(**{**vars(POLICY),'child_gpu_cap_bytes':91*1024**3,'deadline_seconds':20,
                                    'claim_gpu_estimate':('test-stated-estimate',91*1024**3)})
        busy = self.card(10, used_mib=20875, free_mib=76564)
        ready = self.card(10, used_mib=2186, free_mib=95253)
        observations = [([busy], [])]*2 + [([ready], [])]*11
        seen=[]
        receipt, count = self.claim(observations, policy=policy, clock=FakeClock(), body=lambda r:seen.append(r))
        self.assertEqual(count, 13)
        self.assertEqual(receipt['admission_wait']['headroom_shortfalls'], 2)
        self.assertEqual(receipt['admission_wait']['headroom_wait_seconds'], .002)
        self.assertEqual(len(seen), 1)
        # Headroom can disappear during admission sampling: discard that partial pass,
        # await readiness and still require a complete fresh five-sample pass.
        snapshots = [([ready], []),([ready], []),([busy], []),([busy], []),([ready], [])]+[([ready], [])]*10
        receipt, _ = self.claim(snapshots, policy=policy, clock=FakeClock())
        self.assertEqual(receipt['admission_wait']['stable_sample_restarts'], 1)
        self.assertEqual(receipt['admission_background_activity']['sample_count'], 5)
        self.assertEqual(receipt['claim_deadline_monotonic'], 20.)

    def test_timed_out_window_releases_lock_then_admits_only_five_new_samples(self):
        # CI_IMPACT: scripted selected-card reads, an in-memory clock and a fake Windows UUID lock.
        partial = self.card(25, used_mib=7500, free_mib=90387)
        fresh = self.card(1)
        for failure, completed in ((TimeoutError('GPU inventory sampling deadline exhausted'), 2),
                                   (subprocess.TimeoutExpired(['nvidia-smi', '--id=' + UUID], .1), 0)):
            locks, attempts, persisted = [], [], []
            def read(output, label, deadline, **_kwargs):
                attempts.append(label)
                if label == 'model-proof-admission-sample-' + str(completed + 1) and len(attempts) == completed + 2:
                    raise failure
                if label == 'model-proof-claimed-device' and len(attempts) > 1:
                    persisted.extend(json.loads((output / 'model-proof-admission-samples.json').read_text(encoding="utf-8")))
                return ([partial if len(attempts) <= completed + 1 else fresh], [])
            with self.subTest(completed=completed):
                receipt, _ = self.claim(read, clock=FakeClock(), original_deadline_monotonic=20, lock_trace=locks)
                self.assertEqual(locks, [1, 2, 1, 2])
                self.assertEqual([row['selected_used_mib'] for row in persisted], [7500] * completed)
                self.assertEqual(attempts[completed + 2:completed + 8],
                                 ['model-proof-claimed-device', *[f'model-proof-admission-sample-{n}' for n in range(1, 6)]])
                self.assertEqual(receipt['claim_deadline_monotonic'], 20)
                self.assertEqual(receipt['admission_baseline']['reserved_baseline_mib'], fresh['used_mib'])
                self.assertEqual(receipt['admission_utilization_percent'], [1] * 5)
                self.assertEqual(receipt['admission_wait']['inventory_window_timeouts'], 1)
                self.assertEqual(receipt['admission_wait']['last_inventory_window_timeout'],
                                 {'error': str(failure), 'discarded_samples': completed})

    def test_window_timeout_cannot_extend_claim_deadline_or_suppress_identity_and_query_errors(self):
        ready, locks, clock = ([self.card()], []), [], FakeClock()
        def expire(output, label, _deadline, **_kwargs):
            if label == 'model-proof-admission-sample-3':
                clock.now = 3
                raise TimeoutError('GPU inventory sampling deadline exhausted')
            return ready
        with self.assertRaisesRegex(resources.CudaClaimRejected, 'original CUDA claim deadline expired') as caught:
            self.claim(expire, clock=clock, original_deadline_monotonic=3, lock_trace=locks,
                       body=lambda _: self.fail('exhausted admission entered the body'))
        self.assertEqual(locks, [1, 2])
        self.assertEqual(caught.exception.diagnostic['admission_samples']['sample_count'], 2)
        for bad in ([([self.card(index=0)], [])], [([self.card(uuid=OTHER)], [])],
                    [([self.card(total_mib=98000)], [])], [TimeoutError('unrelated I/O')],
                    [RuntimeError(resources.release_window_overrun('model-proof-admission'))]):
            locks = []
            with self.subTest(bad=bad), self.assertRaises(resources.CudaClaimRejected) as caught:
                self.claim([ready, *bad], clock=FakeClock(), lock_trace=locks,
                           body=lambda _: self.fail('invalid sample reached the body'))
            self.assertEqual(locks, [1, 2])
            self.assertNotIn('inventory_window_timeouts', caught.exception.diagnostic['admission_wait'])
        locks = []
        with self.assertRaises(resources.CudaClaimRejected) as caught:
            self.claim([ready, TimeoutError('GPU inventory sampling deadline exhausted'), ([self.card(index=0)], [])],
                       clock=FakeClock(), lock_trace=locks, body=lambda _: self.fail('reacquired wrong card'))
        self.assertEqual(locks, [1, 2, 1, 2])
        self.assertEqual(caught.exception.diagnostic['admission_error'], 'invalid selected primary identity/metadata')

    def test_malformed_selected_utilization_refuses_initial_and_fresh_windows_after_timeout(self):
        ready = ([self.card()], [])
        timeout = TimeoutError('GPU inventory sampling deadline exhausted')
        for utilization in (-5, 101, False, 4.0, '5'):
            malformed = ([self.card(utilization)], [])
            for initial, before, expected_locks in (
                (True, [], [1, 2]), (False, [ready, timeout], [1, 2, 1, 2]),
                (False, [ready, timeout, ready], [1, 2, 1, 2]),
            ):
                locks = []
                with self.subTest(utilization=utilization, reads=len(before)), self.assertRaises(resources.CudaClaimRejected) as caught:
                    self.claim([*before, *[malformed] * 5], clock=FakeClock(), lock_trace=locks,
                               body=lambda _: self.fail('malformed utilization entered the body'))
                diagnostic = caught.exception.diagnostic
                self.assertIn('invalid selected primary identity/metadata', diagnostic['admission_error'])
                self.assertEqual(locks, expected_locks)
                self.assertEqual(diagnostic['admission_wait']['stable_sample_restarts'], 0 if initial else 1)
        unknown = ([self.card(None)], [])
        receipt, _ = self.claim([ready, timeout, *[unknown] * 11], clock=FakeClock())
        self.assertEqual(receipt['admission_utilization_percent'], [None] * 5)

    def test_shared_wait_timeout_invalid_metadata_and_query_failures_never_enter_body(self):
        busy = self.card(100, used_mib=87648, free_mib=10239)
        policy = SimpleNamespace(**{**vars(POLICY),'deadline_seconds':.003})
        with self.assertRaises(resources.CudaClaimRejected) as caught:
            self.claim(itertools.repeat(([busy], [])), policy=policy, clock=FakeClock(),
                       body=lambda _:self.fail('deadline wait reached body'))
        self.assertIn('original CUDA claim deadline expired', caught.exception.diagnostic['admission_error'])
        self.assertGreaterEqual(caught.exception.diagnostic['admission_wait']['headroom_shortfalls'], 1)
        for invalid in (self.card(uuid=OTHER), self.card(index=0), self.card(used_mib=float('nan'))):
            with self.subTest(invalid=invalid),self.assertRaises(resources.CudaClaimRejected):
                self.claim([([busy], []),([invalid], [])],clock=FakeClock())
        # A non-timeout query failure still refuses at once; timeouts are retried (see below).
        with self.assertRaises(resources.CudaClaimRejected):
            self.claim([([busy], []),RuntimeError('MOCK primary query failed')],clock=FakeClock())

    def test_changing_foreign_contexts_are_diagnostic_and_release_activity_is_not_judged(self):
        # CI_IMPACT: synthetic claim + five admission + five release snapshots, no real processes.
        # Admission needs a quiet window (<= 30%, spread <= 15); release only checks identity/counters.
        utilization = [3, 4, 6, 18, 3, 6, 99, 5, 4, 3, 100]
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
            return f'1, {UUID}, MOCK, 596.36, 97887, 5247, 92192, 10'
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
                self.assertEqual(receipt['admission_utilization_percent'], [10] * 5)
                self.assertIsNone(receipt['admission_background_activity']['samples'][0]['selected_process_count'])

    def test_primary_timeout_preserves_refusal_details_and_never_enters_body(self):
        policy = SimpleNamespace(**{**vars(POLICY), 'deadline_seconds': .005})
        for error in (TimeoutError('GPU inventory sampling deadline exhausted'),
                      subprocess.TimeoutExpired('nvidia-smi', 5)):
            valid = ([self.card()], [])
            # A query that keeps timing out is retried until the admission deadline, then refused.
            with self.subTest(error=error), self.assertRaises(resources.CudaClaimRejected) as caught, \
                    patch('sys.stderr') as stderr:
                self.claim(itertools.repeat(error), policy=policy, clock=FakeClock(),
                           body=lambda _: self.fail('primary timeout reached body'))
            wait = caught.exception.diagnostic['admission_wait']
            self.assertIn('original CUDA claim deadline expired', caught.exception.diagnostic['admission_error'])
            self.assertEqual(wait['primary_polls'], 0)
            self.assertGreaterEqual(wait['inventory_timeouts'], 2)
            self.assertIn(str(error), wait['last_inventory_timeout'])
            self.assertEqual(wait['last_refusal'], 'nvidia-smi inventory query timed out')
            printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn('CUDA admission waiting: ' + UUID + ': nvidia-smi inventory query timed out; retrying', printed)
            # Post-run: a slow driver query is retried, then recorded as inconclusive; the
            # owned-Job exit still seals the passing proof (run 44431476698).
            receipt, _ = self.claim([valid] * 7 + [error] * resources.RELEASE_QUERY_ATTEMPTS)
            self.assertEqual(receipt['release_metadata_verification']['status'],
                             'inconclusive-driver-query-timing')
            receipt, _ = self.claim([valid] * 7 + [error] + [valid] * 4)
            self.assertEqual(receipt['release_metadata_verification'], {'status': 'verified'})

    def test_resident_idle_foreign_allocation_with_enough_free_memory_is_admitted(self):
        # Run 45185609661 shape: a foreign process holds ~36 GiB of GPU0 while ~58 GiB stay free.
        # With a 12 GiB child cap and 6 GiB reserve, a quiet card admits; occupancy is no veto.
        policy = SimpleNamespace(**{**vars(POLICY), 'child_gpu_cap_bytes': 12 * 1024**3,
                                    'gpu_free_reserve_bytes': 6 * 1024**3})
        occupied = self.card(2, used_mib=36864, free_mib=59392)
        foreign = [[UUID, '4242', 'foreign.exe', 'C']]
        receipt, count = self.claim([([occupied], foreign)] * 11, policy=policy, clock=FakeClock())
        self.assertEqual(count, 11)
        self.assertEqual(receipt['admission_utilization_percent'], [2] * 5)
        self.assertEqual(receipt['admission_wait']['stable_sample_restarts'], 0)

    def test_inventory_timeout_is_a_logged_retry_then_admits(self):
        # Run 45215252055: one selected-card nvidia-smi query timed out before admission.
        ready = ([self.card()], [])
        timeout = subprocess.TimeoutExpired(['nvidia-smi', '--id=' + UUID], 30)
        with patch('sys.stderr') as stderr:
            receipt, count = self.claim([timeout, *[ready] * 11], clock=FakeClock())
        self.assertEqual(count, 12)
        self.assertEqual(receipt['admission_wait']['inventory_timeouts'], 1)
        printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn('nvidia-smi inventory query timed out; retrying', printed)

    def test_busy_card_keeps_queueing_with_logged_reason_until_the_claim_deadline(self):
        # A busy card is a queueing condition, not a refusal: the job waits its turn (well past
        # one child allowance), logging the reason, and is refused only at the claim deadline.
        clock = FakeClock()
        policy = SimpleNamespace(**{**vars(POLICY), 'poll_millis': 5000})
        busy = self.card(99, used_mib=36864, free_mib=59392)
        with patch('sys.stderr') as stderr, \
                self.assertRaises(resources.CudaClaimRejected) as caught:
            self.claim(itertools.repeat(([busy], [])), policy=policy, clock=clock,
                       original_deadline_monotonic=1800., body=lambda _: self.fail('busy card reached the body'))
        self.assertGreaterEqual(clock.now, 1800.)
        diagnostic = caught.exception.diagnostic
        self.assertIn('deadline expired', diagnostic['admission_error'])
        self.assertGreater(diagnostic['admission_wait']['stable_sample_restarts'], 100)
        self.assertEqual(diagnostic['admission_wait']['last_refusal'], 'utilization 99% (> 30%)')
        printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn('restarting stable sampling: utilization 99% (> 30%)', printed)
        # Repeats of one reason are rate-limited to once a minute, not printed every poll.
        self.assertLessEqual(printed.count('utilization'), 1800 // 60 + 2)

    def test_insufficient_free_memory_waits_with_logged_reason_then_refuses_at_the_deadline(self):
        clock = FakeClock()
        policy = SimpleNamespace(**{**vars(POLICY), 'poll_millis': 5000})
        short = self.card(0, used_mib=90000, free_mib=7887)
        with patch('sys.stderr') as stderr, self.assertRaises(resources.CudaClaimRejected) as caught:
            self.claim(itertools.repeat(([short], [])), policy=policy, clock=clock,
                       original_deadline_monotonic=600., body=lambda _: self.fail('short card reached the body'))
        self.assertGreaterEqual(clock.now, 600.)
        wait = caught.exception.diagnostic['admission_wait']
        self.assertGreaterEqual(wait['headroom_shortfalls'], 2)
        self.assertIn('7887 MiB free < 10240 MiB required', wait['last_refusal'])
        printed = ''.join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn('CUDA admission waiting: ' + UUID + ': 7887 MiB free < 10240 MiB required', printed)

    def test_memory_and_physical_identity_refusals_never_enter_body(self):
        # CI_IMPACT: pure snapshots exercise initial and later phase failures, not model computation.
        invalid = [self.card(uuid=OTHER), self.card(index=0), self.card(total_mib='97887'),
                   self.card(used_mib=-1), self.card(free_mib=97888),
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
        policy = SimpleNamespace(**{**vars(POLICY), 'child_gpu_cap_bytes': 91 * 1024**3,
                                    'claim_gpu_estimate': ('test-stated-estimate', 91 * 1024**3)})
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
            self.claim(itertools.repeat(([release], [])), policy=SimpleNamespace(**{**vars(policy),'deadline_seconds':3}),
                       clock=FakeClock(), body=lambda _: self.fail('release snapshot admitted new work'))
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

    def test_callers_without_owned_job_record_seal_on_release_metadata_alone(self):
        # Comparison/allocator drivers reap their own supervised trees before the claim exits.
        record, _ = self.claim([([self.card()], [])] * 11, owned_completion=False, require_owned_completion=False)
        self.assertTrue(record['post_execution_verified'])
        self.assertTrue(record['lease_released'])

    def test_every_single_uuid_cuda_row_requires_the_headroom_release_seal(self):
        record, _ = self.claim([([self.card()], [])] * 11)
        body = {'argv': ['cargo', 'test'], 'supervision': {
            'pid': 1234, 'returncode': 0, 'owned_completion': record['owned_execution_completion'],
            'admission': {'originalClaimDeadlineMonotonic': record['claim_deadline_monotonic']}}}
        # A row that declares no profile and pins no device still gets the same seal check.
        plain = {'backend': 'candle-cuda', 'package': 'any', 'fqn': 'any::case'}
        resources.validate_shared_release_record(record, [body], plain)
        for mutation in (lambda r: r.update(activity_profile=resources.STRICT_CUDA_ACTIVITY),
                         lambda r: r.update(lease_released=False),
                         lambda r: r.pop('owned_execution_completion')):
            invalid = copy.deepcopy(record)
            mutation(invalid)
            with self.assertRaises(ValueError):
                resources.validate_shared_release_record(invalid, [body], plain)
        # Multi-UUID rows carry their own lease record and are not joined here.
        self.assertIsNone(resources.validate_shared_release_record(
            {'approved_uuid_roles': [], 'post_execution_verified': True}, [], {**plain, 'required_cuda_ordinals': [0, 1]}))

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
        # The admission claim deadline expiring during the body is not a release failure: post-run
        # sampling has its own fixed budget and the passing proof is sealed.
        with patch.object(resources.time, 'monotonic', return_value=100.) as clock:
            def expire(receipt):
                held.append(receipt)
                clock.return_value = 100. + POLICY.deadline_seconds
            receipt, _ = self.claim([valid] * 11, body=expire)
        self.assertEqual(receipt['release_metadata_verification'], {'status': 'verified'})
        self.assertTrue(held[-1]['post_execution_verified'])
        self.assertTrue(held[-1]['lease_released'])

    def test_legacy_profile_with_unselected_configuration_keeps_its_original_schema(self):
        legacy = {'activity_profile': resources.STRICT_CUDA_ACTIVITY, 'post_execution_verified': True}
        self.assertIsNone(resources.validate_shared_release_record(legacy, [], {'selected_configuration': None}))
        self.assertNotIn('owned_execution_completion', legacy)
        self.assertNotIn('lease_released', legacy)

    def test_release_seal_joins_exact_supervisor_owned_completion_and_lease(self):
        record, _ = self.claim([([self.card()], [])] * 11)
        body = {'argv': ['cargo', 'test'], 'supervision': {
            'pid': 1234, 'returncode': 0, 'owned_completion': record['owned_execution_completion'],
            'admission':{'originalClaimDeadlineMonotonic':record['claim_deadline_monotonic']}}}
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

    def test_projection_proofs_bind_only_the_registered_device_uuid_and_index(self):
        rows = [row for row in proof.load_registry()
                if (row['package'], row['target'], row['fqn']) in proof.PROJECTION_PROOF_DEVICES]
        self.assertEqual(len(rows), 6)
        for row in rows:
            with self.subTest(case=proof.identity(row)):
                configured = proof.configured_case(row, 'fixed_cuda')
                device = proof.PROJECTION_PROOF_DEVICES[(row['package'], row['target'], row['fqn'])]
                expected_uuid, expected_index = proof.training_comparison.DEVICES[device]
                self.assertEqual(configured['required_cuda_device_uuid'], expected_uuid)
                self.assertEqual(configured['required_cuda_physical_index'], expected_index)
                self.assertEqual(resources.bound_cuda_activity_profile(configured), PROFILE)
                with self.assertRaises(ValueError):
                    proof.select((row,), proof.identity(row), 'candle-cpu')
                for key, value in [('uuid', OTHER), ('index', 1 - expected_index)]:
                    modified = copy.deepcopy(row)
                    if key == 'uuid':
                        modified['required_cuda_device_uuid'] = value
                    elif key == 'index':
                        modified['configurations']['fixed_cuda']['required_cuda_physical_index'] = value
                    with self.assertRaises(ValueError):
                        proof.validate_configurations(modified)
                unregistered = {**configured, 'fqn': 'unbound_fixture'}
                with self.assertRaises(ValueError):
                    resources.bound_cuda_activity_profile(unregistered)

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



GPU0_UUID, GPU1_UUID = sorted(resources.REGISTERED_CUDA_CARDS, key=resources.REGISTERED_CUDA_CARDS.get)


class CardSelection(unittest.TestCase):
    """A proof that names no card runs on whichever registered card fits; no driver is touched."""

    def policies(self, *uuids):
        return {uuid: SimpleNamespace(cuda_device_uuid=uuid, poll_millis=250, gpu_free_reserve_bytes=6 * 1024**3,
                                      child_gpu_cap_bytes=84 * 1024**3,
                                      claim_gpu_estimate=('test-stated-estimate', 60 * 1024**3))
                for uuid in uuids}

    def card(self, uuid, used_mib):
        return {'uuid': uuid, 'index': resources.REGISTERED_CUDA_CARDS[uuid], 'total_mib': 97887,
                'used_mib': used_mib, 'free_mib': 97887 - used_mib, 'utilization_percent': 5}

    def select(self, polls, policies, deadline=100.):
        clock = FakeClock()
        script = iter(polls)
        state = {}
        def reader(uuid, _deadline):
            if uuid == GPU0_UUID or GPU0_UUID not in policies:
                state['poll'] = next(script)
            value = state['poll'][uuid]
            if isinstance(value, Exception):
                raise value
            return [value]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = resources.select_cuda_card(policies, deadline, reader=reader, pause=clock.sleep,
                                                clock=clock.monotonic)
        return result, stderr.getvalue(), clock.now

    def test_gpu0_busy_or_short_selects_gpu1_at_once(self):
        both = self.policies(GPU0_UUID, GPU1_UUID)
        for gpu0 in (self.card(GPU0_UUID, 40000), self.card(GPU0_UUID, 97000),
                     RuntimeError('MOCK GPU 0 inventory failed')):
            with self.subTest(gpu0=gpu0):
                (chosen, record), log, waited = self.select(
                    [{GPU0_UUID: gpu0, GPU1_UUID: self.card(GPU1_UUID, 3000)}], both)
                self.assertEqual((chosen, record['chosen_index'], record['polls'], waited), (GPU1_UUID, 1, 1, 0.))
                self.assertEqual(record['rule'], resources.CARD_SELECTION_RULE)
                refused = record['cards'][0]
                self.assertEqual(refused['uuid'], GPU0_UUID)
                self.assertIn('refusal', refused)
                self.assertIn(f'card 0 {GPU0_UUID} refused', log)
                self.assertIn(f'CUDA card selected: index 1 {GPU1_UUID}', log)

    def test_the_card_with_the_most_free_memory_wins_when_both_fit(self):
        (chosen, record), _, _ = self.select(
            [{GPU0_UUID: self.card(GPU0_UUID, 9000), GPU1_UUID: self.card(GPU1_UUID, 3000)}],
            self.policies(GPU0_UUID, GPU1_UUID))
        self.assertEqual(chosen, GPU1_UUID)
        (chosen, _), _, _ = self.select(
            [{GPU0_UUID: self.card(GPU0_UUID, 2000), GPU1_UUID: self.card(GPU1_UUID, 3000)}],
            self.policies(GPU0_UUID, GPU1_UUID))
        self.assertEqual(chosen, GPU0_UUID)

    def test_waits_only_while_no_card_fits_and_logs_each_refusal(self):
        short = {GPU0_UUID: self.card(GPU0_UUID, 40000), GPU1_UUID: self.card(GPU1_UUID, 45000)}
        freed = {GPU0_UUID: self.card(GPU0_UUID, 40000), GPU1_UUID: self.card(GPU1_UUID, 3000)}
        (chosen, record), log, waited = self.select([short, short, freed], self.policies(GPU0_UUID, GPU1_UUID))
        self.assertEqual((chosen, record['polls'], waited), (GPU1_UUID, 3, 2.))
        self.assertIn('no card fits, waiting', log)
        self.assertIn(f'card 1 {GPU1_UUID} refused: 52887 MiB free < 67584 MiB required', log)
        with self.assertRaises(resources.CudaClaimRejected) as caught, contextlib.redirect_stderr(io.StringIO()):
            clock = FakeClock()
            resources.select_cuda_card(self.policies(GPU0_UUID, GPU1_UUID), 3.,
                                       reader=lambda uuid, _: [short[uuid]], pause=clock.sleep, clock=clock.monotonic)
        self.assertEqual([card['uuid'] for card in caught.exception.diagnostic['cards']], [GPU0_UUID, GPU1_UUID])

    def test_a_card_free_proof_verifies_against_the_card_it_recorded_selecting(self):
        request = {'backend': 'windows-cuda', 'deadlineSeconds': 60, 'gpuFreeReserveBytes': 6 * 1024**3,
                   'childGpuCapBytes': 84 * 1024**3}
        (chosen, record), _, _ = self.select(
            [{GPU0_UUID: self.card(GPU0_UUID, 40000), GPU1_UUID: self.card(GPU1_UUID, 3000)}],
            self.policies(GPU0_UUID, GPU1_UUID))
        receipt = {'request_resource_policy': request, 'cuda_card_selection': record}
        self.assertEqual(proof.executed_policy(receipt, {'policy': request}), dict(request, cudaDeviceUuid=GPU1_UUID))
        swapped = copy.deepcopy(receipt)
        swapped['cuda_card_selection'].update(chosen_uuid=GPU0_UUID, chosen_index=0)
        for bad in (swapped, {'cuda_card_selection': record}, {}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'most-free registered card'):
                proof.executed_policy(bad, {'policy': request})
        named = dict(request, cudaDeviceUuid=GPU0_UUID)
        self.assertEqual(proof.executed_policy({}, {'policy': named}), named)
        with self.assertRaises(ValueError):
            proof.executed_policy(receipt, {'policy': named})

    def test_a_card_taken_before_the_first_claim_switches_to_the_other_card_and_records_why(self):
        request = {'schemaVersion': 1, 'backend': 'windows-cuda', 'deadlineSeconds': 60, 'pollMillis': 250,
                   'termGraceMillis': 2000, 'stdoutCapBytes': 4096, 'stderrCapBytes': 4096, 'eventCapBytes': 4096,
                   'childFootprintCapBytes': 64 * 1024**3, 'hostFreeReserveBytes': 8 * 1024**3,
                   'childGpuCapBytes': 84 * 1024**3, 'gpuFreeReserveBytes': 6 * 1024**3}
        estimate = ('test-stated-estimate', 60 * 1024**3)
        free = {GPU0_UUID: self.card(GPU0_UUID, 1000), GPU1_UUID: self.card(GPU1_UUID, 3000)}
        reader = lambda uuid, _deadline: [free[uuid]]
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()) as log:
            work = Path(temporary)
            payload, selection = resources.choose_card(request, work, estimate=estimate, reader=reader)
            self.assertEqual(payload['cudaDeviceUuid'], GPU0_UUID)
            abandon, state = resources.card_switcher(request, work, selection, estimate=estimate, reader=reader)
            # GPU 1 is short too: keep waiting for GPU 0 rather than switch.
            free[GPU1_UUID] = self.card(GPU1_UUID, 60000)
            self.assertFalse(abandon(f'{GPU0_UUID}: UUID lock held by another CI job'))
            self.assertIs(state['selection'], selection)
            # Another job took GPU 0 and GPU 1 fits: switch, and record the card left and why.
            free[GPU1_UUID] = self.card(GPU1_UUID, 3000)
            self.assertTrue(abandon(f'{GPU0_UUID}: UUID lock held by another CI job'))
        switched = state['selection']
        self.assertEqual((switched['chosen_uuid'], [card['uuid'] for card in switched['cards']]), (GPU1_UUID, [GPU1_UUID]))
        self.assertEqual(switched['superseded'][0]['left_uuid'], GPU0_UUID)
        self.assertIn('lock held', switched['superseded'][0]['reason'])
        self.assertIn(f'CUDA card switched: {GPU0_UUID} is taken', log.getvalue())
        self.assertEqual(resources.executed_card_policy(request, switched, estimate=estimate)['cudaDeviceUuid'], GPU1_UUID)
        for mutate in (lambda r: r['superseded'][0].update(left_uuid=GPU1_UUID),
                       lambda r: r['superseded'][0].update(reason=''),
                       lambda r: r['superseded'][0].update(switched_to=GPU0_UUID),
                       lambda r: r.update(superseded='x')):
            bad = copy.deepcopy(switched)
            mutate(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resources.executed_card_policy(request, bad, estimate=estimate)
        with self.assertRaisesRegex(ValueError, 'different estimate'):
            resources.executed_card_policy(request, switched, estimate=('other', 1))

    def test_a_switch_reads_the_other_card_with_its_own_driver_budget(self):
        # The real reader (veda_cuda_guard.command) refuses before running nvidia-smi once its
        # deadline has passed; a switch must still read the other card.
        request = {'schemaVersion': 1, 'backend': 'windows-cuda', 'deadlineSeconds': 60, 'pollMillis': 250,
                   'termGraceMillis': 2000, 'stdoutCapBytes': 4096, 'stderrCapBytes': 4096, 'eventCapBytes': 4096,
                   'childFootprintCapBytes': 64 * 1024**3, 'hostFreeReserveBytes': 8 * 1024**3,
                   'childGpuCapBytes': 84 * 1024**3, 'gpuFreeReserveBytes': 6 * 1024**3}
        estimate = ('test-stated-estimate', 60 * 1024**3)
        free = {GPU0_UUID: self.card(GPU0_UUID, 1000), GPU1_UUID: self.card(GPU1_UUID, 3000)}
        budgets = []

        def reader(uuid, deadline):
            budgets.append(deadline - time.monotonic())
            if deadline <= time.monotonic():
                raise TimeoutError('GPU inventory sampling deadline exhausted')
            return [free[uuid]]

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()) as log:
            _, selection = resources.choose_card(request, Path(temporary), estimate=estimate, reader=reader)
            abandon, state = resources.card_switcher(request, Path(temporary), selection, estimate=estimate, reader=reader)
            budgets.clear()
            self.assertTrue(abandon(f'{GPU0_UUID}: UUID lock held by another CI job'))
            self.assertEqual(len(budgets), 1)
            self.assertGreater(budgets[0], resources.SWITCH_READ_SECONDS - 5)
            # No card fits: one read, no polling, and the repeated refusal is logged once a minute.
            free[GPU0_UUID] = self.card(GPU0_UUID, 90000)
            budgets.clear()
            self.assertFalse(abandon(f'{GPU1_UUID}: lock held'))
            self.assertFalse(abandon(f'{GPU1_UUID}: lock held'))
            self.assertEqual(len(budgets), 2)
        self.assertEqual(log.getvalue().count(f'card 0 {GPU0_UUID} refused'), 1)
        self.assertEqual(state['selection']['chosen_uuid'], GPU1_UUID)

    def test_every_card_free_caller_switches_through_the_shared_claim_and_a_named_card_waits(self):
        request = {'schemaVersion': 1, 'backend': 'windows-cuda', 'deadlineSeconds': 60, 'pollMillis': 250,
                   'termGraceMillis': 2000, 'stdoutCapBytes': 4096, 'stderrCapBytes': 4096, 'eventCapBytes': 4096,
                   'childFootprintCapBytes': 64 * 1024**3, 'hostFreeReserveBytes': 8 * 1024**3,
                   'childGpuCapBytes': 12 * 1024**3, 'gpuFreeReserveBytes': 2 * 1024**3}
        free = {GPU0_UUID: self.card(GPU0_UUID, 1000), GPU1_UUID: self.card(GPU1_UUID, 3000)}
        reader = lambda uuid, _deadline: [free[uuid]]
        attempts, rebound = [], []

        @contextlib.contextmanager
        def claim_with(backend, policy, output, env, *, abandon_wait=None, **_options):
            attempts.append((policy.cuda_device_uuid, output.name, abandon_wait is not None))
            if len(attempts) == 1 and abandon_wait is not None and abandon_wait(f'{policy.cuda_device_uuid}: lock held'):
                raise resources.CudaCardTaken({'abandoned_wait': 'lock held'})
            yield {'device_uuid': policy.cuda_device_uuid}

        from scripts.media_campaign_supervisor import load_policy

        def claim_policy(chosen, root):
            path = root / 'claim-policy.json'
            path.write_text(json.dumps(chosen), encoding='utf-8')
            return load_policy(path)

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()):
            root = Path(temporary)
            payload, selection = resources.choose_card(request, root, reader=reader)
            record = {'cuda_card_selection': selection}
            with resources.claim_chosen_card(
                    'candle-cuda', request, payload, selection, root, {}, claim_policy=lambda c: claim_policy(c, root),
                    rebind=rebound.append, record=record, claim_with=claim_with, selector={'reader': reader}) as held:
                pass
        self.assertEqual(attempts[0][0], GPU0_UUID)
        self.assertEqual((held['device_uuid'], attempts[1][1]), (GPU1_UUID, 'switch-2'))
        self.assertEqual(rebound[0]['cudaDeviceUuid'], GPU1_UUID)
        self.assertEqual(record['cuda_card_selection']['superseded'][0]['left_uuid'], GPU0_UUID)
        self.assertEqual(resources.executed_card_policy(request, record['cuda_card_selection'])['cudaDeviceUuid'], GPU1_UUID)
        # A named card is never switched: no abandon hook reaches its claim.
        attempts.clear()
        with tempfile.TemporaryDirectory() as temporary:
            named = dict(request, cudaDeviceUuid=GPU0_UUID)
            with resources.claim_chosen_card('candle-cuda', named, named, None, Path(temporary), {},
                                             claim_policy=lambda c: claim_policy(c, Path(temporary)),
                                             claim_with=claim_with) as held:
                pass
        self.assertEqual(attempts, [(GPU0_UUID, Path(temporary).name, False)])

    def test_the_claim_gives_up_its_card_only_when_the_caller_can_switch(self):
        short = {'uuid': UUID, 'index': 1, 'total_mib': 97887, 'used_mib': 60000, 'free_mib': 37887,
                 'utilization_percent': 4}
        policy = SimpleNamespace(**{**vars(POLICY), 'child_gpu_cap_bytes': 84 * 1024**3,
                                    'gpu_free_reserve_bytes': 6 * 1024**3, 'deadline_seconds': 20,
                                    'claim_gpu_estimate': ('test-stated-estimate', 60 * 1024**3)})
        harness = SharedHeadroom('test_comparison_absolute_deadline_survives_slow_discovery_before_claim')
        harness.setUpClass()
        asked = []
        def claim(answer):
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(resources, 'os', SimpleNamespace(name='nt', fstat=os.fstat)), \
                    patch.dict(sys.modules, {'msvcrt': SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_: None)}), \
                    patch.object(guard, 'inventory', side_effect=itertools.repeat(([short], []))), \
                    patch.object(proof, 'load_registry', return_value=(harness.source,)), \
                    contextlib.redirect_stderr(io.StringIO()):
                clock = FakeClock()
                with patch.object(resources.time, 'monotonic', side_effect=clock.monotonic), \
                        patch.object(resources.time, 'sleep', side_effect=clock.sleep):
                    with resources.execution_claim('candle-cuda', policy, Path(temporary), {'PUBLIC': temporary},
                                                   constraints=harness.configured, require_owned_completion=True,
                                                   abandon_wait=lambda reason: asked.append(reason) or answer):
                        self.fail('a short card reached the body')
        with self.assertRaises(resources.CudaCardTaken) as caught:
            claim(True)
        self.assertIn('MiB free < 67584 MiB required', caught.exception.diagnostic['abandoned_wait'])
        self.assertEqual(len(asked), 1)
        policy.deadline_seconds = .003
        with self.assertRaises(resources.CudaClaimRejected) as caught:
            claim(False)
        self.assertNotIsInstance(caught.exception, resources.CudaCardTaken)

    def test_a_named_card_waits_for_itself_and_unregistered_cards_are_refused(self):
        short = {GPU0_UUID: self.card(GPU0_UUID, 40000)}
        ready = {GPU0_UUID: self.card(GPU0_UUID, 1000)}
        (chosen, record), log, waited = self.select([short, ready], self.policies(GPU0_UUID))
        self.assertEqual((chosen, waited), (GPU0_UUID, 1.))
        self.assertIn('no card fits, waiting', log)
        with self.assertRaises(ValueError):
            resources.select_cuda_card(self.policies(OTHER), 1.)
        moved = {GPU0_UUID: dict(self.card(GPU0_UUID, 1000), index=1), GPU1_UUID: self.card(GPU1_UUID, 3000)}
        (chosen, record), _, _ = self.select([moved], self.policies(GPU0_UUID, GPU1_UUID))
        self.assertEqual((chosen, record['cards'][0]['refusal']), (GPU1_UUID, 'card moved to index 1'))

if __name__ == '__main__':
    unittest.main()
