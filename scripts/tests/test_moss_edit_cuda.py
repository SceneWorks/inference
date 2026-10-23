"""CPU-only reservation/orchestration/ownership tests. Never query a GPU or CIM."""
import copy
import datetime as dt
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from scripts.ci import moss_edit_cuda as gate

NOW = dt.datetime(2026, 10, 23, 12, tzinfo=dt.timezone.utc)
TIME = '2026-10-23T12:00:00Z'
ENV = {'GITHUB_SHA': 'a' * 40, 'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_RUN_ATTEMPT': '1',
       'GITHUB_ACTOR': 'coordinator', 'COMPUTERNAME': 'MICHAEL-TRX50', 'GITHUB_RUN_ID': '123'}


def lease():
    return dict(schemaVersion=1, reservationId='reviewed-once', source=ENV['GITHUB_SHA'], actor=ENV['GITHUB_ACTOR'],
                host=ENV['COMPUTERNAME'], physicalGpu=1, uuid='GPU-1234', pci='00000000:C1:00.0', driver='596.36',
                counterLuid='luid_0x00000000_0x00020806', reviewedEvidence=dict(run='122', sha256='b' * 64, completedUtc=TIME),
                ownerIdleConfirmedUtc=TIME, validFromUtc=TIME, expiresUtc='2026-10-23T13:00:00Z', testTimeoutSeconds=60,
                processes=[dict(pid=42, name='reviewed.exe', createdUtc='2026-10-23T10:00:00.0000000Z', contextType='C+G',
                                path=None, engineTypes=['3d', 'copy', 'compute_0'], allowedActiveEngineTypes=['3d', 'copy'])])


def evidence(spec):
    process = spec['processes'][0]
    identity = dict(pid=42, name=process['name'], createdUtc=process['createdUtc'], status='resolved', path=None, pathAvailable=False)
    device = dict(index='1', uuid=spec['uuid'], pci=spec['pci'], driver=spec['driver'], driverModel='WDDM',
                  totalMiB='100', usedMiB='10', utilization='0', measurements=dict(status='valid', totalMiB=100, usedMiB=10, utilization=0))
    census = [dict(pid=42, type='C+G', name='', memory='N/A')]
    counters = []
    for category in sorted(gate.COUNTERS):
        adapter = category.startswith('GPU Adapter Memory')
        counters.append(dict(counter='\\' + category, utc=TIME, status=1, valid=True, valuesValidated=True, value=0, rawValue='0', pid=None if adapter else 42,
                             instance=('' if adapter else 'pid_42_') + spec['counterLuid'] + '_phys_0_eng_0_engtype_3d'))
    sample = dict(device=device, windowStartedUtc=TIME, windowEndedUtc=TIME, processes=census, processesAfter=copy.deepcopy(census),
                  identitiesBefore=[identity], identitiesAfter=[copy.deepcopy(identity)], counters=counters)
    return dict(completed=True, collectionComplete=True, telemetryComplete=True, evidenceTruncated=False, errors=[],
                admission='not_evaluated', applicationJobOwnership='unknown', physicalGpu=1, source=spec['source'], host=spec['host'],
                queries=[dict(exitCode=0)], run='123', attempt='1', startedUtc=TIME, completedUtc=TIME, identity=device,
                mapping=dict(status='mapped', counterLuid=spec['counterLuid']), samples=[copy.deepcopy(sample) for _ in range(3)],
                optionalMetadataWarnings=['Executable path unavailable for PID 42'])


def final_record(full):
    final = {k: copy.deepcopy(v) for k, v in full.items() if k not in ('samples', 'mapping')}
    final.update(snapshotKind='identity-census-only', finalSnapshotComplete=True, collectionComplete=False, telemetryComplete=False,
                 processes=copy.deepcopy(full['samples'][-1]['processesAfter']), identities=copy.deepcopy(full['samples'][-1]['identitiesAfter']))
    return final


class AdmissionTests(unittest.TestCase):
    def test_optional_paths_and_reviewed_activity_pass_without_inferring_admission(self):
        spec = lease(); record = evidence(spec)
        for sample in record['samples']:
            row = next(r for r in sample['counters'] if 'GPU Engine' in r['counter'])
            row.update(value=0.75, rawValue='0.75')
        gate.validate_evidence(record, spec, ENV, NOW, NOW)
        gate.validate_final(final_record(record), record, spec, ENV, NOW)
        self.assertEqual(record['admission'], 'not_evaluated')
        self.assertTrue(record['optionalMetadataWarnings'])

    def test_reservation_rejects_context_source_actor_expiry_and_compute_approval(self):
        for key, value in [('source', 'c' * 40), ('actor', 'foreign'), ('host', 'OTHER'), ('physicalGpu', 0),
                           ('expiresUtc', TIME), ('validFromUtc', '2026-10-23T12:01:00Z'), ('testTimeoutSeconds', 0)]:
            with self.subTest(key=key):
                spec = lease(); spec[key] = value
                with self.assertRaises(ValueError): gate.validate_reservation(spec, ENV, NOW)
        for key, value in [('GITHUB_EVENT_NAME', 'push'), ('GITHUB_RUN_ATTEMPT', '2')]:
            with self.assertRaises(ValueError): gate.validate_reservation(lease(), dict(ENV, **{key: value}), NOW)
        for engine in ('compute_0', 'videoencode'):
            spec = lease(); spec['processes'][0]['engineTypes'].append(engine) if engine == 'videoencode' else None
            spec['processes'][0]['allowedActiveEngineTypes'].append(engine)
            with self.assertRaises(ValueError): gate.validate_reservation(spec, ENV, NOW)
        with self.assertRaises(ValueError): json.loads('{"source":1,"source":2}', object_pairs_hook=gate.unique_fields)

    def test_all_mandatory_telemetry_and_new_process_failures_refuse(self):
        mutations = [
            lambda r: r.update(errors=['unknown']), lambda r: r.update(telemetryComplete=False),
            lambda r: r.update(evidenceTruncated=True), lambda r: r['mapping'].update(counterLuid='foreign'),
            lambda r: r['identity']['measurements'].update(usedMiB=float('nan')),
            lambda r: r['identity'].update(usedMiB='N/A'),
            lambda r: r['samples'][0]['identitiesBefore'][0].update(pid=43),
            lambda r: r['samples'][0]['identitiesBefore'][0].update(name=''),
            lambda r: r['samples'][0]['identitiesBefore'][0].update(createdUtc='reused'),
            lambda r: r['samples'][0]['identitiesBefore'][0].update(status='unknown'),
            lambda r: r['samples'][0]['processesAfter'][0].update(type='C'),
            lambda r: r['samples'][0]['counters'][0].update(status=2),
            lambda r: r['samples'][0]['counters'][0].update(value=None),
            lambda r: r['samples'][0]['counters'].pop(),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                spec = lease(); record = evidence(spec); mutate(record)
                with self.assertRaises((ValueError, TypeError)): gate.validate_evidence(record, spec, ENV, NOW, NOW)

    def test_active_compute_new_engine_and_unapproved_activity_refuse(self):
        for engine, value, allow in [('compute_0', 0.1, True), ('newengine', 0, True), ('3d', 0.1, False)]:
            spec = lease(); record = evidence(spec)
            if not allow: spec['processes'][0]['allowedActiveEngineTypes'] = []
            row = next(r for r in record['samples'][0]['counters'] if 'GPU Engine' in r['counter'])
            row.update(instance=row['instance'].replace('engtype_3d', 'engtype_' + engine), value=value, rawValue=str(value))
            with self.assertRaises(ValueError): gate.validate_evidence(record, spec, ENV, NOW, NOW)

    def test_counter_only_identity_and_known_path_mismatch(self):
        spec = lease(); spec['processes'][0]['contextType'] = None
        record = evidence(spec)
        for sample in record['samples']:
            sample['processes'] = []; sample['processesAfter'] = []
        gate.validate_evidence(record, spec, ENV, NOW, NOW)
        spec['processes'][0]['path'] = r'C:\expected.exe'
        row = record['samples'][0]['identitiesAfter'][0]
        row.update(pathAvailable=True, path=r'C:\other.exe')
        with self.assertRaises(ValueError): gate.validate_evidence(record, spec, ENV, NOW, NOW)

    def test_final_snapshot_requires_same_incarnations_after_full_window(self):
        for mutate in (lambda r: r.update(finalSnapshotComplete=False), lambda r: r['identities'][0].update(createdUtc='reused'),
                       lambda r: r['processes'][0].update(type='C'), lambda r: r.update(startedUtc='2026-10-23T11:59:59Z')):
            spec = lease(); full = evidence(spec); final = final_record(full); mutate(final)
            with self.assertRaises(ValueError): gate.validate_final(final, full, spec, ENV, NOW)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.out = self.root / 'run'; self.out.mkdir()
        self.target = self.root / 'target'; self.target.mkdir()
        self.spec = lease(); self.env = dict(ENV, CARGO_TARGET_DIR=str(self.target))
        self.calls = []; self.fail_arm = None; self.mutate_arm = None; self.bad_list = False; self.bad_admission = False; self.ignored_mutation = None; self.skipped_arm = None; self.zero_arm = None

    def runner(self, args, env, out, label, seconds, **policy):
        self.calls.append((label, args, env.copy()))
        self.assertGreater(seconds, 0)
        expected_policy = {'finalize_compile_descendants': True} if label == 'compile' else (
            {'finalize_failed_collector_descendants': True} if label.endswith('-collector') else {})
        self.assertEqual(policy, expected_policy)
        if label == 'compile':
            if self.fail_arm == 'compile': return 1, ''
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1'); self.assertIn('--no-run', args)
            rows = []
            for package in (gate.MOSS, gate.FLUX):
                path = self.target / (package + '.exe'); path.write_bytes(package.encode())
                rows.append(dict(reason='compiler-artifact', executable=str(path), target=dict(name=package.replace('-', '_'), kind=['lib']), profile=dict(test=True)))
            return 0, '\n'.join(map(json.dumps, rows))
        if label.endswith(('-list', '-ignored')):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1')
            arm = next(arm for arm in gate.ARMS if label.startswith(arm[0]))
            names = arm[4] if label.endswith('-ignored') else arm[3]
            if self.ignored_mutation and label == self.ignored_mutation[0]: names = self.ignored_mutation[1]
            return 0, '' if self.bad_list else '\n'.join(name + ': test' for name in sorted(names))
        if label.endswith(('-collector', '-refresh')):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1')
            directory = Path(args[args.index('-OutputDirectory') + 1]); directory.mkdir()
            record = evidence(self.spec)
            if self.bad_admission: record['errors'] = ['query failed']
            if label.endswith('-refresh'): record = final_record(record)
            (directory / 'evidence.json').write_text(json.dumps(record), encoding='utf-8')
            if self.mutate_arm and label.startswith(self.mutate_arm + '-admission-') and label.endswith('-refresh'):
                (self.target / (gate.MOSS + '.exe')).write_bytes(b'tampered')
            return 0, ''
        self.assertTrue(label.endswith('-tests'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], self.spec['uuid']); self.assertEqual(env['SCENEWORKS_GPU_ID'], '1')
        self.assertIn('--test-threads', args); self.assertNotIn('--ignored', args)
        arm = next(arm for arm in gate.ARMS if label == arm[0] + '-tests')
        self.assertEqual(args[1:-3], arm[2] + (['--include-ignored'] if arm[4] else []))
        if label == self.skipped_arm: return 0, 'test result: ok. 0 passed; 0 failed; 1 ignored;'
        if label == self.zero_arm: return 0, 'test result: ok. 0 passed; 0 failed; 0 ignored;'
        if label == self.fail_arm: return 1, 'test result: FAILED. 0 passed; 1 failed; 0 ignored;'
        return 0, f'test result: ok. {len(arm[3])} passed; 0 failed; 0 ignored;'

    def run_gate(self):
        return gate.execute(self.spec, self.env, self.out, runner=self.runner, now=lambda: NOW, sleep=lambda _: None, source_check=lambda: {'fixed': 'source'})

    def test_two_binaries_all_compilation_before_three_exact_invocations(self):
        result = self.run_gate(); self.assertEqual(len(result), 3)
        labels = [row[0] for row in self.calls]
        self.assertEqual(labels[0], 'compile'); self.assertEqual(labels.count('compile'), 1)
        self.assertEqual([x for x in labels if x.endswith('-tests')], ['edit-tests', 'layout-tests', 'encoder-tests'])
        for index, label in enumerate(labels):
            if label.endswith('-tests'): self.assertEqual(labels[index-1], label.replace('-tests', '-admission-001-refresh'))
        self.assertEqual([len(arm[3]) for arm in gate.ARMS], [2, 1, 1])
        self.assertEqual([arm[4] for arm in gate.ARMS], [set(), {
            'transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts'}, set()])
        self.assertEqual(sum(len(row['expectedTests']) for row in result), 4)

    def test_compilation_disables_ambient_and_config_wrappers_only_in_child_environment(self):
        self.env.update(RUSTC_WRAPPER='ambient-sccache.exe', RUSTC_WORKSPACE_WRAPPER='ambient-workspace.exe',
                        CARGO_BUILD_RUSTC_WRAPPER='configured-sccache.exe',
                        CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER='configured-workspace.exe')
        original = self.env.copy()
        self.run_gate()
        self.assertEqual(self.env, original)
        label, command, child_env = self.calls[0]
        self.assertEqual(label, 'compile')
        self.assertEqual(command, ['cargo', 'test', '--locked', '--lib', '-p', gate.MOSS, '-p', gate.FLUX,
                                   '--features', 'cuda', '--no-run', '--message-format=json'])
        for _, args, child_env in self.calls:
            self.assertEqual(child_env['RUSTC_WRAPPER'], '')
            self.assertEqual(child_env['RUSTC_WORKSPACE_WRAPPER'], '')
            self.assertFalse(any('sccache' in arg for arg in args))
            # Empty direct overrides supersede config without changing caller/global configuration.
            self.assertEqual(child_env['CARGO_BUILD_RUSTC_WRAPPER'], original['CARGO_BUILD_RUSTC_WRAPPER'])
            self.assertEqual(child_env['CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'], original['CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'])

    def test_no_gpu_execution_on_missing_cases_or_failed_admission(self):
        for mode in ('bad_list', 'bad_admission'):
            with self.subTest(mode=mode):
                # execute owns fresh cache directories; each negative run has its own root.
                self.out = self.root / mode; self.out.mkdir(); self.calls = []; setattr(self, mode, True)
                with self.assertRaises(ValueError): self.run_gate()
                self.assertFalse(any(label.endswith('-tests') for label, _, _ in self.calls)); setattr(self, mode, False)

    def test_ignored_inventory_drift_refuses_before_any_gpu_execution(self):
        cases = [('edit-ignored', {'transformer::edit::tests::frozen_edit_oracle_cuda'}),
                 ('layout-ignored', set()),
                 ('layout-ignored', gate.ARMS[1][4] | {'unexpected_cuda_test'}),
                 ('encoder-ignored', {'vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda'})]
        for index, mutation in enumerate(cases):
            with self.subTest(mutation=mutation):
                self.out = self.root / f'ignored-{index}'; self.out.mkdir(); self.calls = []
                self.ignored_mutation = mutation
                with self.assertRaisesRegex(ValueError, 'missing/extra/ignored'): self.run_gate()
                self.assertFalse(any(label.endswith('-tests') for label, _, _ in self.calls))

    def test_skipped_or_zero_executed_layout_cannot_pass(self):
        for mode in ('skipped_arm', 'zero_arm'):
            with self.subTest(mode=mode):
                self.out = self.root / mode; self.out.mkdir(); self.calls = []
                setattr(self, mode, 'layout-tests')
                with self.assertRaisesRegex(ValueError, 'numerical arm failed'): self.run_gate()
                results = json.loads((self.out/'results.json').read_text(encoding='utf-8'))
                self.assertEqual([row['passed'] for row in results], [True, False])
                setattr(self, mode, None)

    def test_tampered_binary_after_final_refresh_is_not_launched(self):
        self.mutate_arm = 'edit'
        with self.assertRaisesRegex(ValueError, 'binary changed'): self.run_gate()
        self.assertFalse(any(label.endswith('-tests') for label, _, _ in self.calls))

    def test_nonzero_compile_stops_before_listing_or_gpu(self):
        self.fail_arm = 'compile'
        with self.assertRaisesRegex(ValueError, 'test compilation failed'): self.run_gate()
        self.assertEqual([label for label, _, _ in self.calls], ['compile'])

    def test_nonzero_test_exit_is_terminal_without_retry_or_later_arms(self):
        self.fail_arm = 'edit-tests'
        with self.assertRaisesRegex(ValueError, 'numerical arm failed'): self.run_gate()
        self.assertEqual(len([x for x, _, _ in self.calls if x.endswith('-tests')]), 1)
        self.assertFalse(json.loads((self.out / 'results.json').read_text(encoding='utf-8'))[0]['passed'])
        self.assertFalse(gate.successful_cases(0, 'test result: ok. 0 passed; 0 failed; 0 ignored;', {'expected'}))
        self.assertFalse(gate.successful_cases(0, 'test result: ok. 1 passed; 0 failed; 1 ignored;', {'expected'}))

    def test_missing_foreign_and_tampered_prepared_executables_refuse(self):
        with self.assertRaises(ValueError): gate.prepared_binaries('', self.target)
        path = self.root / 'foreign.exe'; path.write_bytes(b'x')
        with self.assertRaises(ValueError): gate.checked_binary(dict(path=str(path), sha256=gate.digest(path)), self.target)
        with self.assertRaises(FileNotFoundError): gate.checked_binary(dict(path=str(self.target/'missing.exe'), sha256='0'*64), self.target)


    def test_busy_and_incomplete_attempts_remain_queued_then_each_arm_executes_once(self):
        sleeps = []
        def runner(*args, **kwargs):
            code, output = self.runner(*args, **kwargs)
            command, env, out, label, seconds = args
            if label in ('edit-admission-001-collector', 'edit-admission-002-collector'):
                path = Path(command[command.index('-OutputDirectory') + 1]) / 'evidence.json'
                record = json.loads(path.read_text(encoding='utf-8'))
                record['telemetryComplete'] = False
                record['errors'] = ['changing foreign process census']
                path.write_text(json.dumps(record), encoding='utf-8')
                return (1 if '001' in label else 0), output
            return code, output
        results = gate.execute(self.spec, self.env, self.out, runner=runner, now=lambda: NOW,
                               sleep=sleeps.append, source_check=lambda: {'fixed': 'source'})
        self.assertEqual(sleeps, [30, 30]); self.assertEqual(len(results), 3)
        self.assertEqual([name for name, _, _ in self.calls if name.endswith('-tests')], ['edit-tests', 'layout-tests', 'encoder-tests'])
        attempts = sorted(self.out.glob('edit-admission-*/admission.json'))
        self.assertEqual(len(attempts), 3)
        self.assertEqual([json.loads(p.read_text(encoding='utf-8'))['admitted'] for p in attempts], [False, False, True])
        self.assertTrue(all((p.parent / 'full/evidence.json').is_file() for p in attempts))
        self.assertFalse((attempts[0].parent / 'final').exists())
        self.assertTrue((attempts[-1].parent / 'final/evidence.json').is_file())

    def test_cleanup_source_and_expiry_failures_do_not_retry_or_launch(self):
        for failure in ('cleanup', 'source', 'expiry'):
            with self.subTest(failure=failure):
                self.out = self.root / failure; self.out.mkdir(); self.calls = []
                source = {'fixed': 'source'}; clock = [NOW]; sleeps = []
                def runner(*args, **kwargs):
                    result = self.runner(*args, **kwargs)
                    if args[3].endswith('-collector'):
                        if failure == 'cleanup': raise ValueError('owned cleanup incomplete')
                        if failure == 'source': source['fixed'] = 'corrupt'
                        if failure == 'expiry': return 1, ''
                    return result
                def sleep(seconds):
                    sleeps.append(seconds); clock[0] = NOW + dt.timedelta(hours=2)
                with self.assertRaises(ValueError):
                    gate.execute(self.spec, self.env, self.out, runner=runner, now=lambda: clock[0], sleep=sleep, source_check=lambda: source.copy())
                self.assertEqual(len([label for label, _, _ in self.calls if label.endswith('-collector')]), 1)
                self.assertFalse(any(label.endswith('-tests') for label, _, _ in self.calls))
                self.assertEqual(sleeps, [30] if failure == 'expiry' else [])
                self.assertEqual(len(list(self.out.glob('edit-admission-*/admission.json'))), 1)

    def test_expiring_lease_refuses_before_starting_incomplete_arm_budget(self):
        clock = NOW + dt.timedelta(minutes=55)
        with self.assertRaisesRegex(ValueError, 'complete admission and invocation budget'):
            gate.execute(self.spec, self.env, self.out, runner=self.runner, now=lambda: clock, sleep=lambda _: self.fail('unexpected retry'), source_check=lambda: {})
        self.assertFalse(any(label.endswith(('-collector', '-tests')) for label, _, _ in self.calls))


class OwnershipTests(unittest.TestCase):
    def test_suspended_assignment_cleanup_and_deadline_only_touch_owned_child(self):
        for mode in ('success', 'transient', 'timeout', 'assign-failure', 'leftover', 'short-deadline', 'query-error', 'query-overrun'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                events = []; clock = [0.0]; observations = [0]
                class Child:
                    pid = 123
                    returncode = 0
                    def poll(self):
                        if mode == 'short-deadline': clock[0] = 0.99
                        return None if mode == 'timeout' else 0
                    def kill(self): events.append('child-kill')
                    def wait(self, timeout): events.append('child-wait')
                class Job:
                    def __init__(self): self.live = False
                    def assign_and_resume(self, child):
                        events.append('assign')
                        if mode == 'assign-failure': raise RuntimeError('assignment denied')
                        self.live = mode in ('transient', 'timeout', 'leftover', 'short-deadline', 'query-error')
                    def members(self):
                        observations[0] += 1
                        if mode == 'query-error' and observations[0] == 1: raise RuntimeError('Job query failed')
                        if mode == 'transient' and observations[0] >= 3: self.live = False
                        if mode == 'query-overrun': clock[0] = 1.01
                        return {456 if mode in ('leftover', 'short-deadline') else 123} if self.live else set()
                    def terminate_and_reap(self, child, grace): events.append('owned-job-terminate'); self.live = False
                    def close(self): events.append('close')
                def spawn(args, **kwargs):
                    self.assertEqual(kwargs['creationflags'], 0x00000004); events.append('suspended-spawn'); return Child()
                def sleep(seconds): clock[0] += seconds
                with patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate.time, 'sleep', side_effect=sleep):
                    if mode in ('success', 'transient'): gate.run_owned(['fixture'], {}, Path(directory), 'case', 2 if mode == 'query-overrun' else 1, job_type=Job, popen=spawn)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): gate.run_owned(['fixture'], {}, Path(directory), 'case', 2 if mode == 'query-overrun' else 1, job_type=Job, popen=spawn)
                self.assertEqual(events[:2], ['suspended-spawn', 'assign']); self.assertEqual(events[-1], 'close')
                self.assertEqual('owned-job-terminate' in events, mode in ('timeout', 'leftover', 'short-deadline', 'query-error'))
                self.assertEqual('child-kill' in events, mode == 'assign-failure')
                receipt = json.loads((Path(directory)/'case.owned.json').read_text(encoding='utf-8'))
                self.assertEqual(receipt['remainingOwnedPids'], [])
                if mode not in ('timeout', 'assign-failure'):
                    self.assertEqual(receipt['returncode'], 0)
                    self.assertIn('rootExitObservedUtc', receipt)
                    self.assertLessEqual(len(receipt['jobDrainObservations']), 21)
                if mode == 'transient':
                    self.assertEqual([r['pids'] for r in receipt['jobDrainObservations']], [[123], [123], []])
                    self.assertNotIn('error', receipt)
                if mode in ('leftover', 'short-deadline'):
                    self.assertEqual(receipt['preCleanupOwnedPids'], [456])
                    self.assertTrue(all(r['pids'] == [456] for r in receipt['jobDrainObservations']))
                    self.assertIn('survived root exit drain', receipt['error'])
                    self.assertLessEqual(clock[0], 1.0)
                if mode == 'short-deadline':
                    self.assertEqual(len(receipt['jobDrainObservations']), 2)
                if mode == 'query-overrun':
                    self.assertEqual(receipt['error'], 'owned Job drain deadline exceeded')
                    self.assertEqual(receipt['jobDrainObservations'], [{'elapsedSeconds': 1.01, 'pids': []}])
                if mode == 'query-error':
                    self.assertEqual(receipt['error'], 'Job query failed')
                    self.assertEqual(receipt['jobDrainObservations'], [])


    def test_compile_finalization_requires_completed_owned_cleanup(self):
        for mode in ('success', 'nonzero', 'cleanup-error', 'cleanup-survivor', 'timeout', 'query-error', 'deadline', 'final-output', 'output-overflow'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                clock = [0.0]; events = []; queried = [False]
                class Child:
                    pid = 123
                    returncode = 7 if mode == 'nonzero' else 0
                    def poll(self): return None if mode == 'timeout' else self.returncode
                class Job:
                    def __init__(self): self.live = True
                    def assign_and_resume(self, child): events.append('assigned')
                    def members(self):
                        if mode == 'query-error' and not queried[0]:
                            queried[0] = True; raise RuntimeError('query failed')
                        return {456} if self.live else set()
                    def terminate_and_reap(self, child, grace):
                        events.append('owned-job-finalize')
                        if mode == 'cleanup-error': raise RuntimeError('cleanup failed')
                        if mode != 'cleanup-survivor': self.live = False
                        if mode in ('final-output', 'output-overflow'):
                            (Path(directory)/'compile.stdout.log').write_text('final output', encoding='utf-8')
                    def close(self): events.append('closed')
                def sleep(seconds): clock[0] += seconds
                def invoke():
                    return gate.run_owned(['cargo', 'test'], {}, Path(directory), 'compile',
                                          1 if mode == 'deadline' else 2,
                                          finalize_compile_descendants=True, job_type=Job,
                                          popen=lambda *args, **kwargs: Child())
                with patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate.time, 'sleep', side_effect=sleep), patch.object(gate, 'MAX_LOG', 3 if mode == 'output-overflow' else gate.MAX_LOG):
                    if mode in ('success', 'nonzero', 'final-output'):
                        code, output = invoke(); events.append('returned')
                        self.assertEqual(output, 'final output' if mode == 'final-output' else '')
                        self.assertEqual(code, 7 if mode == 'nonzero' else 0)
                        self.assertEqual(events, ['assigned', 'owned-job-finalize', 'closed', 'returned'])
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): invoke()
                receipt = json.loads((Path(directory)/'compile.owned.json').read_text(encoding='utf-8'))
                self.assertTrue(receipt['finalizeCompileDescendants'])
                if mode in ('success', 'nonzero', 'final-output'):
                    self.assertEqual(receipt['preCleanupOwnedPids'], [456])
                    self.assertEqual(receipt['remainingOwnedPids'], [])
                    self.assertIn('cleanupCompletedUtc', receipt)
                    self.assertNotIn('error', receipt)
                if mode == 'output-overflow':
                    self.assertEqual(receipt['cleanupError'], 'owned output exceeded bound after cleanup')
                    self.assertEqual(receipt['remainingOwnedPids'], [])
                if mode.startswith('cleanup-'):
                    self.assertIn('cleanupError', receipt)
                    self.assertNotIn('cleanupCompletedUtc', receipt)
        with self.assertRaisesRegex(ValueError, 'compile-only'):
            gate.run_owned(['test'], {}, Path('.'), 'edit-tests', 1, finalize_compile_descendants=True)


class CollectorFinalizationTests(unittest.TestCase):
    def run_case(self, directory, mode, label='edit-admission-005-collector', policy=True):
        events, clock = [], [0.0]
        class Child:
            pid = 123
            returncode = 0 if mode == 'success' else 7
            def poll(self): return None if mode == 'timeout' else self.returncode
            def wait(self, timeout): return self.returncode
        class Job:
            live = True
            def assign_and_resume(self, child): events.append('assigned')
            def members(self):
                if mode == 'query-error': raise RuntimeError('membership failed')
                return {456} if self.live else set()
            def member_identities(self, members):
                events.append('owned-metadata')
                return {'processes': [{'pid': 456, 'status': 'open-unavailable', 'path': None}]}
            def terminate_and_reap(self, child, grace):
                events.append('owned-finalize')
                if mode == 'cleanup-error': raise RuntimeError('cleanup failed')
                if mode != 'cleanup-survivor': self.live = False
            def close(self):
                events.append('close')
                if mode == 'close-error': raise RuntimeError('CloseHandle/job failed')
        def sleep(seconds): clock[0] += seconds
        with patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate.time, 'sleep', side_effect=sleep):
            result = gate.run_owned(['powershell'], {}, directory, label, 2,
                                    finalize_failed_collector_descendants=policy,
                                    job_type=Job, popen=lambda *a, **kw: Child())
        return result, events

    def test_nonzero_collector_returns_original_refusal_only_after_checked_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result, events = self.run_case(directory, 'nonzero')
            self.assertEqual(result, (7, ''))
            self.assertEqual(events, ['assigned', 'owned-metadata', 'owned-finalize', 'close'])
            record = json.loads((directory/'edit-admission-005-collector.owned.json').read_text(encoding='utf-8'))
            self.assertEqual(record['returncode'], 7)
            self.assertEqual(len(record['jobDrainObservations']), 21)
            self.assertIn('survived', record['jobDrainAnomaly'])
            self.assertEqual(record['ownedMemberIdentities']['processes'][0]['status'], 'open-unavailable')
            self.assertEqual(record['remainingOwnedPids'], [])
            self.assertTrue(record['jobCloseSucceeded'])
            self.assertNotIn('error', record)

    def test_success_timeout_membership_cleanup_and_close_failures_remain_terminal(self):
        for mode in ('success', 'timeout', 'query-error', 'cleanup-error', 'cleanup-survivor', 'close-error'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_case(directory, mode)
                record = json.loads((directory/'edit-admission-005-collector.owned.json').read_text(encoding='utf-8'))
                if mode == 'close-error':
                    self.assertFalse(record['jobCloseSucceeded'])
                    self.assertIn('CloseHandle', record['jobCloseError'])
                    self.assertEqual(record['remainingOwnedPids'], [])
                elif mode.startswith('cleanup-'):
                    self.assertIn('cleanupError', record)
                else:
                    self.assertIn('error', record)

    def test_refresh_numerical_and_unmarked_collectors_cannot_use_finalization(self):
        for label in ('edit-admission-005-refresh', 'edit-tests', 'compile', 'edit-admission-005-collector-extra'):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'admission-collector-only'):
                    self.run_case(Path(tmp), 'nonzero', label)
        for label in ('edit-admission-005-refresh', 'edit-tests', 'edit-admission-005-collector'):
            with self.subTest(strict_label=label), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'owned (Job drain deadline exceeded|descendants survived root exit drain)'):
                    self.run_case(Path(tmp), 'nonzero', label, policy=False)

    def test_cleaned_failed_collector_queues_once_before_fresh_admission(self):
        fixture = OrchestrationTests('test_two_binaries_all_compilation_before_three_exact_invocations')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sleeps = []
        def runner(args, env, out, label, seconds, **policy):
            if label == 'edit-admission-001-collector':
                fixture.calls.append((label, args, env.copy()))
                return self.run_case(out, 'nonzero', label)[0]
            return fixture.runner(args, env, out, label, seconds, **policy)
        result = gate.execute(fixture.spec, fixture.env, fixture.out, runner=runner, now=lambda: NOW,
                              sleep=sleeps.append, source_check=lambda: {'fixed': 'source'})
        self.assertEqual(sleeps, [30])
        self.assertEqual(len(result), 3)
        self.assertEqual([label for label, _, _ in fixture.calls if label.endswith('-tests')],
                         ['edit-tests', 'layout-tests', 'encoder-tests'])
        first = json.loads((fixture.out/'edit-admission-001/admission.json').read_text(encoding='utf-8'))
        second = json.loads((fixture.out/'edit-admission-002/admission.json').read_text(encoding='utf-8'))
        self.assertEqual(first['refusal'], 'fresh collector failed')
        self.assertFalse(first['admitted']); self.assertTrue(second['admitted'])
        self.assertTrue((fixture.out/'edit-admission-001-collector.owned.json').exists())


class WindowsJobMetadataTests(unittest.TestCase):
    def test_metadata_never_queries_reused_or_unproven_handles(self):
        from scripts import media_campaign_windows as windows
        for mode in ('resolved', 'left-job', 'open-failed', 'reused', 'membership-unavailable', 'metadata-unavailable', 'close-failed'):
            with self.subTest(mode=mode):
                events = []
                class Kernel:
                    def OpenProcess(self, access, inherit, pid):
                        events.append('open'); self_case.assertEqual(access, 0x1000)
                        return 0 if mode == 'open-failed' else 55
                    def IsProcessInJob(self, process, job, owned):
                        events.append('membership'); owned._obj.value = mode != 'reused'
                        return mode != 'membership-unavailable'
                    def GetProcessTimes(self, process, created, *unused):
                        events.append('times'); created._obj.dwHighDateTime = 2; created._obj.dwLowDateTime = 3
                        return mode != 'metadata-unavailable'
                    def QueryFullProcessImageNameW(self, process, flags, buffer, size):
                        events.append('path'); buffer.value = 'C:/owned/helper.exe'
                        return mode != 'metadata-unavailable'
                    def CloseHandle(self, process):
                        events.append('close'); return mode != 'close-failed'
                self_case = self
                job = windows.WindowsJob.__new__(windows.WindowsJob)
                job.handle = 99; job.kernel = Kernel()
                job.members = lambda: set() if mode == 'left-job' else {123}
                with patch.object(windows, '_failed', side_effect=lambda op: windows.WindowsJobError(op)):
                    if mode == 'close-failed':
                        with self.assertRaisesRegex(windows.WindowsJobError, 'owned-identity'):
                            job.member_identities([123])
                        continue
                    result = job.member_identities([123])
                row = result['processes'][0]
                if mode == 'resolved':
                    self.assertEqual(row['createdFileTime'], (2 << 32) | 3)
                    self.assertEqual(row['path'], 'C:/owned/helper.exe')
                    self.assertEqual(row['status'], 'owned-identity-resolved')
                    self.assertEqual(events, ['open', 'membership', 'times', 'path', 'close'])
                elif mode == 'metadata-unavailable':
                    self.assertEqual(row['status'], 'owned-identity-partial')
                    self.assertIsNone(row['path']); self.assertIsNone(row['createdFileTime'])
                else:
                    self.assertNotIn('times', events); self.assertNotIn('path', events)
                    self.assertIsNone(row['path']); self.assertIsNone(row['createdFileTime'])
                if 'membership' in events: self.assertEqual(events[-1], 'close')

    def test_metadata_is_bounded_and_close_failure_retains_handle(self):
        from scripts import media_campaign_windows as windows
        job = windows.WindowsJob.__new__(windows.WindowsJob)
        job.handle = 99
        job.members = lambda: set()
        result = job.member_identities(list(range(100)))
        self.assertEqual(len(result['processes']), 32)
        self.assertTrue(result['truncated']); self.assertEqual(result['observedCount'], 100)
        calls = []
        class Kernel:
            success = False
            def CloseHandle(self, handle): calls.append(handle); return self.success
        job.kernel = Kernel()
        with patch.object(windows, '_failed', side_effect=lambda op: windows.WindowsJobError(op)):
            with self.assertRaisesRegex(windows.WindowsJobError, 'CloseHandle/job'): job.close()
        self.assertEqual(job.handle, 99)
        job.kernel.success = True
        job.close(); self.assertIsNone(job.handle)
        job.close(); self.assertEqual(calls, [99, 99])


class WorkflowTests(unittest.TestCase):
    def test_dedicated_manual_graph_and_owned_target_preserve_device_boundary(self):
        workflow = yaml.safe_load((gate.ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
        job = workflow['jobs']['moss-edit-cuda']
        self.assertIn("github.event_name == 'workflow_dispatch'", job['if'])
        self.assertIn("inputs.lanes == 'moss-edit-cuda'", job['if'])
        self.assertIn('github.run_attempt == 1', job['if']); self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(job['env']['CUDA_COMPUTE_CAP'], '120'); self.assertEqual(job['env']['CARGO_BUILD_JOBS'], '12')
        target = next(step for step in job['steps'] if step.get('name', '').startswith('Reserve an exclusive'))
        self.assertIn('$env:RUNNER_TEMP', target['run']); self.assertIn('$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT', target['run'])
        self.assertIn('Target directory already exists', target['run'])
        self.assertFalse(job['concurrency']['cancel-in-progress'])
        commands = '\n'.join(step.get('run', '') for step in job['steps'])
        self.assertNotIn('uv python install', commands); self.assertNotIn('pip install', commands)
        self.assertIn('-m scripts.ci.moss_edit_cuda', commands)
        run = next(step for step in job['steps'] if 'scripts.ci.moss_edit_cuda' in step.get('run', ''))
        self.assertEqual(run['env']['MOSS_CUDA_ADMISSION'], '${{ inputs.moss_cuda_admission }}')
        self.assertNotIn('${{ inputs.moss_cuda_admission }}', commands)

    def test_new_selector_isolates_job_with_explicit_sibling_guards(self):
        from scripts.tests.test_windows_gpu1_diagnostic import selected_jobs
        workflow = yaml.safe_load((gate.ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
        for selected in (True, False):
            for fork in (True, False):
                for tag in (True, False):
                    self.assertEqual(selected_jobs(workflow, 'workflow_dispatch', 'moss-edit-cuda', selected=selected, fork=fork, tag=tag), {'moss-edit-cuda'})
        for event in ('push', 'pull_request', 'schedule', 'merge_group'):
            self.assertNotIn('moss-edit-cuda', selected_jobs(workflow, event, 'moss-edit-cuda'))
        for lane in ('all', 'windows-cuda', 'windows-gpu-diagnostic', 'moss-edit-cuda-extra'):
            self.assertNotIn('moss-edit-cuda', selected_jobs(workflow, 'workflow_dispatch', lane))
        guard = "(github.event_name != 'workflow_dispatch' || inputs.lanes != 'moss-edit-cuda')"
        for name, job in workflow['jobs'].items():
            if name in ('moss-edit-cuda', 'windows-gpu-diagnostic'):
                continue
            self.assertIn(guard, job['if'], name)
        mutant = copy.deepcopy(workflow)
        mutant['jobs']['moss-edit-cuda']['if'] = mutant['jobs']['moss-edit-cuda']['if'].replace("inputs.lanes == 'moss-edit-cuda'", 'True')
        self.assertIn('moss-edit-cuda', selected_jobs(mutant, 'workflow_dispatch', 'windows-gpu-diagnostic'))

    def test_final_snapshot_is_not_full_telemetry_and_uses_existing_queries(self):
        script = (gate.ROOT/'scripts/ci/windows-gpu1-diagnostic.ps1').read_text(encoding='utf-8')
        branch = script.split('if ($PreviousEvidencePath) {', 1)[1].split("try {\n    $identity = Identity", 1)[0]
        for token in ("'identity-census-only'", 'Identity \'1\'', 'Census $identity.uuid.Trim()', 'Process-Identity $ids', "'finalSnapshotComplete'", '$previous.telemetryComplete', '$previous.samples.Count -ne 3'):
            self.assertIn(token, branch)
        self.assertNotIn('$record.telemetryComplete = $true', branch)
        self.assertNotIn('Get-Counter', branch)


if __name__ == '__main__':
    unittest.main()
