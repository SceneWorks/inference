"""Pure ownership and workflow mutations; never query or allocate a GPU."""
import copy
import datetime as dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from scripts.ci import moss_edit_metal as gate
from scripts.tests.test_windows_gpu1_diagnostic import selected_jobs


def snapshot():
    owner = dict(pid=10, parent=1, group=10, birth='runner-birth', comm='/runner/Runner.Worker')
    pipeline = dict(pid=50, parent=1, group=50, birth='pipeline-birth', comm='/python3')
    return dict(controllerJobOwner=owner, processes=[owner, pipeline], utc='2026-10-23T12:00:00Z', host='reviewed-mac', runner='nax-macos',
                device={'IORegistryEntryID': 1, 'IORegistryEntryName': 'AGX', 'IOObjectClass': 'AGXAccelerator'},
                clients=[dict(entry_id=11, pid=21, birth='fixed-birth', comm='/WindowServer', owner='pid 21, WindowServer', usage_present=True, usage={'count': 2})],
                listeners=[dict(pid=31, birth='other-birth', comm='/reviewed-listener', endpoints=['127.0.0.1:123'])])


def lease():
    old = snapshot()
    return dict(schemaVersion=1, source='a' * 40, actor=gate.REVIEWER, preparationRunId=42, preparationAttempt=1,
                host=old['host'], runner=old['runner'], device=old['device'], binarySha256='b' * 64,
                metadataSha256='c' * 64, sourceInventorySha256='d' * 64, classificationUtc=old['utc'],
                validFromUtc='2026-10-23T12:10:00Z', expiresUtc='2026-10-23T13:10:00Z',
                exclusiveMetalExecutionAuthorized=True,
                clients=[dict(entry_id=11, mustBeAbsent=False, allowBackgroundActivity=False)],
                listeners=[dict(pid=31, mustBeAbsent=False, disposition='explicitly-reviewed-no-compute')],
                processes=[dict(pid=10, disposition='reviewed-no-compute'), dict(pid=50, disposition='reviewed-no-compute')],
                runnerNamesOnPhysicalHost=['nax-macos', 'nax-macos-2'], computeJobsToDrain=[])


class OwnershipTests(unittest.TestCase):
    def test_exact_reviewed_owner_and_explicit_background_activity(self):
        spec, old, current = lease(), snapshot(), snapshot()
        gate.validate_lease(spec); gate.ownership(spec, old, current)
        current['clients'][0]['usage']['count'] += 1
        with self.assertRaises(ValueError): gate.ownership(spec, old, current)
        spec['clients'][0]['allowBackgroundActivity'] = True
        gate.ownership(spec, old, current)

    def test_each_unreviewed_owner_device_incarnation_and_listener_refuses(self):
        mutations = [
            lambda c: c.update(host='other'), lambda c: c.update(runner='nax-macos-2'),
            lambda c: c['device'].update(IORegistryEntryID=2),
            lambda c: c['clients'][0].update(entry_id=12), lambda c: c['clients'][0].update(pid=22),
            lambda c: c['clients'][0].update(birth='reused'), lambda c: c['clients'][0].update(comm='/foreign'),
            lambda c: c['clients'][0].update(owner='pid 21, foreign'),
            lambda c: c['clients'][0].update(usage_present=False),
            lambda c: c['listeners'][0].update(pid=32), lambda c: c['listeners'][0].update(birth='reused'),
            lambda c: c['listeners'][0].update(comm='/foreign'), lambda c: c['listeners'][0].update(endpoints=['*:456']),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                current = snapshot(); mutation(current)
                with self.assertRaises(ValueError): gate.ownership(lease(), snapshot(), current)

    def test_busy_is_queued_and_only_matching_owned_birth_bypasses_classification(self):
        for category in ('clients', 'listeners'):
            spec, old, current = lease(), snapshot(), snapshot()
            spec[category][0]['mustBeAbsent'] = True
            with self.assertRaises(ValueError): gate.ownership(spec, old, current)
            current[category] = []
            gate.ownership(spec, old, current)
        current = snapshot(); current['clients'].append(dict(current['clients'][0], entry_id=99, pid=100, birth='owned-birth'))
        with self.assertRaises(ValueError): gate.ownership(lease(), snapshot(), current, {'100': {'birth': 'reused-birth'}})
        gate.ownership(lease(), snapshot(), current, {'100': {'birth': 'owned-birth'}})

    def test_schema_duplicate_missing_review_and_wildcards_refuse(self):
        mutations = [lambda s: s.update(actor='foreign'), lambda s: s.update(exclusiveMetalExecutionAuthorized=False),
                     lambda s: s.update(source='main'), lambda s: s.update(binarySha256=''),
                     lambda s: s.update(expiresUtc='2026-10-23T12:11:00Z'),
                     lambda s: s.update(classificationUtc='2026-10-24T12:00:00Z'),
                     lambda s: s['clients'].append(copy.deepcopy(s['clients'][0])),
                     lambda s: s['clients'][0].update(allowBackgroundActivity='yes'),
                     lambda s: s['clients'][0].update(wildcard=True),
                     lambda s: s['listeners'][0].update(disposition='idle'),
                     lambda s: s['computeJobsToDrain'].append(dict(repository='foreign/repo', runId=1, attempt=1))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                spec = lease(); mutate(spec)
                with self.assertRaises(ValueError): gate.validate_lease(spec)
        for category in ('clients', 'listeners'):
            spec = lease(); spec[category] = []
            with self.assertRaises(ValueError): gate.ownership(spec, snapshot(), snapshot())
        with self.assertRaises(ValueError): json.loads('{"source":1,"source":2}', object_pairs_hook=gate.unique_fields)

    def test_missing_ioreg_creator_properties_fail_closed(self):
        import plistlib
        node = dict(IORegistryEntryID=1, IORegistryEntryName='AGX', IOObjectClass='AGXAccelerator',
                    IORegistryEntryChildren=[dict(IORegistryEntryID=11, IORegistryEntryName='AGXDeviceUserClient', IOObjectClass='AGXDeviceUserClient')])
        with patch.object(gate.subprocess, 'check_output', return_value=plistlib.dumps([node])) as query, patch.object(gate, 'rows', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'lacks owner properties'): gate.snapshot()
        self.assertEqual(query.call_args.args[0], ['/usr/sbin/ioreg', '-a', '-l', '-r', '-c', 'AGXAccelerator'])


class WorkflowTests(unittest.TestCase):
    def test_manual_operations_are_isolated_and_do_not_deadlock_publisher(self):
        workflow = yaml.safe_load((gate.ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
        for lane in ('moss-edit-metal', 'moss-edit-metal-lease'):
            for selected in (False, True):
                for fork in (False, True):
                    self.assertEqual(selected_jobs(workflow, 'workflow_dispatch', lane, selected=selected, fork=fork), {lane})
            for event in ('push', 'pull_request', 'merge_group', 'schedule'):
                self.assertNotIn(lane, selected_jobs(workflow, event, lane))
            for other in ('all', 'windows-cuda', 'windows-gpu-diagnostic', 'moss-edit-cuda'):
                self.assertNotIn(lane, selected_jobs(workflow, 'workflow_dispatch', other))
        self.assertIn("format('metal-{0}', github.run_id)", workflow['concurrency']['group'])
        self.assertEqual(workflow['concurrency']['cancel-in-progress'], "${{ github.event_name == 'pull_request' }}")
        prepare, publisher = (workflow['jobs'][name] for name in ('moss-edit-metal', 'moss-edit-metal-lease'))
        self.assertEqual(prepare['runs-on'], ['self-hosted', 'macOS', 'ARM64', 'nax'])
        self.assertEqual(publisher['runs-on'], 'ubuntu-latest')
        self.assertNotIn('needs', publisher)
        commands = '\n'.join(step.get('run', '') for step in prepare['steps'])
        self.assertEqual(commands.strip(), 'python3 scripts/ci/moss_edit_metal.py prepare\n\npython3 scripts/ci/moss_edit_metal.py execute')
        self.assertNotIn('${{ inputs.', commands)
        uploads = [step for step in prepare['steps'] if step.get('uses', '').startswith('actions/upload-artifact@')]
        self.assertEqual(len(uploads), 2)
        mutant = copy.deepcopy(workflow)
        mutant['jobs']['moss-edit-metal']['if'] = mutant['jobs']['moss-edit-metal']['if'].replace("inputs.lanes == 'moss-edit-metal'", 'True')
        self.assertIn('moss-edit-metal', selected_jobs(mutant, 'workflow_dispatch', 'all'))


class ExecutionBoundaryTests(unittest.TestCase):
    def test_current_nonlistener_pipeline_drains_and_reincarnation_does_not_inherit(self):
        spec, old, current = lease(), snapshot(), snapshot()
        spec['processes'][1]['disposition'] = 'must-drain'
        with self.assertRaisesRegex(ValueError, 'must drain'): gate.ownership(spec, old, current)
        current['processes'] = current['processes'][:1]
        gate.ownership(spec, old, current)
        current = snapshot(); current['processes'][1]['birth'] = 'reused'
        spec['processes'][1]['disposition'] = 'reviewed-no-compute'
        with self.assertRaisesRegex(ValueError, 'new or changed'): gate.ownership(spec, old, current)
        current = snapshot(); current['processes'].append(dict(pid=51, parent=1, group=51, birth='new', comm='/sceneworks_worker-test'))
        with self.assertRaisesRegex(ValueError, 'new or changed'): gate.ownership(spec, old, current)

    def test_reconciliation_includes_nonlistener_workers_and_pipeline_children(self):
        records = [dict(pid=1, parent=0, group=1, birth='a', comm='/launchd'),
                   dict(pid=2, parent=1, group=2, birth='b', comm='/cargo'),
                   dict(pid=3, parent=2, group=2, birth='c', comm='/target/debug/deps/unfamiliar-test-name'),
                   dict(pid=4, parent=1, group=4, birth='d', comm='/sceneworks_worker'),
                   dict(pid=5, parent=1, group=5, birth='e', comm='/WindowServer')]
        self.assertEqual({r['pid'] for r in gate.relevant_processes(records)}, {2, 3, 4})
        self.assertEqual(set(records[0]), {'pid', 'parent', 'group', 'birth', 'comm'})

    def prepared(self, folder):
        binary = folder / 'binary'; binary.write_bytes(b'inert')
        (folder / 'runtime.json').write_text('{}', encoding='utf-8')
        return dict(source='a' * 40, actor=gate.REVIEWER, runId=42, attempt=1, host='reviewed-mac',
                    binary=str(binary), binarySha256=gate.sha(binary), controllerSha256='e' * 64,
                    tokenIdentity={'path': '/shared/lock', 'device': 1, 'inode': 2}, members=[gate.TEST])

    def test_two_authenticated_waiters_share_one_token_but_foreign_compute_blocks(self):
        if os.name != 'posix': self.skipTest('physical flock is a macOS/POSIX protocol')
        import fcntl
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); prepared = self.prepared(out)
            other_owner = dict(pid=70, parent=1, group=70, birth='other-runner', comm='/runner/Runner.Worker')
            other = dict(prepared, runId=43, jobOwner=other_owner)
            run = dict(id=43, event='workflow_dispatch', run_attempt=1, head_sha=prepared['source'], actor={'login': gate.REVIEWER})
            job = dict(id=430, status='in_progress', runner_name='nax-macos-2', name='moss-edit-metal',
                       steps=[dict(name='Wait for the reviewed lease, recheck physical ownership, and run one exact oracle', status='in_progress')])
            cache = {'waiters': {43: other}, 'not_publishers': set()}
            def api(path):
                if path.startswith('repos/SceneWorks/SceneWorks/'): return {'total_count': 0, 'workflow_runs': []}
                if '/runs?' in path: return {'total_count': 1, 'workflow_runs': [run]}
                return {'total_count': 1, 'jobs': [job]}
            with patch.object(gate, 'api', side_effect=api):
                blocked, authenticated = gate.competing_jobs(lease(), prepared, out, cache)
                self.assertEqual(blocked, []); self.assertEqual(authenticated, [other_owner])
                current = snapshot(); current['processes'].extend([other_owner, dict(pid=71, parent=70, group=70, birth='waiter', comm='/python3')])
                gate.ownership(lease(), snapshot(), current, authenticated_jobs=authenticated)
                job['name'] = 'unrelated-NAX-compute'
                self.assertEqual(len(gate.competing_jobs(lease(), prepared, out, cache)[0]), 1)
                job['name'] = 'moss-edit-metal'; other['controllerSha256'] = 'f' * 64
                self.assertEqual(len(gate.competing_jobs(lease(), prepared, out, cache)[0]), 1)
            token = out / 'physical.lock'
            launched = []
            with token.open('a', encoding='utf-8') as first, token.open('a', encoding='utf-8') as second:
                self.assertTrue(gate.try_execution_token(first)); launched.append(42)
                self.assertFalse(gate.try_execution_token(second))
                self.assertEqual(launched, [42])
                gate.release_execution_token(first)
                self.assertTrue(gate.try_execution_token(second)); launched.append(43)
                self.assertEqual(launched, [42, 43])

    def test_mutation_during_final_capacity_or_inventory_never_reaches_popen(self):
        for stage in ('capacity', 'inventory'):
            for mutation in ('client', 'job', 'pipeline'):
                with self.subTest(stage=stage, mutation=mutation), tempfile.TemporaryDirectory() as temp:
                    out = Path(temp); prepared = self.prepared(out); current = snapshot(); active_jobs = []
                    def mutate(*args, **kwargs):
                        if mutation == 'client': current['clients'][0]['entry_id'] = 99
                        elif mutation == 'job': active_jobs.append({'jobId': 99})
                        else: current['processes'][1]['birth'] = 'changed'
                        return {}
                    with patch.object(gate, 'capacity', side_effect=mutate if stage == 'capacity' else None), \
                         patch.object(gate, 'inventory', side_effect=mutate if stage == 'inventory' else lambda: {}), \
                         patch.object(gate, 'runtime_identity', return_value={}), patch.object(gate, 'child_env', return_value={}), \
                         patch.object(gate, 'competing_jobs', side_effect=lambda *args: (active_jobs, [])), \
                         patch.object(gate, 'snapshot', return_value=current), patch.object(gate, 'utc', return_value=dt.datetime(2026, 10, 23, 12, 15, tzinfo=dt.timezone.utc)), patch.object(gate.subprocess, 'Popen') as launch:
                        admit = lambda: gate.final_admission(lease(), prepared, snapshot(), {}, Path(prepared['binary']), out, {})
                        with self.assertRaises(gate.AdmissionDeferred): gate.run_owned(['inert'], {}, out, 'oracle', 1, prelaunch=admit)
                        launch.assert_not_called()

    def test_complete_final_admission_positive_control(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); prepared = self.prepared(out)
            with patch.object(gate, 'capacity'), patch.object(gate, 'inventory', return_value={}), \
                 patch.object(gate, 'runtime_identity', return_value={}), patch.object(gate, 'child_env', return_value={}), \
                 patch.object(gate, 'competing_jobs', return_value=([], [])), patch.object(gate, 'snapshot', return_value=snapshot()), \
                 patch.object(gate, 'utc', return_value=dt.datetime(2026, 10, 23, 12, 15, tzinfo=dt.timezone.utc)):
                admission = gate.final_admission(lease(), prepared, snapshot(), {}, Path(prepared['binary']), out, {})
            self.assertEqual(admission['competingJobs'], [])
            self.assertEqual(admission['freshSnapshot'], snapshot())

    def test_fast_child_cannot_bypass_prelaunch_admission(self):
        class Finished:
            pid = 999
            def poll(self): return 0
            def wait(self): return 0
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); order = []
            def launch(*args, **kwargs):
                order.append('popen'); kwargs['stdout'].write('inert output'); kwargs['stdout'].flush(); return Finished()
            def admit(): order.append('admission'); return {'fresh': True}
            with patch.object(gate.subprocess, 'Popen', side_effect=launch), patch.object(gate, 'rows', return_value=[]):
                gate.run_owned(['inert'], {}, out, 'fast', 1, prelaunch=admit)
            self.assertEqual(order, ['admission', 'popen'])
            receipt = json.loads((out / 'fast-process.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['prelaunch'], {'fresh': True}); self.assertEqual(receipt['remaining'], [])


class PreparationCapacityTests(unittest.TestCase):
    def test_measured_capacity_is_distinct_and_telemetry_errors_are_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(gate.os, 'cpu_count', return_value=18), patch.object(gate.os, 'getloadavg', return_value=(6, 6, 6)), \
                 patch.object(gate, 'rows', return_value=[]), patch.object(gate, 'metadata', return_value='System-wide memory free percentage: 29%'), \
                 patch.object(gate.os, 'statvfs') as disk:
                disk.return_value.f_bavail = 100 * 1024**3; disk.return_value.f_frsize = 1
                with self.assertRaises(gate.CapacityUnavailable) as caught: gate.capacity(out, prelaunch=True)
                self.assertEqual(caught.exception.sample['memoryFreePercent'], 29)
                self.assertIn('memory', caught.exception.predicate)
                disk.return_value.f_bavail = 1
                with patch.object(gate, 'metadata', return_value='System-wide memory free percentage: 30%'):
                    with self.assertRaises(gate.CapacityUnavailable) as caught: gate.capacity(out)
                    self.assertIn('disk', caught.exception.predicate)
                with patch.object(gate, 'metadata', return_value='not a capacity sample'):
                    with self.assertRaises(ValueError) as caught: gate.capacity(out)
                    self.assertNotIsInstance(caught.exception, gate.CapacityUnavailable)
            self.assertEqual(len((out / 'capacity-history.log').read_text(encoding='utf-8').splitlines()), 2)

    def context(self, out, samples, *, deadline=1200):
        from contextlib import ExitStack
        stack = self.enterContext(ExitStack())
        target = out / 'target'; target.mkdir()
        env = {'CARGO_TARGET_DIR': str(target)}; clock = [0.0]; launches = []; signals = []; processes = []
        sample_iter = iter(samples)
        def capacity(*args, **kwargs):
            value = next(sample_iter, 30)
            if value < 30: raise gate.CapacityUnavailable('memory', {'memoryFreePercent': value})
        class Process:
            def __init__(self, pid, code): self.pid, self.code = pid, code
            def poll(self): return self.code
            def wait(self, **_kwargs): return self.code
        def popen(command, **kwargs):
            if launches: self.assertEqual((target / 'partial').read_bytes(), b'preserved')
            (target / 'partial').write_bytes(b'preserved')
            process = Process(900 + len(launches), None if not launches else 0)
            processes.append(process); launches.append((command, kwargs['env']['CARGO_TARGET_DIR'], clock[0]))
            kwargs['stdout'].write('attempt ' + str(len(launches))); kwargs['stdout'].flush()
            return process
        def rows(*_args):
            return [(p.pid, 1, p.pid, 'owned-birth', '/cargo') for p in processes if p.code is None] + [(777, 1, 777, 'foreign-birth', '/foreign')]
        def kill(pid, sig):
            signals.append((pid, sig)); self.assertNotEqual(pid, 777)
            next(p for p in processes if p.pid == pid).code = -sig
        stack.enter_context(patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]))
        stack.enter_context(patch.object(gate.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
        stack.enter_context(patch.object(gate, 'capacity', side_effect=capacity))
        stack.enter_context(patch.object(gate, 'inventory', return_value={'source': 'fixed'}))
        stack.enter_context(patch.object(gate, 'runtime_identity', return_value={'rustc': 'fixed'}))
        stack.enter_context(patch.object(gate.subprocess, 'Popen', side_effect=popen))
        stack.enter_context(patch.object(gate, 'rows', side_effect=rows))
        stack.enter_context(patch.object(gate.os, 'kill', side_effect=kill))
        return env, clock, launches, signals, processes

    def test_initial_wait_then_owned_interruption_reuses_partial_target_and_preserves_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            env, clock, launches, signals, processes = self.context(out, [29, 30, 30, 30, 29, 30, 30, 30])
            log = gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200)
            self.assertEqual(log, 'attempt 2')
            self.assertEqual(len(launches), 2)
            self.assertEqual(launches[0][:2], launches[1][:2])
            self.assertGreaterEqual(launches[0][2], 60)
            self.assertEqual(signals, [(900, gate.signal.SIGTERM)])
            first = gate.read(out / 'compile-001-process.json'); second = gate.read(out / 'compile-002-process.json')
            self.assertTrue(first['capacityInterrupted']); self.assertEqual(first['remaining'], [])
            self.assertEqual(first['exitCode'], -gate.signal.SIGTERM); self.assertEqual(second['exitCode'], 0)
            self.assertEqual((out / 'compile-001.log').read_text(encoding='utf-8'), 'attempt 1')
            states = [row['status'] for row in gate.read(out / 'preparation-queue.json')['events']]
            self.assertIn('owned-drained', states); self.assertIn('retrying', states)
            self.assertLess(states.index('owned-draining'), states.index('owned-drained'))
            self.assertEqual(first['sentSignals'], [{'pid': 900, 'birth': 'owned-birth', 'signal': int(gate.signal.SIGTERM)}])
            self.assertEqual(states[-1], 'compiled')
            self.assertFalse((out / 'prepared.json').exists())

    def test_persistent_or_oscillating_shortage_consumes_one_deadline_without_launch(self):
        for samples in ([29] * 100, [29, 30] * 50):
            with self.subTest(samples=samples[:2]), tempfile.TemporaryDirectory() as temp:
                out = Path(temp); env, clock, launches, *_ = self.context(out, samples)
                with self.assertRaisesRegex(ValueError, 'deadline expired'):
                    gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 190)
                self.assertEqual(launches, []); self.assertEqual(clock[0], 120)
                self.assertEqual(gate.read(out / 'preparation-queue.json')['events'][-1]['status'], 'deadline-expired')

    def test_compiler_cleanup_source_runtime_and_target_failures_never_retry(self):
        for failure in ('compiler', 'cleanup', 'source', 'runtime', 'target', 'telemetry'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                out = Path(temp); env, clock, launches, *_ = self.context(out, [30] * 20)
                calls = []
                def run(*args, **kwargs):
                    calls.append(args[3])
                    if failure == 'compiler': raise ValueError('compiler failed')
                    if failure == 'cleanup':
                        gate.write(out / (args[3] + '-process.json'), {'capacityUnavailable': {'predicate': 'memory'}, 'remaining': [[900, 'owned-birth']], 'capacityInterrupted': True, 'exitCode': -15})
                        raise gate.CapacityUnavailable('memory', {})
                    return 'compiled'
                with patch.object(gate, 'run_owned', side_effect=run):
                    if failure == 'source': self.enterContext(patch.object(gate, 'inventory', side_effect=[{'source': 'fixed'}, {'source': 'changed'}]))
                    if failure == 'runtime': self.enterContext(patch.object(gate, 'runtime_identity', side_effect=[{'rustc': 'fixed'}, {'rustc': 'changed'}]))
                    if failure == 'target': (out / 'target').rmdir()
                    if failure == 'telemetry': self.enterContext(patch.object(gate, 'capacity', side_effect=ValueError('unparseable memory')))
                    with self.assertRaises(ValueError): gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200)
                self.assertLessEqual(len(calls), 1)
                self.assertEqual(launches, [])


    def test_source_or_runtime_change_after_drained_interruption_prevents_second_launch(self):
        for identity in ('inventory', 'runtime_identity'):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temp:
                out = Path(temp); env, clock, launches, *_ = self.context(out, [30, 30, 30, 29])
                key = 'source' if identity == 'inventory' else 'rustc'
                with patch.object(gate, identity, side_effect=[{key: 'fixed'}] * 3 + [{key: 'changed'}]):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200)
                self.assertEqual(len(launches), 1)
                self.assertEqual(gate.read(out / 'compile-001-process.json')['remaining'], [])

    def test_final_prelaunch_capacity_refusal_records_no_child(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(gate.subprocess, 'Popen') as launch:
                with self.assertRaises(gate.CapacityUnavailable):
                    gate.run_owned(['inert'], {}, out, 'refused', 10, prelaunch=lambda: (_ for _ in ()).throw(gate.CapacityUnavailable('memory', {})))
                launch.assert_not_called()
            row = gate.read(out / 'refused-process.json')
            self.assertFalse(row['launched']); self.assertEqual(row['remaining'], []); self.assertIsNone(row['exitCode'])

    def test_listing_mismatch_or_exhausted_shared_deadline_cannot_publish_prepared(self):
        for failure in ('listing', 'deadline'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                out = Path(temp); clock = [0.0]
                def compiled(*args):
                    binary = out / 'target' / 'inert'; binary.write_bytes(b'inert')
                    if failure == 'deadline': clock[0] = 1201
                    return json.dumps({'reason': 'compiler-artifact', 'target': {'name': 'mlx_gen_moss_image'}, 'profile': {'test': True}, 'executable': str(binary)})
                with patch.object(gate.sys, 'platform', 'darwin'), patch.dict(os.environ, {'RUNNER_NAME': 'nax-macos'}), \
                     patch.object(gate, 'paths', return_value=out), patch.object(gate, 'inventory', return_value={}), \
                     patch.object(gate, 'runtime_identity', return_value={}), patch.object(gate, 'prepare_compile', side_effect=compiled), \
                     patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate, 'run_owned', return_value='wrong: test') as run, \
                     patch.object(gate, 'snapshot') as gpu_metadata:
                    with self.assertRaisesRegex(ValueError, 'membership' if failure == 'listing' else 'deadline'): gate.prepare()
                self.assertEqual(run.call_count, 1 if failure == 'listing' else 0)
                gpu_metadata.assert_not_called(); self.assertFalse((out / 'prepared.json').exists())


    def test_external_signal_between_capacity_poll_and_cleanup_is_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, clock, launches, signals, processes = self.context(out, [30, 30, 30, 29])
            original_capacity = gate.capacity.side_effect; original_launch = gate.subprocess.Popen.side_effect
            armed = [False]; polls = [0]
            def capacity(*args, **kwargs):
                try: return original_capacity(*args, **kwargs)
                except gate.CapacityUnavailable:
                    armed[0] = True
                    raise
            def launch(*args, **kwargs):
                process = original_launch(*args, **kwargs)
                def poll():
                    if armed[0]:
                        polls[0] += 1
                        if polls[0] >= 2: process.code = -gate.signal.SIGTERM
                    return process.code
                process.poll = poll
                return process
            with patch.object(gate, 'capacity', side_effect=capacity), patch.object(gate.subprocess, 'Popen', side_effect=launch):
                with self.assertRaisesRegex(ValueError, 'not a controller-initiated'):
                    gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200)
            self.assertEqual(len(launches), 1); self.assertEqual(signals, [])
            receipt = gate.read(out / 'compile-001-process.json')
            self.assertTrue(receipt['capacityInterrupted']); self.assertEqual(receipt['exitCode'], -gate.signal.SIGTERM)
            self.assertEqual(receipt['remaining'], []); self.assertEqual(receipt['sentSignals'], [])

    def test_slow_prelaunch_cannot_shift_absolute_work_deadline_or_launch_late(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); clock = [0.0]
            def prelaunch(): clock[0] = 91
            with patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'absolute preparation operation deadline'):
                    gate.run_owned(['inert'], {}, out, 'slow-prelaunch', 90, prelaunch=prelaunch, absolute_deadline=100)
                launch.assert_not_called()
            receipt = gate.read(out / 'slow-prelaunch-process.json')
            self.assertFalse(receipt['launched']); self.assertEqual(receipt['workDeadline'], 90)

    def test_slow_monitor_is_bounded_and_drains_owned_child_within_cleanup_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, clock, launches, signals, processes = self.context(out, [30])
            timeouts = []
            def metadata_run(*args, **kwargs):
                timeouts.append(kwargs['timeout'])
                clock[0] += kwargs['timeout']
                raise gate.subprocess.TimeoutExpired(args[0], kwargs['timeout'])
            def monitor(_owned):
                clock[0] = 88
                gate.metadata(['inert-telemetry'], deadline=90)
            with patch.object(gate.subprocess, 'run', side_effect=metadata_run):
                with self.assertRaises(gate.subprocess.TimeoutExpired):
                    gate.run_owned(['inert'], env, out, 'slow-monitor', 90, monitor=monitor, absolute_deadline=100)
            self.assertEqual(timeouts, [2]); self.assertEqual(clock[0], 90)
            self.assertEqual(signals, [(900, gate.signal.SIGTERM)])
            receipt = gate.read(out / 'slow-monitor-process.json')
            self.assertEqual(receipt['remaining'], []); self.assertEqual(receipt['workDeadline'], 90)
            self.assertNotIn('capacityUnavailable', receipt)

    def test_slow_observation_uses_remaining_work_budget_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, clock, launches, signals, processes = self.context(out, [30])
            original_rows = gate.rows.side_effect; observed_limits = []
            def rows(limit=None):
                observed_limits.append(limit)
                if len(observed_limits) == 1:
                    clock[0] = 90
                    raise gate.subprocess.TimeoutExpired('ps', 90)
                return original_rows()
            with patch.object(gate, 'rows', side_effect=rows):
                with self.assertRaises(gate.subprocess.TimeoutExpired):
                    gate.run_owned(['inert'], env, out, 'slow-observe', 90, absolute_deadline=100)
            self.assertEqual(observed_limits[0], 90)
            self.assertTrue(all(value == 100 for value in observed_limits[1:]))
            self.assertEqual(signals, [(900, gate.signal.SIGTERM)])
            self.assertEqual(gate.read(out / 'slow-observe-process.json')['remaining'], [])


class ScratchQuarantineTests(unittest.TestCase):
    name = 'pmetal-mlx-sys-0123456789abcdef'

    def fixture(self, out):
        target = out / 'target'; target.mkdir()
        generated = target / 'debug/build' / self.name / 'out'
        generated.mkdir(parents=True)
        (generated / 'stale-patch').write_bytes(b'preserved patch state')
        (target / 'rust-sentinel').write_bytes(b'unrelated Rust cache')
        return target, generated, (target.stat().st_dev, target.stat().st_ino)

    def test_rotation_retains_inode_bytes_and_rust_cache_across_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); target, generated, identity = self.fixture(out)
            old_inode = generated.stat().st_ino
            first = gate.quarantine_interrupted_pmetal_out(out, identity, 1, gate.time.monotonic() + 60)
            retained = out / first['moves'][0]['destination']
            self.assertEqual(retained.stat().st_ino, old_inode)
            self.assertEqual((retained / 'stale-patch').read_bytes(), b'preserved patch state')
            self.assertEqual(list(generated.iterdir()), [])
            self.assertEqual((target / 'rust-sentinel').read_bytes(), b'unrelated Rust cache')
            (generated / 'second').write_bytes(b'second interruption')
            second = gate.quarantine_interrupted_pmetal_out(out, identity, 2, gate.time.monotonic() + 60)
            self.assertNotEqual(first['moves'][0]['destination'], second['moves'][0]['destination'])
            self.assertEqual((out / second['moves'][0]['destination'] / 'second').read_bytes(), b'second interruption')
            self.assertEqual((target.stat().st_dev, target.stat().st_ino), identity)
            self.assertTrue(gate.read(out / 'compile-001-scratch-quarantine.json')['moves'][0]['recreated'])

    def test_actual_capacity_retry_rotates_after_drain_before_second_launch(self):
        fixture = PreparationCapacityTests('test_initial_wait_then_owned_interruption_reuses_partial_target_and_preserves_receipts')
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            env, clock, launches, signals, processes = fixture.context(out, [30, 30, 30, 29, 30, 30, 30])
            generated = out / 'target/debug/build' / self.name / 'out'
            original = gate.subprocess.Popen.side_effect
            def popen(*args, **kwargs):
                if not launches:
                    generated.mkdir(parents=True); (generated / 'stale-patch').write_bytes(b'interrupted')
                else:
                    self.assertEqual(processes[0].code, -gate.signal.SIGTERM)
                    self.assertEqual(list(generated.iterdir()), [])
                    self.assertEqual((out / 'scratch-quarantine/compile-001' / self.name / 'stale-patch').read_bytes(), b'interrupted')
                return original(*args, **kwargs)
            with patch.object(gate.subprocess, 'Popen', side_effect=popen):
                self.assertEqual(gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200), 'attempt 2')
            self.assertEqual(len(launches), 2)
            self.assertEqual(signals, [(900, gate.signal.SIGTERM)])
            states = [r['status'] for r in gate.read(out / 'preparation-queue.json')['events']]
            self.assertLess(states.index('owned-drained'), states.index('scratch-quarantined'))
            self.assertLess(states.index('scratch-quarantined'), states.index('retrying'))
            for n in (1, 2): self.assertTrue((out / f'compile-{n:03d}.log').exists())

    def test_missing_outputs_are_recorded_noop_and_prelaunch_never_rotates(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); target = out / 'target'; target.mkdir()
            result = gate.quarantine_interrupted_pmetal_out(out, (target.stat().st_dev, target.stat().st_ino), 1, gate.time.monotonic() + 60)
            self.assertEqual(result['status'], 'no-pmetal-output')
            self.assertFalse((out / 'scratch-quarantine').exists())
        fixture = PreparationCapacityTests('test_initial_wait_then_owned_interruption_reuses_partial_target_and_preserves_receipts')
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, *_ = fixture.context(out, [30] * 20)
            calls = []
            def run(*args, **kwargs):
                calls.append(args[3])
                if len(calls) == 1:
                    gate.write(out / (args[3] + '-process.json'), {'launched': False, 'exitCode': None, 'remaining': [], 'capacityUnavailable': {'predicate': 'memory'}})
                    raise gate.CapacityUnavailable('memory', {})
                return 'success'
            with patch.object(gate, 'run_owned', side_effect=run), patch.object(gate, 'quarantine_interrupted_pmetal_out', side_effect=AssertionError('prelaunch rotation forbidden')):
                self.assertEqual(gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200), 'success')
            self.assertEqual(len(calls), 2)

    def test_symlink_components_substitution_and_existing_destination_refuse(self):
        for mode in ('target', 'debug', 'build', 'package', 'out', 'quarantine', 'identity', 'existing', 'foreign-owner'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as external:
                out = Path(temp); target, generated, identity = self.fixture(out)
                foreign = Path(external); (foreign / 'sentinel').write_bytes(b'untouched')
                components = {'target': target, 'debug': target/'debug', 'build': target/'debug/build', 'package': generated.parent, 'out': generated}
                if mode in components:
                    component = components[mode]; component.rename(component.with_name(component.name + '-original')); component.symlink_to(foreign, target_is_directory=True)
                if mode == 'quarantine': (out/'scratch-quarantine').symlink_to(foreign, target_is_directory=True)
                if mode == 'identity': identity = (identity[0], identity[1] + 1)
                if mode == 'existing': (out/'scratch-quarantine/compile-001').mkdir(parents=True, mode=0o700)
                from contextlib import nullcontext
                owner = patch.object(gate.os, 'getuid', return_value=out.stat().st_uid + 1) if mode == 'foreign-owner' else nullcontext()
                with owner, self.assertRaises((ValueError, OSError)):
                    gate.quarantine_interrupted_pmetal_out(out, identity, 1, gate.time.monotonic() + 60)
                self.assertEqual((foreign / 'sentinel').read_bytes(), b'untouched')
                self.assertEqual(list(foreign.iterdir()), [foreign/'sentinel'])

    def test_changed_inode_rename_failure_and_deadline_preserve_failure_receipts(self):
        for mode in ('changed-inode', 'rename-failed', 'deadline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                out = Path(temp); target, generated, identity = self.fixture(out)
                original_stat = gate.os.stat; original_rename = gate.os.rename; changed = [False]
                def inspect(name, *args, **kwargs):
                    if mode == 'changed-inode' and name == 'out' and kwargs.get('dir_fd') is not None and not changed[0]:
                        # Open captures this original identity; substitute just before recheck.
                        result = original_stat(name, *args, **kwargs)
                        changed[0] = True
                        return result
                    if mode == 'changed-inode' and name == 'out' and changed[0] is True:
                        generated.rename(generated.with_name('out-original')); generated.mkdir(); changed[0] = 'replaced'
                    return original_stat(name, *args, **kwargs)
                def rename(*args, **kwargs): raise OSError('inert rename failure')
                clock = [0]
                def now():
                    clock[0] += 1
                    return clock[0] if mode == 'deadline' else 0
                with patch.object(gate.os, 'stat', side_effect=inspect), patch.object(gate.os, 'rename', side_effect=rename if mode == 'rename-failed' else original_rename), patch.object(gate.time, 'monotonic', side_effect=now):
                    with self.assertRaises((ValueError, OSError)):
                        gate.quarantine_interrupted_pmetal_out(out, identity, 1, 8 if mode == 'deadline' else 60)
                self.assertIn('error', gate.read(out / 'compile-001-scratch-quarantine.json'))
                self.assertFalse((out/'scratch-quarantine/compile-001'/self.name).exists())

    def test_parallelism_is_two_without_weakening_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            env = gate.child_env(Path(temp))
            self.assertEqual(env['CARGO_BUILD_JOBS'], '2')
            self.assertEqual(env['CMAKE_BUILD_PARALLEL_LEVEL'], '2')
            self.assertEqual(env['CARGO_NET_OFFLINE'], 'true')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')


class ScratchFreshnessTests(unittest.TestCase):
    name = ScratchQuarantineTests.name
    fingerprint = 'run-build-script-build-script-build'

    def completed(self, out):
        generated = out / 'target/debug/build' / self.name / 'out'
        generated.mkdir(parents=True, exist_ok=True)
        (generated/'bindings.rs').write_bytes(b'complete bindings')
        (generated/'native.a').write_bytes(b'complete native archive')
        (generated.parent/'output').write_bytes(b'cargo native search path')
        unit = out/'target/debug/.fingerprint'/self.name
        unit.mkdir(parents=True)
        fingerprint = unit/self.fingerprint
        fingerprint.write_bytes(b'0123456789abcdef')
        gate.write(unit/(self.fingerprint+'.json'), {'local': [{'RerunIfChanged': {'output': f'debug/build/{self.name}/output'}}]})
        (unit/'unrelated-rust-unit').write_bytes(b'leave unchanged')
        return generated, fingerprint

    def test_completed_pmetal_downstream_interruption_invalidates_only_affected_freshness(self):
        fixture = PreparationCapacityTests('test_initial_wait_then_owned_interruption_reuses_partial_target_and_preserves_receipts')
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, clock, launches, signals, processes = fixture.context(out, [30, 30, 30, 29, 30, 30, 30])
            original = gate.subprocess.Popen.side_effect; saved = {}; regenerated = []
            def popen(*args, **kwargs):
                if not launches:
                    generated, fingerprint = self.completed(out)
                    saved.update(generated=generated, fingerprint=fingerprint,
                                 metadata=fingerprint.with_name(self.fingerprint+'.json').read_bytes())
                else:
                    generated, fingerprint = saved['generated'], saved['fingerprint']
                    # Cargo1.96's source-backed rule: an absent fingerprint dirties
                    # the build-script unit. This is an inert model, not Cargo.
                    fresh = fingerprint.exists()
                    self.assertFalse(fresh)
                    self.assertEqual(list(generated.iterdir()), [])
                    self.assertEqual(fingerprint.with_name(self.fingerprint+'.json').read_bytes(), saved['metadata'])
                    self.assertEqual((fingerprint.parent/'unrelated-rust-unit').read_bytes(), b'leave unchanged')
                    self.assertEqual((generated.parent/'output').read_bytes(), b'cargo native search path')
                    retained = out/'scratch-quarantine/compile-001'/self.name
                    self.assertEqual((retained/'bindings.rs').read_bytes(), b'complete bindings')
                    self.assertEqual((retained/'native.a').read_bytes(), b'complete native archive')
                    if not fresh:
                        (generated/'bindings.rs').write_bytes(b'regenerated bindings')
                        (generated/'native.a').write_bytes(b'regenerated native archive')
                        fingerprint.write_bytes(b'fedcba9876543210'); regenerated.append(True)
                return original(*args, **kwargs)
            with patch.object(gate.subprocess, 'Popen', side_effect=popen):
                self.assertEqual(gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200), 'attempt 2')
            self.assertEqual(regenerated, [True]); self.assertEqual(len(launches), 2)
            receipt = gate.read(out/'compile-001-scratch-quarantine.json')
            detail = receipt['moves'][0]['buildFingerprint']
            self.assertEqual(detail['status'], 'invalidated')
            self.assertEqual((out/detail['destination']).read_bytes(), b'0123456789abcdef')
            self.assertEqual((saved['generated']/'bindings.rs').read_bytes(), b'regenerated bindings')

    def test_invalid_fingerprint_or_mapping_refuses_before_output_rotation(self):
        for mode in ('symlink', 'nonregular', 'foreign-owner', 'malformed', 'missing-json', 'wrong-json', 'oversized-json', 'substituted', 'destination', 'rename-failure', 'deadline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as foreign:
                out = Path(temp); (out/'target').mkdir(); generated, fingerprint = self.completed(out)
                target = out/'target'; identity = (target.stat().st_dev, target.stat().st_ino); old_inode = generated.stat().st_ino
                external = Path(foreign)/'sentinel'; external.write_bytes(b'external unchanged')
                metadata = fingerprint.with_name(self.fingerprint+'.json')
                if mode == 'symlink': fingerprint.unlink(); fingerprint.symlink_to(external)
                if mode == 'nonregular': fingerprint.unlink(); fingerprint.mkdir()
                if mode == 'malformed': fingerprint.write_bytes(b'not-a-valid-hash!')
                if mode == 'missing-json': metadata.unlink()
                if mode == 'wrong-json': gate.write(metadata, {'local': [{'RerunIfChanged': {'output': 'debug/build/other/output'}}]})
                if mode == 'oversized-json': metadata.write_bytes(b'x'*(gate.MAX_LOG+1))
                original_stat, original_rename = gate.os.stat, gate.os.rename
                observations = [0]; clock = [0]
                def inspect(name, *args, **kwargs):
                    if name == self.fingerprint and kwargs.get('dir_fd') is not None:
                        observations[0] += 1
                        if mode == 'substituted' and observations[0] == 3:
                            fingerprint.rename(fingerprint.with_name('retained-original')); fingerprint.write_bytes(b'1111111111111111')
                        if mode == 'foreign-owner':
                            fields = list(original_stat(name, *args, **kwargs)); fields[4] += 1
                            return gate.os.stat_result(fields)
                        if mode == 'deadline': clock[0] = 60
                    if mode == 'destination' and name == self.name+'.run-build-fingerprint':
                        fd = os.open(name, gate.os.O_WRONLY | gate.os.O_CREAT | gate.os.O_EXCL, 0o600, dir_fd=kwargs['dir_fd'])
                        gate.os.close(fd)
                    return original_stat(name, *args, **kwargs)
                def rename(source, *args, **kwargs):
                    if mode == 'rename-failure' and source == self.fingerprint: raise OSError('inert fingerprint rename failure')
                    return original_rename(source, *args, **kwargs)
                with patch.object(gate.os, 'stat', side_effect=inspect), patch.object(gate.os, 'rename', side_effect=rename), patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]):
                    with self.assertRaises((ValueError, OSError)):
                        gate.quarantine_interrupted_pmetal_out(out, identity, 1, 60)
                self.assertEqual(generated.stat().st_ino, old_inode)
                self.assertEqual((generated/'bindings.rs').read_bytes(), b'complete bindings')
                self.assertEqual(external.read_bytes(), b'external unchanged')
                self.assertIn('error', gate.read(out/'compile-001-scratch-quarantine.json'))

    def test_output_rotation_failure_after_invalidation_is_terminal_and_retains_proof(self):
        fixture = PreparationCapacityTests('test_initial_wait_then_owned_interruption_reuses_partial_target_and_preserves_receipts')
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); env, clock, launches, signals, processes = fixture.context(out, [30, 30, 30, 29, 30])
            original_popen, original_rename = gate.subprocess.Popen.side_effect, gate.os.rename
            saved = {}
            def popen(*args, **kwargs):
                saved['generated'], saved['fingerprint'] = self.completed(out)
                return original_popen(*args, **kwargs)
            def rename(source, *args, **kwargs):
                if source == 'out': raise OSError('inert output rotation failure')
                return original_rename(source, *args, **kwargs)
            with patch.object(gate.subprocess, 'Popen', side_effect=popen), patch.object(gate.os, 'rename', side_effect=rename):
                with self.assertRaisesRegex(OSError, 'output rotation failure'):
                    gate.prepare_compile(out, env, {'source': 'fixed'}, {'rustc': 'fixed'}, 1200)
            self.assertEqual(len(launches), 1)
            self.assertFalse(saved['fingerprint'].exists())
            self.assertEqual((saved['generated']/'bindings.rs').read_bytes(), b'complete bindings')
            record = gate.read(out/'compile-001-scratch-quarantine.json')
            self.assertEqual(record['moves'][0]['buildFingerprint']['status'], 'invalidated')
            self.assertFalse(record['moves'][0]['renamed'])
            self.assertEqual((out/record['moves'][0]['buildFingerprint']['destination']).read_bytes(), b'0123456789abcdef')


if __name__ == '__main__':
    unittest.main()
