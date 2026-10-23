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


if __name__ == '__main__':
    unittest.main()
