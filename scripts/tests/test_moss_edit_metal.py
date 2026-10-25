"""CPU-only tests for direct exact Metal execution; no device allocation."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import yaml
from scripts.ci import moss_edit_metal as gate
from scripts.tests.test_windows_gpu1_diagnostic import selected_jobs


class DirectExecutionTests(unittest.TestCase):
    def exercise(self, mode='success'):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()/'run'
            output = Path(directory)/'output'
            environment = {'GITHUB_SHA': 'a'*40, 'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '2', 'RUNNER_NAME': 'nax-test', 'GITHUB_OUTPUT': str(output)}
            calls = []; source = {'source': 'fixed'}; runtime = {'rustc': 'fixed'}
            def paths(create=False):
                if create: out.mkdir()
                return out
            def runner(command, env, dest, label, timeout):
                calls.append((label, command, env.copy()))
                self.assertEqual(env['CARGO_BUILD_JOBS'], '2')
                self.assertEqual(env['CMAKE_BUILD_PARALLEL_LEVEL'], '2')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                binary = out/'target'/'test-binary'
                if label == 'compile':
                    self.assertNotIn('MOSS_NATIVE_TEST_ADMITTED', env)
                    if mode == 'compile-failure': raise ValueError('compiler exit 101')
                    binary.write_bytes(b'fixture')
                    event = dict(reason='compiler-artifact', executable=str(binary), target={'name': 'mlx_gen_moss_image', 'kind': ['lib']}, profile={'test': True})
                    if mode == 'wrong-target': event['target']['kind'] = ['bin']
                    return json.dumps(event)
                if label.startswith('list'):
                    self.assertNotIn('MOSS_NATIVE_TEST_ADMITTED', env)
                    if mode == 'missing-ignored' and label == 'list-ignored': return ''
                    if mode == 'duplicate-test': return (gate.TEST+': test\n')*2
                    return gate.TEST+': test\n'
                self.assertEqual(label, 'oracle')
                self.assertEqual(command, [str(binary), gate.TEST, '--exact', '--ignored', '--nocapture', '--test-threads=1'])
                self.assertEqual(env['MOSS_NATIVE_TEST_ADMITTED'], '1')
                if mode == 'numerical-failure': raise ValueError('oracle exit 101')
                if mode == 'source-during': source['source'] = 'mutated'
                if mode == 'binary-during': binary.write_bytes(b'mutated')
                if mode == 'runtime-during': runtime['rustc'] = 'mutated'
                return 'test result: ok. 0 passed; 0 failed; 1 ignored;' if mode == 'skipped' else 'test result: ok. 1 passed; 0 failed; 0 ignored;'
            with patch.dict(os.environ, environment), patch.object(gate.sys, 'platform', 'darwin'), patch.object(gate, 'paths', side_effect=paths), patch.object(gate, 'inventory', side_effect=lambda: source.copy()), patch.object(gate, 'runtime_identity', side_effect=lambda *args: runtime.copy()), patch.object(gate, 'run_owned', side_effect=runner), patch('builtins.print'):
                if mode in ('compile-failure', 'wrong-target', 'missing-ignored', 'duplicate-test'):
                    with self.assertRaises(ValueError): gate.prepare()
                    self.assertNotIn('oracle', [row[0] for row in calls]); return
                gate.prepare()
                prepared = gate.read(out/'prepared.json')
                if mode == 'binary-before': (out/'target'/'test-binary').write_bytes(b'changed')
                if mode == 'source-before': source['source'] = 'changed'
                if mode == 'runtime-before': runtime['rustc'] = 'changed'
                if mode == 'prepared-test': prepared['members'] = ['other']
                if mode == 'prepared-source': prepared['source'] = 'b'*40
                if mode == 'prepared-attempt': prepared['attempt'] = 1
                if mode == 'prepared-controller': prepared['controllerSha256'] = 'b'*64
                gate.write(out/'prepared.json', prepared)
                if mode == 'success': gate.execute()
                else:
                    with self.assertRaises(ValueError): gate.execute()
                result = gate.read(out/'result.json')
                self.assertEqual(result['passed'], mode == 'success')
                expected_runs = 1 if mode in ('success', 'numerical-failure', 'skipped', 'source-during', 'binary-during', 'runtime-during') else 0
                self.assertEqual(sum(row[0] == 'oracle' for row in calls), expected_runs)

    def test_direct_prepare_then_exact_full_graph_without_lease(self): self.exercise()

    def test_compile_and_membership_errors_never_execute(self):
        for mode in ('compile-failure', 'wrong-target', 'missing-ignored', 'duplicate-test'):
            with self.subTest(mode=mode): self.exercise(mode)

    def test_changed_source_binary_runtime_and_prepared_identity_refuse(self):
        for mode in ('binary-before', 'source-before', 'runtime-before', 'prepared-test', 'prepared-source', 'prepared-attempt', 'prepared-controller'):
            with self.subTest(mode=mode): self.exercise(mode)

    def test_numerical_failure_skip_and_during_run_mutation_are_terminal(self):
        for mode in ('numerical-failure', 'skipped', 'source-during', 'binary-during', 'runtime-during'):
            with self.subTest(mode=mode): self.exercise(mode)


class OwnedProcessTests(unittest.TestCase):
    def test_retained_leader_fast_exit_failure_timeout_and_descendants(self):
        for mode in ('fast', 'failed', 'timeout', 'descendant', 'cleanup-failure', 'identity-change', 'exec-change', 'foreign-group'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                clock = [0.]; live = [mode in ('timeout', 'exec-change')]; child_live = [mode in ('descendant', 'cleanup-failure', 'identity-change')]; calls = []; queries = [0]; reaped = [False]
                class Process:
                    pid = 100
                    def poll(self): raise AssertionError('leader must remain unreaped until drain')
                    def wait(self, timeout):
                        reaped[0] = True; calls.append(('wait', 100))
                        return 101 if mode == 'failed' else 0
                def rows(pgid, deadline):
                    self.assertEqual(pgid, 100)
                    self.assertFalse(reaped[0], 'no group adoption after leader is reaped')
                    queries[0] += 1
                    if mode == 'exec-change' and queries[0] > 1: live[0] = False
                    result = {100: {'birth': 'reused-root' if mode == 'foreign-group' and queries[0] > 1 else 'root', 'comm': 'cargo' if queries[0] == 1 else 'cargo-exited', 'state': 'S' if live[0] else 'Z'}}
                    if child_live[0]: result[101] = {'birth': 'reused' if mode == 'identity-change' and queries[0] > 1 else 'child', 'comm': 'clang', 'state': 'S'}
                    if mode == 'foreign-group' and queries[0] > 1: result[999] = {'birth': 'foreign', 'comm': 'foreign', 'state': 'S'}
                    return result
                def kill(pid, sig):
                    self.assertFalse(reaped[0]); calls.append((pid, sig))
                    self.assertIn(pid, (100, 101))
                    if pid == 100: live[0] = False
                    elif mode != 'cleanup-failure': child_live[0] = False
                with patch.object(gate.subprocess, 'Popen', return_value=Process()) as spawn, patch.object(gate, 'group_rows', side_effect=rows), patch.object(gate.os, 'kill', side_effect=kill), patch.object(gate.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(gate.time, 'sleep', side_effect=lambda value: clock.__setitem__(0, clock[0]+value)):
                    if mode in ('fast', 'descendant', 'exec-change'): gate.run_owned(['fixture'], {}, Path(directory), 'case', 1)
                    else:
                        with self.assertRaises(ValueError): gate.run_owned(['fixture'], {}, Path(directory), 'case', 1)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
                receipt = gate.read(Path(directory)/'case-process.json')
                self.assertEqual(calls[-1], ('wait', 100))
                if mode in ('identity-change', 'foreign-group'): self.assertEqual(calls, [('wait', 100)])
                elif mode != 'cleanup-failure': self.assertEqual(receipt['remaining'], {})
                self.assertLessEqual(clock[0], 11)

    def test_group_query_is_scoped_and_refuses_unrelated_group(self):
        class Result:
            returncode = 0
            stdout = '100 100 Z Fri Oct 23 12:00:00 2026 /cargo\n'
        with patch.object(gate.subprocess, 'run', return_value=Result()) as run:
            self.assertEqual(set(gate.group_rows(100, gate.time.monotonic()+10)), {100})
            self.assertEqual(run.call_args.args[0][:3], ['ps', '-g', '100'])
            Result.stdout = Result.stdout.replace('100 100', '100 200')
            with self.assertRaises(ValueError): gate.group_rows(100, gate.time.monotonic()+10)


class WorkflowTests(unittest.TestCase):
    def test_only_selected_numerical_lane_runs_without_operator_inputs(self):
        workflow = yaml.safe_load((gate.ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
        inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
        self.assertEqual(set(inputs), {'lanes', 'synthetic_w4a8'})
        self.assertEqual(inputs['lanes']['default'], 'all')
        self.assertEqual(inputs['synthetic_w4a8']['default'], 'none')
        self.assertEqual(inputs['synthetic_w4a8']['options'], ['none', 'cuda', 'mlx', 'both'])
        self.assertTrue(all(not spec.get('required', False) for spec in inputs.values()))
        self.assertNotIn('moss-edit-metal-lease', workflow['jobs'])
        for lane in ('moss-edit-metal', 'moss-edit-cuda'):
            for selected in (True, False):
                for fork in (True, False):
                    for synthetic in ('none', 'cuda', 'mlx', 'both'):
                        self.assertEqual(selected_jobs(workflow, 'workflow_dispatch', lane, selected=selected, fork=fork, synthetic=synthetic), {lane})
            for event in ('push', 'pull_request', 'schedule', 'merge_group'):
                self.assertNotIn(lane, selected_jobs(workflow, event, lane))
            mutant = copy.deepcopy(workflow)
            mutant['jobs'][lane]['if'] = mutant['jobs'][lane]['if'].replace(f"inputs.lanes == '{lane}'", 'True')
            self.assertIn(lane, selected_jobs(mutant, 'workflow_dispatch', 'all'))
        job = workflow['jobs']['moss-edit-metal']
        commands = '\n'.join(step.get('run', '') for step in job['steps'])
        self.assertIn('moss_edit_metal.py prepare', commands)
        self.assertIn('moss_edit_metal.py execute', commands)
        self.assertNotIn('GH_TOKEN', job.get('env', {}))
        self.assertNotIn('concurrency', job)


if __name__ == '__main__': unittest.main()
