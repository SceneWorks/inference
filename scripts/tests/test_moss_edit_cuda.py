"""CPU tests for exact direct CUDA execution and retained owned-child cleanup."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from scripts.ci import moss_edit_cuda as gate


class DirectExecutionTests(unittest.TestCase):
    def exercise(self, mode='success'):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); target = out/'target'; target.mkdir()
            env = {'CARGO_TARGET_DIR': str(target), 'GITHUB_SHA': 'a'*40}
            binaries = {}
            for name in (gate.MOSS, gate.FLUX):
                binaries[name] = target/(name+'.exe'); binaries[name].write_bytes(b'fixture')
            calls = []; changed = [False]
            def runner(args, child_env, output, label, timeout, **kwargs):
                calls.append((label, args, child_env.copy()))
                self.assertGreater(timeout, 0)
                if label == 'compile':
                    self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'], '-1')
                    self.assertEqual(child_env['CARGO_BUILD_JOBS'], '4')
                    self.assertTrue(kwargs['finalize_compile_descendants'])
                    if mode == 'compile-failure': return 101, ''
                    rows = [dict(reason='compiler-artifact', executable=str(path), target={'name': name.replace('-', '_'), 'kind': ['lib']}, profile={'test': True}) for name, path in binaries.items()]
                    if mode == 'missing-binary': rows.pop()
                    if mode == 'source-change': changed[0] = True
                    return 0, '\n'.join(map(json.dumps, rows))
                arm = next(row for row in gate.ARMS if label.startswith(row[0]+'-'))
                _, package, filters, expected, ignored = arm
                if label.endswith(('-list', '-ignored')):
                    self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'], '-1')
                    names = ignored if label.endswith('-ignored') else expected
                    if mode == 'missing-test' and label == 'edit-list': names = set()
                    if mode == 'unexpected-ignored' and label == 'edit-ignored': names = expected
                    if mode == 'binary-change' and label == 'encoder-ignored': binaries[gate.MOSS].write_bytes(b'changed')
                    return 0, '\n'.join(name+': test' for name in sorted(names))
                if label.endswith('-device'):
                    self.assertEqual(args, ['nvidia-smi', '--id=1', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader,nounits'])
                    return 0, f"1, {gate.GPU_UUID}, {gate.GPU_PCI}" if mode != 'wrong-device' else f"0, {gate.GPU_UUID}, {gate.GPU_PCI}"
                self.assertTrue(label.endswith('-tests'))
                self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'], gate.GPU_UUID)
                self.assertEqual(child_env['SCENEWORKS_GPU_ID'], '1')
                self.assertEqual(child_env['REQUIRE_SM120'], '1')
                self.assertEqual('--include-ignored' in args, bool(ignored))
                if mode == 'numerical-failure': return 101, 'test result: FAILED. 0 passed; 1 failed; 0 ignored;'
                if mode == 'skipped': return 0, f'test result: ok. 0 passed; 0 failed; {len(expected)} ignored;'
                return 0, f'test result: ok. {len(expected)} passed; 0 failed; 0 ignored;'
            source = lambda: {'fixture': 'changed' if changed[0] else 'original'}
            if mode == 'success':
                result = gate.execute(env, out, runner=runner, source_check=source)
                self.assertEqual([r['arm'] for r in result], ['edit', 'layout', 'encoder'])
                self.assertEqual(sum(len(r['expectedTests']) for r in result), 4)
                self.assertTrue(all(r['passed'] for r in result))
            else:
                with self.assertRaises(ValueError): gate.execute(env, out, runner=runner, source_check=source)
                numeric = [label for label, _, _ in calls if label.endswith('-tests')]
                self.assertEqual(numeric, ['edit-tests'] if mode in ('numerical-failure', 'skipped') else [])
            return calls

    def test_exact_four_numerical_cases_execute_without_external_admission(self):
        calls = self.exercise()
        self.assertEqual([label for label, _, _ in calls if label.endswith('-tests')], ['edit-tests', 'layout-tests', 'encoder-tests'])

    def test_compile_identity_membership_source_binary_and_numerical_failures_are_terminal(self):
        for mode in ('compile-failure', 'missing-binary', 'missing-test', 'unexpected-ignored', 'source-change', 'binary-change', 'wrong-device', 'numerical-failure', 'skipped'):
            with self.subTest(mode=mode): self.exercise(mode)

    def test_device_identity_rejects_other_physical_index_uuid_pci_or_ambiguous_rows(self):
        good = f'1, {gate.GPU_UUID}, {gate.GPU_PCI}'
        self.assertEqual(gate.device_identity(good)['index'], 1)
        for bad in (good.replace('1,', '0,', 1), good.replace(gate.GPU_UUID, 'GPU-other'), good.replace('C1:', '41:'), good+'\n'+good, '', good+',extra'):
            with self.subTest(bad=bad), self.assertRaises(ValueError): gate.device_identity(bad)

    def test_success_requires_exact_nonignored_nonzero_case_count(self):
        for code, log in ((0, ''), (1, 'test result: ok. 1 passed; 0 failed; 0 ignored;'), (0, 'test result: ok. 0 passed; 0 failed; 1 ignored;')):
            self.assertFalse(gate.successful_cases(code, log, {'case'}))


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
        self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(job['env']['CUDA_COMPUTE_CAP'], '120'); self.assertEqual(job['env']['CARGO_BUILD_JOBS'], '4')
        target = next(step for step in job['steps'] if step.get('name', '').startswith('Reserve an exclusive'))
        self.assertIn('$env:RUNNER_TEMP', target['run']); self.assertIn('$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT', target['run'])
        self.assertIn('Target directory already exists', target['run'])
        self.assertNotIn('concurrency', job)
        commands = '\n'.join(step.get('run', '') for step in job['steps'])
        self.assertNotIn('uv python install', commands); self.assertNotIn('pip install', commands)
        self.assertIn('-m scripts.ci.moss_edit_cuda', commands)
        run = next(step for step in job['steps'] if 'scripts.ci.moss_edit_cuda' in step.get('run', ''))
        self.assertNotIn('MOSS_CUDA_ADMISSION', run.get('env', {}))
        self.assertNotIn('${{ inputs.moss_cuda_admission }}', commands)

    def test_new_selector_isolates_job_with_semantic_sibling_barriers(self):
        from scripts.tests.test_windows_gpu1_diagnostic import selected_jobs
        workflow = yaml.safe_load((gate.ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
        for selected in (True, False):
            for fork in (True, False):
                for tag in (True, False):
                    for synthetic in ('none', 'cuda', 'mlx', 'both'):
                        self.assertEqual(selected_jobs(workflow, 'workflow_dispatch', 'moss-edit-cuda', selected=selected, fork=fork, tag=tag, synthetic=synthetic), {'moss-edit-cuda'})
        for event in ('push', 'pull_request', 'schedule', 'merge_group'):
            self.assertNotIn('moss-edit-cuda', selected_jobs(workflow, event, 'moss-edit-cuda'))
        for lane in ('all', 'windows-cuda', 'windows-gpu-diagnostic', 'moss-edit-cuda-extra'):
            self.assertNotIn('moss-edit-cuda', selected_jobs(workflow, 'workflow_dispatch', lane))
        guard = "(github.event_name != 'workflow_dispatch' || inputs.lanes != 'moss-edit-cuda')"
        for name, job in workflow['jobs'].items():
            if name == 'moss-edit-cuda':
                continue
            condition = job['if']
            selector = f"inputs.lanes == '{name}'"
            self.assertTrue(guard in condition or selector in condition, name)
            condition = condition.replace(guard + ' && ', '').replace(guard, 'True')
            condition = condition.replace(selector, "inputs.lanes == 'moss-edit-cuda'")
            mutant = copy.deepcopy(workflow)
            mutant['jobs'][name]['if'] = condition
            self.assertTrue(any(name in selected_jobs(mutant, 'workflow_dispatch', 'moss-edit-cuda', synthetic=synthetic, tag=tag)
                                for synthetic in ('none', 'cuda', 'mlx', 'both') for tag in (False, True)), name)
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
