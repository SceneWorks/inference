"""CPU policy checks for the read-only Windows GPU1 evidence transport.

These exercise event selection and constrain the collector's operation surface.
Actual Windows SDK compilation, CIM and counters remain Windows-only validation.
"""
import copy
import math
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/ci.yml'
SCRIPT = ROOT / 'scripts/ci/windows-gpu1-diagnostic.ps1'
MAPPER = ROOT / 'scripts/ci/windows-gpu1-luid.cpp'
LANE = 'windows-gpu-diagnostic'


def selected_jobs(workflow, event, lane, *, selected=True, fork=False, tag=False):
    """Evaluate actual job expressions with upstream lanes adversarially selected.

    Always returns true even after dependency failure, so a missing diagnostic
    guard on the aggregate gate cannot hide behind the implicit success policy.
    """
    context = {
        'github.event.pull_request.head.repo.full_name': 'foreign/repo' if fork else 'SceneWorks/inference',
        'github.repository': 'SceneWorks/inference',
        'github.event_name': event,
        'github.ref': 'refs/tags/runtime-test' if tag else 'refs/heads/story/test',
        'inputs.lanes': lane,
    }
    result = set()
    for name, job in workflow['jobs'].items():
        expression = job.get('if', 'True')
        expression = re.sub(r'needs\.changes\.outputs\.\w+', repr(str(selected).lower()), expression)
        for key in sorted(context, key=len, reverse=True):
            expression = expression.replace(key, repr(context[key]))
        expression = expression.replace('&&', ' and ').replace('||', ' or ')
        if eval(expression, {'__builtins__': {}}, {'always': lambda: True, 'startsWith': str.startswith}):
            result.add(name)
    return result


def diagnostic_isolation_errors(workflow):
    errors = []
    for selected in (False, True):
        for fork in (False, True):
            for tag in (False, True):
                actual = selected_jobs(workflow, 'workflow_dispatch', LANE, selected=selected, fork=fork, tag=tag)
                if actual != {LANE}:
                    errors.append(actual)
    return errors


def scalar_predicate(script, name, **values):
    """Evaluate the actual pure PS predicate in a deliberately narrow CPU subset.

    Does not emulate PowerShell parsing/binding, CIM, PDH or native execution.
    """
    match = re.search(r'function ' + re.escape(name) + r'\([^\n]+\) \{\n    return (.+)\n\}', script)
    if match is None:
        raise AssertionError('missing pure scalar predicate: ' + name)
    expression = match[1]
    expression = expression.replace('[double]::IsNaN', 'isnan').replace('[double]::IsInfinity', 'isinf')
    for op, py in (('-not ', 'not '), ('-and', 'and'), ('-or', 'or'), ('-gt', '>'), ('-ge', '>='), ('-lt', '<'), ('-le', '<='), ('-eq', '==')):
        expression = expression.replace(op, py)
    expression = re.sub(r'\$(\w+)', r'\1', expression)
    if re.search(r'[^\w\s()<>.=,]', expression):
        raise AssertionError('predicate outside tested scalar subset')
    return bool(eval(expression, {'__builtins__': {}}, dict(values, isnan=math.isnan, isinf=math.isinf)))


def cohort_structure_errors(script):
    """Pin the pre-window identity boundary, including non-NVML counter PIDs."""
    errors = []
    sequence = (
        '$discovery = @(Read-Mapped-Counters $record.mapping.counterLuid 1 $false)',
        '$cohort = @(@($census',
        "$sample['identitiesBefore'] = @(Process-Identity $cohort)",
        "$sample['windowStartedUtc']",
        '$sample.counters = @(Read-Mapped-Counters $record.mapping.counterLuid 2 $true)',
        "$sample['windowEndedUtc']",
        '$observed = @(@($afterCensus',
        '$observedPid -notin $cohort',
        "$sample['identitiesAfter'] = @(Process-Identity $processIds)",
        '$before.createdUtc -ne $after[0].createdUtc',
    )
    positions = [script.find(token) for token in sequence]
    if -1 in positions or positions != sorted(positions):
        errors.append('discovery/pre-CIM/window/post-CIM order is not preserved')
    cohort = next((line for line in script.splitlines() if '$cohort = ' in line), '')
    if '$discovery | Where-Object' not in cohort:
        errors.append('counter-only PIDs missing from pre-window cohort')
    for token in ('New PID during counter window without pre-window identity', '$cohort.Count -gt 256', '$processIds.Count -gt 256', 'CIM identity unknown for PID', 'Process identity changed for PID'):
        if token not in script:
            errors.append(token)
    return errors


class WindowsGpu1DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_diagnostic_dispatch_skips_every_sibling_even_with_all_lanes_selected(self):
        self.assertEqual(diagnostic_isolation_errors(self.workflow), [])
        self.assertGreater(len(self.workflow['jobs']), 15)

    def test_isolation_detects_each_missing_sibling_guard(self):
        guard = "(github.event_name != 'workflow_dispatch' || inputs.lanes != 'windows-gpu-diagnostic')"
        for name, job in self.workflow['jobs'].items():
            if name == LANE:
                continue
            with self.subTest(job=name):
                self.assertTrue(job['if'].startswith(guard))
                mutated = copy.deepcopy(self.workflow)
                tail = job['if'][len(guard):].removeprefix(' && ')
                mutated['jobs'][name]['if'] = tail or 'True'
                self.assertNotEqual(diagnostic_isolation_errors(mutated), [])

    def test_diagnostic_never_runs_for_other_events_or_selectors(self):
        for event in ('push', 'pull_request', 'schedule', 'workflow_call', 'merge_group'):
            for lane in ('', 'all', 'windows-cuda', LANE):
                with self.subTest(event=event, lane=lane):
                    self.assertNotIn(LANE, selected_jobs(self.workflow, event, lane))
        for lane in ('', 'all', 'windows-cuda', LANE + '-extra'):
            self.assertNotIn(LANE, selected_jobs(self.workflow, 'workflow_dispatch', lane))

    def test_normal_gate_still_runs_after_failed_dependencies_and_cuda_trust_is_preserved(self):
        for event in ('push', 'pull_request', 'workflow_dispatch'):
            for lane in ('all', 'windows-cuda'):
                self.assertTrue({'gate', 'supply-chain', 'changes', 'resolve', 'format'} <= selected_jobs(self.workflow, event, lane, selected=False))
        self.assertNotIn('windows-cuda-check', selected_jobs(self.workflow, 'pull_request', 'all', fork=True))
        self.assertIn('windows-cuda-check', selected_jobs(self.workflow, 'pull_request', 'all'))
        self.assertIn('windows-cuda', selected_jobs(self.workflow, 'workflow_dispatch', 'windows-cuda'))

    def test_diagnostic_job_is_independent_bounded_and_uploads_only_metadata(self):
        job = self.workflow['jobs'][LANE]
        self.assertNotIn('needs', job)
        self.assertEqual(job['runs-on'], ['self-hosted', 'windows', 'cuda'])
        self.assertEqual(job['timeout-minutes'], 5)
        self.assertEqual(job['env'], {'CUDA_VISIBLE_DEVICES': '-1'})
        self.assertEqual(len(job['steps']), 3)
        checkout, capture, upload = job['steps']
        self.assertRegex(checkout['uses'], r'^actions/checkout@[0-9a-f]{40}$')
        self.assertEqual(capture['shell'], 'powershell')
        self.assertIn('.\\scripts\\ci\\windows-gpu1-diagnostic.ps1 -OutputDirectory $out', capture['run'])
        self.assertEqual(upload['if'], 'always()')
        self.assertRegex(upload['uses'], r'^actions/upload-artifact@[0-9a-f]{40}$')
        self.assertEqual(upload['with']['path'], '${{ runner.temp }}/gpu1-diagnostic-${{ github.run_id }}-${{ github.run_attempt }}/evidence.json')
        self.assertEqual(upload['with']['retention-days'], 3)

    def test_collector_has_no_admission_or_process_control_or_product_api(self):
        script = SCRIPT.read_text(encoding="utf-8")
        commands = '\n'.join(line for line in script.splitlines() if not line.lstrip().startswith('#'))
        for forbidden in ('Stop-Process', 'taskkill', 'Invoke-WebRequest', 'Invoke-RestMethod', 'CommandLine', 'Get-ChildItem Env:', 'cuInit', 'nvcuda', 'cargo ', 'pip ', 'D3D11CreateDevice', 'CreateDevice'):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.lower(), commands.lower())
        self.assertEqual(script.count("admission = 'not_evaluated'"), 1)
        self.assertEqual(script.count("applicationJobOwnership = 'unknown'"), 1)
        self.assertNotRegex(script, r'(?i)(?:\.admission|\[.admission.\]|\.applicationJobOwnership)\s*=')
        self.assertIn("if ($issues.Count -ne 0) { exit 1 }", script)
        self.assertIn("$record['telemetryComplete'] = ($issues.Count -eq 0)", script)

    def test_census_mapping_and_pid_reuse_errors_remain_explicit(self):
        script = SCRIPT.read_text(encoding="utf-8")
        for required in ('Identity \'1\'', 'Identity $uuid', 'Census $uuid', 'Executable path unavailable', 'Process census structure unavailable', 'Empty process census has unsupported status', 'Unknown process type', 'GPU process census changed during sample', 'Process identity changed for PID', 'Windows counter attribution unavailable', 'Mapped counter rows missing or exceed bound', 'Test-Counter-Value $row.Status $row.CookedValue', '$before.createdUtc -ne $after[0].createdUtc', "$record.mapping.counterLuid", '[regex]::Escape($Luid)', "'(^|_)pid_(\\d+)(_|$)'", '4MB', '-lt 3', '-MaxSamples $Count', '-gt 2048', '-gt 256'):
            with self.subTest(required=required):
                self.assertIn(required, script)
        self.assertIn('-Property ProcessId,Name,CreationDate,ParentProcessId,ExecutablePath', script)
        self.assertNotIn('phys_0', script)

    def test_device_measurement_predicate_rejects_nonfinite_and_inconsistent_values(self):
        script = SCRIPT.read_text(encoding="utf-8")
        cases = (
            ((97887, 1685, 0), True), ((1, 0, 100), True), ((1, 1, 0), True),
            ((0, 0, 0), False), ((-1, 0, 0), False), ((1, -1, 0), False),
            ((1, 2, 0), False), ((1, 0, -0.01), False), ((1, 0, 100.01), False),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(scalar_predicate(script, 'Test-Device-Numbers', **dict(zip(('Total', 'Used', 'Utilization'), args))), expected)
        for field in ('Total', 'Used', 'Utilization'):
            for value in (float('nan'), float('inf'), float('-inf')):
                args = dict(Total=97887, Used=1685, Utilization=1)
                args[field] = value
                with self.subTest(field=field, value=value):
                    self.assertFalse(scalar_predicate(script, 'Test-Device-Numbers', **args))
        numeric_parser = script.split('function Device-Numbers(', 1)[1].split('function Test-Counter-Value(', 1)[0]
        for required in ("@('totalMiB', 'usedMiB', 'utilization')", '[double]::TryParse([string]$Row.$field', '[Globalization.NumberStyles]::Float', '[Globalization.CultureInfo]::InvariantCulture', "return @{ status = 'unknown' }", 'if (-not (Test-Device-Numbers'):
            self.assertIn(required, numeric_parser)
        self.assertIn('-NotePropertyValue (Device-Numbers $rows[0])', script)
        self.assertIn('Device total memory changed during diagnostic', script)

    def test_pdh_accepts_only_documented_valid_and_new_data_with_finite_values(self):
        script = SCRIPT.read_text(encoding="utf-8")
        for status in (0, 1, 2, 0x800007D1, 0x800007D3, 0xC0000BBA, 0xFFFFFFFF):
            with self.subTest(status=status):
                self.assertEqual(scalar_predicate(script, 'Test-Counter-Value', Status=status, Value=1.5), status in (0, 1))
        for status in (0, 1):
            for value in (float('nan'), float('inf'), float('-inf'), -1):
                self.assertFalse(scalar_predicate(script, 'Test-Counter-Value', Status=status, Value=value))
        self.assertIn('status = $row.Status', script)
        self.assertIn('rawValue = $row.CookedValue.ToString', script)
        self.assertIn('if ($ValidateValues -and -not $valid) { $issues.Add(', script)
        self.assertIn('if ($ValidateValues -and $valid) { $value = $row.CookedValue }', script)
        self.assertIn('pdh-error-codes', script)

    def test_counter_only_pids_have_pre_window_identity_and_changes_are_unknown(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(cohort_structure_errors(script), [])
        mutations = (
            script.replace('$discovery | Where-Object', '$census | Where-Object'),
            script.replace("$sample['identitiesBefore'] = @(Process-Identity $cohort)", ''),
            script.replace('$observedPid -notin $cohort', '$observedPid -in $cohort'),
            script.replace('$before.createdUtc -ne $after[0].createdUtc', '$before.name -ne $after[0].name'),
        )
        for index, mutated in enumerate(mutations):
            with self.subTest(mutation=index):
                self.assertNotEqual(cohort_structure_errors(mutated), [])

    def test_mapper_uses_only_documented_sdk_metadata_and_closes_handles(self):
        mapper = MAPPER.read_text(encoding="utf-8")
        calls = set(re.findall(r'\b(D3D\w+)\s*\(', mapper))
        self.assertEqual(calls, {'D3DKMTEnumAdapters2', 'D3DKMTQueryAdapterInfo', 'D3DKMTCloseAdapter'})
        self.assertIn('#include <d3dkmthk.h>', mapper)
        self.assertIn('query.Type = KMTQAITYPE_ADAPTERADDRESS;', mapper)
        self.assertIn('query.Type = KMTQAITYPE_PHYSICALADAPTERCOUNT;', mapper)
        self.assertIn('physical.Count != 1', mapper)
        self.assertIn('matches != 1', mapper)
        self.assertIn('~Adapters() { close(); }', mapper)
        self.assertIn('if (!adapters.close()) throw', mapper)
        self.assertLess(mapper.index('Adapters adapters;'), mapper.index('enumeration.pAdapters ='))
        self.assertNotRegex(mapper, r'(?i)cuda|D3D\d+Create|CreateContext|CreateDevice|CreateAllocation|malloc\(')
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('/link gdi32.lib', script)
        self.assertIn("Join-Path $root 'host-helper'", script)
        self.assertIn('$mapCode -ne 0', script)
        self.assertIn("$map.status -ne 'mapped'", script)


if __name__ == '__main__':
    unittest.main()
