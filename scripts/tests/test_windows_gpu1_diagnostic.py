"""CPU policy checks for the read-only Windows GPU1 evidence transport.

These exercise event selection and constrain the collector's operation surface.
Actual Windows SDK compilation, CIM and counters remain Windows-only validation.
"""
import copy
import json
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


def identity_observation_contract(script, before, after, *, telemetry_errors=()):
    """Exercise the actual PS branch predicates in a narrow Python subset.

    Source-wiring assertions bind the mandatory/optional destinations. This does
    not emulate PowerShell, CIM access, or prove Windows runtime behavior.
    """
    def condition(marker, bindings):
        line = next(line for line in script.splitlines() if marker in line)
        expression = re.search(r'if \((.*)\) \{', line)[1]
        values = {}
        for index, (token, value) in enumerate(sorted(bindings.items(), key=lambda item: -len(item[0]))):
            name = f'v{index}'
            expression = expression.replace(token, name)
            values[name] = value
        for op, py in (('-not ', 'not '), ('-and', 'and'), ('-or', 'or'), ('-ne', '!='), ('-eq', '==')):
            expression = expression.replace(op, py)
        if re.search(r'[^\w\s()!.=]', expression):
            raise AssertionError('identity condition outside tested subset: ' + expression)
        return bool(eval(expression, {'__builtins__': {}}, values))

    errors, warnings, identities = list(telemetry_errors), [], []
    for row in (before, after):
        fields = row or {}
        bindings = {'$null': None, '$row': row, '$row.ProcessId': fields.get('pid'),
                    '$processIdValue': 42, '$row.CreationDate': fields.get('created'),
                    '$row.Name': fields.get('name')}
        if condition("throw 'Process identity unavailable or exited'", bindings):
            errors.append('identity unavailable')
            identities.append(None)
            continue
        path_available = bool(fields.get('path'))
        if condition('if (-not $row.ExecutablePath)', {'$row.ExecutablePath': fields.get('path')}):
            warnings.append('path unavailable')
        identities.append(dict(fields, pathAvailable=path_available))
    if all(identities):
        bindings = {}
        for prefix, row in zip(('$before', '$after[0]'), identities):
            for ps, field in (('createdUtc', 'created'), ('name', 'name'), ('path', 'path'), ('pathAvailable', 'pathAvailable')):
                bindings[prefix + '.' + ps] = row.get(field)
        if condition('if ($before.createdUtc', bindings):
            errors.append('identity changed')
        if condition('if ($before.pathAvailable', bindings):
            warnings.append('path availability changed')
    return {'telemetryComplete': not errors, 'errors': errors,
            'optionalMetadataWarnings': warnings, 'identities': identities,
            'admission': 'not_evaluated', 'applicationJobOwnership': 'unknown'}


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


EVIDENCE_BUDGET = 4 * 1024 * 1024


def evidence_budget_contract(record):
    """Executable object-level budget oracle, not a PowerShell runtime emulator.

    Full JSON round-trip must be lossless below the byte budget. Overflow must
    terminate with an explicit incomplete summary independently of input size.
    Source-wiring checks below bind this contract to the collector's save sites.
    """
    encoded = json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) <= EVIDENCE_BUDGET:
        return encoded, True
    summary = {
        'schemaVersion': 1, 'completed': True, 'collectionComplete': False,
        'telemetryComplete': False, 'evidenceTruncated': True,
        'recordDetailsOmitted': True, 'terminalReason': 'evidence_budget_exceeded',
        'admission': 'not_evaluated', 'applicationJobOwnership': 'unknown',
        'originalCompactUtf8Bytes': len(encoded), 'limitUtf8Bytes': EVIDENCE_BUDGET,
        'originalSampleCount': len(record.get('samples', [])),
        'originalQueryCount': len(record.get('queries', [])),
        'originalErrorCount': len(record.get('errors', [])),
        'originalOptionalMetadataWarningCount': len(record.get('optionalMetadataWarnings', [])),
        'sampleRows': [{'counters': len(sample['counters']), 'utc': sample['utc'][:128]}
                       for sample in record.get('samples', [])[:3]],
    }
    for field in ('host', 'runner', 'run', 'attempt', 'source'):
        summary[field] = str(record.get(field, ''))[:128]
    return json.dumps(summary, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), False


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
        self.assertEqual(script.count("admission = 'not_evaluated'"), 2)
        self.assertEqual(script.count("applicationJobOwnership = 'unknown'"), 2)
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

    def test_optional_path_metadata_does_not_hide_required_identity_failures(self):
        script = SCRIPT.read_text(encoding="utf-8")
        row = dict(pid=42, name='system.exe', created='2026-10-22T23:00:00Z', path=None)
        observed = identity_observation_contract(script, row, row)
        self.assertTrue(observed['telemetryComplete'])
        self.assertEqual(observed['optionalMetadataWarnings'], ['path unavailable'] * 2)
        self.assertTrue(all(not x['pathAvailable'] for x in observed['identities']))
        self.assertEqual(observed['admission'], 'not_evaluated')
        self.assertEqual(observed['applicationJobOwnership'], 'unknown')
        encoded, full = evidence_budget_contract(observed)
        self.assertTrue(full)
        self.assertEqual(json.loads(encoded), observed)
        for bad in (None, dict(row, pid=43), dict(row, name=''), dict(row, created=None),
                    dict(row, created='2026-10-22T23:01:00Z'), dict(row, name='other.exe')):
            with self.subTest(bad=bad):
                self.assertFalse(identity_observation_contract(script, row, bad)['telemetryComplete'])
        for failure in ('mapping unknown', 'query failed', 'invalid counter', 'new PID', 'device value N/A'):
            self.assertFalse(identity_observation_contract(script, row, row, telemetry_errors=[failure])['telemetryComplete'])

    def test_path_availability_transitions_are_optional_but_known_mismatch_is_mandatory(self):
        script = SCRIPT.read_text(encoding="utf-8")
        row = dict(pid=42, name='system.exe', created='2026-10-22T23:00:00Z', path=None)
        known = dict(row, path=r'C:\Windows\System32\system.exe')
        for before, after in ((row, known), (known, row)):
            observed = identity_observation_contract(script, before, after)
            self.assertTrue(observed['telemetryComplete'])
            self.assertIn('path unavailable', observed['optionalMetadataWarnings'])
            self.assertIn('path availability changed', observed['optionalMetadataWarnings'])
            self.assertEqual([x['pathAvailable'] for x in observed['identities']], [bool(before['path']), bool(after['path'])])
        self.assertTrue(identity_observation_contract(script, known, known)['telemetryComplete'])
        changed = identity_observation_contract(script, known, dict(known, path=r'C:\Other\system.exe'))
        self.assertFalse(changed['telemetryComplete'])
        self.assertIn('identity changed', changed['errors'])
        # Removing the both-known guard is discriminated by the transition case.
        mutant = script.replace('$before.pathAvailable -and $after[0].pathAvailable -and ', '')
        self.assertFalse(identity_observation_contract(mutant, row, known)['telemetryComplete'])

    def test_optional_warning_wiring_keeps_mandatory_errors_and_budget_bounds(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('optionalMetadataWarnings = $warnings', script)
        self.assertIn('$warnings = [Collections.Generic.List[string]]::new()', script)
        self.assertIn('$warnings.Add("Executable path unavailable for PID $processIdValue")', script)
        self.assertIn('$warnings.Add("Executable path availability changed for PID $($before.pid)")', script)
        self.assertNotIn('$issues.Add("Executable path', script)
        self.assertIn('$issues.Add("CIM identity unknown for PID $processIdValue")', script)
        self.assertIn('$issues.Add("Process identity changed for PID $($before.pid)")', script)
        self.assertIn("$record['telemetryComplete'] = ($issues.Count -eq 0)", script)
        self.assertIn('originalOptionalMetadataWarningCount = $warnings.Count', script)
        # Optional details are explicitly counted if the unchanged budget overflows.
        record = {'optionalMetadataWarnings': ['x' * 2048] * 4096}
        encoded, full = evidence_budget_contract(record)
        self.assertFalse(full)
        result = json.loads(encoded)
        self.assertLess(len(encoded), 8192)
        self.assertEqual(result['originalOptionalMetadataWarningCount'], 4096)
        self.assertFalse(result['telemetryComplete'])
        self.assertTrue(result['recordDetailsOmitted'])

    def test_evidence_budget_below_exact_and_above_limit(self):
        envelope = {'completed': False, 'payload': ''}
        overhead = len(json.dumps(envelope, separators=(',', ':')).encode('utf-8'))
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                record = dict(envelope, payload='x' * (EVIDENCE_BUDGET - overhead + delta))
                encoded, complete_record = evidence_budget_contract(record)
                self.assertLessEqual(len(encoded), EVIDENCE_BUDGET)
                self.assertEqual(complete_record, delta <= 0)
                decoded = json.loads(encoded)
                if complete_record:
                    self.assertEqual(decoded, record)
                else:
                    self.assertTrue(decoded['completed'])
                    self.assertFalse(decoded['telemetryComplete'])
                    self.assertTrue(decoded['evidenceTruncated'])
                    self.assertEqual(decoded['originalCompactUtf8Bytes'], EVIDENCE_BUDGET + 1)

    def test_evidence_overflow_uses_utf8_bytes_and_retains_terminal_unknown(self):
        record = {'completed': False, 'telemetryComplete': True, 'samples': [], 'queries': [],
                  'errors': ['error' * 1000] * 1000, 'payload': 'é' * (EVIDENCE_BUDGET // 2),
                  'host': '界' * 10000, 'source': 'x' * 10000}
        encoded, complete_record = evidence_budget_contract(record)
        self.assertFalse(complete_record)
        self.assertLess(len(encoded), 8192)
        decoded = json.loads(encoded)
        self.assertTrue(decoded['completed'])
        self.assertFalse(decoded['collectionComplete'])
        self.assertFalse(decoded['telemetryComplete'])
        self.assertTrue(decoded['recordDetailsOmitted'])
        self.assertEqual(decoded['admission'], 'not_evaluated')
        self.assertEqual(decoded['applicationJobOwnership'], 'unknown')
        self.assertEqual(decoded['originalErrorCount'], 1000)
        self.assertEqual(len(decoded['host']), 128)
        self.assertFalse(record['completed'], 'saving must not mutate the input into an admission')

    def test_compact_evidence_keeps_all_three_counter_windows(self):
        row = {'instance': 'pid_12345_luid_0x00000000_0x00001234_phys_0_eng_1_engtype_3D',
               'counter': r'\GPU Engine(*)\Utilization Percentage', 'utc': '2026-10-22T23:00:00.0000000Z',
               'value': 0.125, 'rawValue': '0.125', 'status': 1, 'pid': 12345,
               'valuesValidated': True, 'valid': True}
        sample = {'utc': row['utc'], 'counters': [row] * 1450, 'counterDiscovery': [row] * 725}
        record = {'completed': True, 'telemetryComplete': False, 'evidenceTruncated': False,
                  'samples': [sample] * 3, 'errors': ['ownership unknown']}
        self.assertGreater(len(json.dumps(record, indent=16).encode('utf-8')), EVIDENCE_BUDGET)
        encoded, complete_record = evidence_budget_contract(record)
        self.assertTrue(complete_record)
        self.assertEqual(json.loads(encoded), record)
        self.assertEqual([len(x['counters']) for x in json.loads(encoded)['samples']], [1450] * 3)

    def test_collector_save_sites_stop_on_overflow_after_terminal_summary(self):
        script = SCRIPT.read_text(encoding="utf-8")
        save = script.split('function Save-Record {', 1)[1].split('function Smi(', 1)[0]
        self.assertIn('ConvertTo-Json -Depth 16 -Compress', save)
        self.assertIn('[Text.Encoding]::UTF8.GetByteCount($json)', save)
        self.assertIn('$withinBudget = ($originalBytes -le 4MB)', save)
        self.assertIn('completed = $true; collectionComplete = $false; telemetryComplete = $false', save)
        self.assertIn('evidenceTruncated = $true; recordDetailsOmitted = $true', save)
        self.assertIn("terminalReason = 'evidence_budget_exceeded'", save)
        self.assertIn('Select-Object -First 3', save)
        self.assertIn('[Math]::Min(128, $text.Length)', script)
        self.assertIn('[IO.File]::WriteAllText($pending, $json, [Text.UTF8Encoding]::new($false))', save)
        self.assertIn('Move-Item -LiteralPath $pending', save)
        self.assertNotIn('throw', save)
        self.assertNotIn('Set-Content', save)
        self.assertEqual(script.count('if (-not (Save-Record)) { exit 1 }'), 3)
        self.assertNotRegex(script, r'(?m)^\s*Save-Record\s*$')
        self.assertLess(save.index('Move-Item -LiteralPath $pending'), save.index('return $withinBudget'))

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
