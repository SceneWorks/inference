"""Manual, synthetic MOSS CUDA gate. Reviewed reservation != inferred GPU idle.

Only the fixed libtest cases below may execute. Compilation/listing hides CUDA;
GPU execution requires fresh collector and final identity records for each arm.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.media_campaign_windows import WindowsJob

ROOT = Path(__file__).resolve().parents[2]
MOSS = 'candle-gen-moss-image'
FLUX = 'candle-gen-flux2'
ARMS = (
    ('edit', MOSS, ['transformer::edit::tests::frozen_edit_', '--skip', '_cpu'], {
        'transformer::edit::tests::frozen_edit_oracle_cuda',
        'transformer::edit::tests::frozen_edit_strict_f32_sampler_cuda'}),
    ('layout', MOSS, ['transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts', '--exact'], {
        'transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts'}),
    ('encoder', FLUX, ['vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda', '--exact'], {
        'vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda'}),
)
COUNTERS = {'GPU Engine(*)\\Utilization Percentage', 'GPU Process Memory(*)\\Dedicated Usage',
            'GPU Process Memory(*)\\Shared Usage', 'GPU Adapter Memory(*)\\Dedicated Usage'}
MAX_LOG = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value):
    require(isinstance(value, str) and value.endswith('Z'), 'UTC timestamp required')
    return dt.datetime.fromisoformat(value[:-1] + '+00:00')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, limit=4 * 1024 * 1024):
    require(path.stat().st_size <= limit, 'JSON evidence exceeded bound')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_fields)


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def fields(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected.split()), 'unexpected reservation schema')


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'unknown/nonfinite numeric telemetry')
    return value


def validate_reservation(lease, env, now):
    fields(lease, 'schemaVersion reservationId source actor host physicalGpu uuid pci driver counterLuid reviewedEvidence ownerIdleConfirmedUtc validFromUtc expiresUtc testTimeoutSeconds processes')
    require(lease['schemaVersion'] == 1 and lease['physicalGpu'] == 1, 'unsupported reservation/device')
    require(re.fullmatch(r'[a-zA-Z0-9-]{1,80}', lease['reservationId']), 'invalid reservation ID')
    require(re.fullmatch('[0-9a-f]{40}', lease['source']) and lease['source'] == env['GITHUB_SHA'], 'source mismatch')
    require(env['GITHUB_EVENT_NAME'] == 'workflow_dispatch' and env['GITHUB_RUN_ATTEMPT'] == '1', 'manual first attempt required')
    require(lease['actor'] == env['GITHUB_ACTOR'] and bool(lease['actor']), 'dispatcher mismatch')
    require(lease['host'].casefold() == env['COMPUTERNAME'].casefold() == 'michael-trx50', 'host mismatch')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]+', lease['uuid']), 'invalid GPU UUID')
    require(re.fullmatch(r'00000000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', lease['pci']), 'invalid PCI address')
    require(re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}', lease['counterLuid']), 'invalid LUID')
    require(isinstance(lease['driver'], str) and lease['driver'], 'driver unavailable')
    fields(lease['reviewedEvidence'], 'run sha256 completedUtc')
    require(re.fullmatch(r'[0-9]+', str(lease['reviewedEvidence']['run'])) and re.fullmatch('[0-9a-f]{64}', lease['reviewedEvidence']['sha256']), 'reviewed evidence provenance required')
    start, end = timestamp(lease['validFromUtc']), timestamp(lease['expiresUtc'])
    require(timestamp(lease['ownerIdleConfirmedUtc']) <= start and timestamp(lease['reviewedEvidence']['completedUtc']) <= start <= now < end, 'reservation not current')
    require(type(lease['testTimeoutSeconds']) is int and 0 < lease['testTimeoutSeconds'] <= 7200, 'invalid bounded execution budget')
    require(isinstance(lease['processes'], list) and len(lease['processes']) <= 256, 'process reservation bound')
    seen = set()
    for row in lease['processes']:
        fields(row, 'pid name createdUtc contextType path engineTypes allowedActiveEngineTypes')
        require(type(row['pid']) is int and row['pid'] > 0 and row['pid'] not in seen, 'duplicate/invalid reserved PID')
        seen.add(row['pid'])
        require(isinstance(row['name'], str) and row['name'], 'reserved name unavailable')
        require(timestamp(row['createdUtc']) <= start, 'future process identity')
        require(row['contextType'] in (None, 'C', 'G', 'C+G', 'M', 'M+C', 'O'), 'unknown reserved context')
        require(row['path'] is None or isinstance(row['path'], str) and row['path'], 'invalid optional path')
        for field in ('engineTypes', 'allowedActiveEngineTypes'):
            require(isinstance(row[field], list) and len(row[field]) <= 64 and len(set(row[field])) == len(row[field]), 'invalid engine classification')
            require(all(isinstance(x, str) and re.fullmatch('[a-z0-9_ -]{1,80}', x) for x in row[field]), 'invalid engine type')
        require(set(row['allowedActiveEngineTypes']) <= set(row['engineTypes']), 'active engine not reviewed')
        # This dedicated desktop-coexistence gate never grants foreign compute activity.
        require(set(row['allowedActiveEngineTypes']) <= {'3d', 'copy'}, 'only reviewed 3d/copy background activity is supported')
    return {row['pid']: row for row in lease['processes']}


def device_matches(device, lease):
    require(str(device['index']).strip() == '1' and device['driverModel'].strip() == 'WDDM', 'wrong physical device/driver model')
    for field in ('uuid', 'pci', 'driver'):
        require(device[field].strip().casefold() == lease[field].casefold(), 'device identity changed: ' + field)
    numbers = device['measurements']
    require(numbers['status'] == 'valid', 'unknown device measurement')
    total, used, utilization = [finite(numbers[k]) for k in ('totalMiB', 'usedMiB', 'utilization')]
    require(total > 0 and 0 <= used <= total and 0 <= utilization <= 100, 'invalid device measurement')
    for key in ('totalMiB', 'usedMiB', 'utilization'):
        require(float(device[key]) == numbers[key], 'raw/parsed measurement mismatch')


def census_map(rows):
    require(isinstance(rows, list) and len(rows) <= 256, 'census bound')
    result = {}
    for row in rows:
        require(type(row['pid']) is int and row['pid'] > 0 and row['pid'] not in result, 'duplicate/invalid census PID')
        require(row['type'] in ('C', 'G', 'C+G', 'M', 'M+C', 'O'), 'unknown census context')
        result[row['pid']] = row['type']
    return result


def identities_match(rows, census, approved):
    require(isinstance(rows, list) and len(rows) <= 256, 'identity bound')
    result = {}
    for row in rows:
        pid = row['pid']
        require(type(pid) is int and pid in approved and pid not in result and row['status'] == 'resolved', 'new/unresolved process incarnation')
        expected = approved[pid]
        require(row['name'] == expected['name'] and row['createdUtc'] == expected['createdUtc'], 'process incarnation changed')
        require(census.get(pid) == expected['contextType'], 'process context changed')
        require(type(row['pathAvailable']) is bool and row['pathAvailable'] == bool(row['path']), 'path metadata inconsistent')
        if row['pathAvailable'] and expected['path']:
            require(row['path'].casefold() == expected['path'].casefold(), 'known process path changed')
        result[pid] = row
    require(set(census) <= set(result), 'census process lacks identity')
    return result


def record_header(record, lease, env, earliest, now):
    require(record['completed'] is True and record['evidenceTruncated'] is False and record['errors'] == [], 'incomplete/failed collector')
    require(record['queries'] and all(type(q['exitCode']) is int and q['exitCode'] == 0 for q in record['queries']), 'failed/missing metadata query')
    require(record['admission'] == 'not_evaluated' and record['applicationJobOwnership'] == 'unknown', 'collector cannot grant admission')
    require(record['physicalGpu'] == 1 and record['source'] == lease['source'] and record['host'].casefold() == lease['host'].casefold(), 'foreign collector record')
    require(str(record['run']) == env['GITHUB_RUN_ID'] and str(record['attempt']) == '1', 'foreign collector run')
    require(earliest <= timestamp(record['startedUtc']) <= timestamp(record['completedUtc']) <= now, 'stale/future collector')
    device_matches(record['identity'], lease)


def validate_evidence(record, lease, env, earliest, now):
    approved = validate_reservation(lease, env, now)
    record_header(record, lease, env, earliest, now)
    require(record['collectionComplete'] is True and record['telemetryComplete'] is True, 'mandatory telemetry incomplete')
    require(record['mapping']['status'] == 'mapped' and record['mapping']['counterLuid'] == lease['counterLuid'], 'mapping changed')
    require(len(record['samples']) == 3, 'three complete windows required')
    last_end = timestamp(record['startedUtc'])
    for sample in record['samples']:
        device_matches(sample['device'], lease)
        require(sample['device']['measurements']['totalMiB'] == record['identity']['measurements']['totalMiB'], 'device capacity changed')
        require(last_end <= timestamp(sample['windowStartedUtc']) <= timestamp(sample['windowEndedUtc']) <= timestamp(record['completedUtc']), 'window ordering invalid')
        last_end = timestamp(sample['windowEndedUtc'])
        census = census_map(sample['processes'])
        require(census == census_map(sample['processesAfter']), 'census changed within window')
        before = identities_match(sample['identitiesBefore'], census, approved)
        after = identities_match(sample['identitiesAfter'], census, approved)
        require(set(before) == set(after), 'identity cohort changed')
        for pid in before:
            if before[pid]['pathAvailable'] and after[pid]['pathAvailable']:
                require(before[pid]['path'].casefold() == after[pid]['path'].casefold(), 'known path changed within window')
        categories = set()
        require(0 < len(sample['counters']) <= 8192, 'counter bound')
        for row in sample['counters']:
            category = row['counter'].lstrip('\\')
            require(category in COUNTERS, 'unknown counter category')
            categories.add(category)
            require(type(row['status']) is int and row['status'] in (0, 1) and row['valid'] is True and row['valuesValidated'] is True, 'invalid counter status')
            require(timestamp(sample['windowStartedUtc']) <= timestamp(row['utc']) <= timestamp(sample['windowEndedUtc']), 'stale counter sample')
            value = finite(row['value'])
            require(value >= 0 and float(row['rawValue']) == value, 'unknown/inconsistent counter value')
            require(re.search(r'(^|_)' + re.escape(lease['counterLuid']) + r'(_|$)', row['instance']), 'foreign counter LUID')
            if category.startswith('GPU Adapter Memory'):
                require(row['pid'] is None, 'adapter counter has process PID')
                continue
            match = re.search(r'(^|_)pid_(\d+)(_|$)', row['instance'])
            require(match and int(match[2]) == row['pid'] and row['pid'] in before, 'counter process lacks pre-window identity')
            if category.startswith('GPU Engine'):
                engine = re.search(r'_engtype_(.+)$', row['instance'])
                require(engine and engine[1] in approved[row['pid']]['engineTypes'], 'new/unreviewed engine')
                require(value == 0 or engine[1] in approved[row['pid']]['allowedActiveEngineTypes'], 'unreviewed active engine')
        require(categories == COUNTERS, 'missing mapped counter category')
    return record['samples'][-1]


def validate_final(record, previous, lease, env, now):
    approved = validate_reservation(lease, env, now)
    record_header(record, lease, env, timestamp(previous['completedUtc']), now)
    require(record['snapshotKind'] == 'identity-census-only' and record['finalSnapshotComplete'] is True, 'final identity refresh incomplete')
    require(record['identity']['measurements']['totalMiB'] == previous['identity']['measurements']['totalMiB'], 'capacity changed after collection')
    census = census_map(record['processes'])
    require(census == census_map(previous['samples'][-1]['processesAfter']), 'census changed since counter window')
    rows = identities_match(record['identities'], census, approved)
    prior = {r['pid']: r for r in previous['samples'][-1]['identitiesAfter']}
    require(set(rows) == set(prior), 'final identity cohort changed')
    for pid, row in rows.items():
        if row['pathAvailable'] and prior[pid]['pathAvailable']:
            require(row['path'].casefold() == prior[pid]['path'].casefold(), 'known path changed after counters')


def run_owned(args, env, out, label, seconds, *, finalize_compile_descendants=False, job_type=WindowsJob, popen=subprocess.Popen):
    """Spawn suspended; only the retained Job/child handles may be terminated."""
    require(seconds > 0, 'execution deadline expired')
    require(not finalize_compile_descendants or label == 'compile', 'descendant finalization is compile-only')
    paths = [out / (label + suffix) for suffix in ('.stdout.log', '.stderr.log')]
    with paths[0].open('wb') as stdout, paths[1].open('wb') as stderr:
        job, child = job_type(), None
        receipt = {'argv': args, 'startedUtc': utc().isoformat(), 'assigned': False,
                   'finalizeCompileDescendants': finalize_compile_descendants}
        started = time.monotonic()
        try:
            child = popen(args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=0x00000004)
            try:
                job.assign_and_resume(child)
                receipt['assigned'] = True
                receipt['pid'] = child.pid
            except BaseException:
                child.kill()  # exact still-suspended Popen handle; no PID/name targeting
                child.wait(timeout=10)
                raise
            while child.poll() is None:
                require(time.monotonic() - started < seconds, 'owned command deadline exceeded')
                require(all(path.stat().st_size <= MAX_LOG for path in paths), 'owned output exceeded bound')
                time.sleep(0.05)
            receipt['returncode'] = child.returncode
            receipt['rootExitObservedUtc'] = utc().isoformat()
            # Root exit alone does not establish that the whole owned Job has drained.
            # Permit natural completion briefly, without extending the command deadline.
            drain_started = time.monotonic()
            drain_deadline = min(started + seconds, drain_started + 1.0)
            receipt['jobDrainObservations'] = []
            for attempt in range(21):
                members = sorted(job.members())
                observed = time.monotonic()
                receipt['jobDrainObservations'].append({
                    'elapsedSeconds': round(observed - started, 6), 'pids': members,
                })
                # Cargo may leave owned descendants after a completed build. Only this
                # compile policy may finalize them through the retained Job in finally;
                # that cleanup must succeed before Python can return the result.
                if finalize_compile_descendants:
                    require(observed < started + seconds, 'owned command deadline exceeded during compile drain')
                    if observed >= drain_deadline or attempt == 20:
                        break
                require(observed <= drain_deadline, 'owned Job drain deadline exceeded')
                if not members:
                    break
                require(observed < drain_deadline and attempt < 20, 'owned descendants survived root exit drain')
                require(all(path.stat().st_size <= MAX_LOG for path in paths), 'owned output exceeded bound')
                time.sleep(min(0.05, drain_deadline - observed))
            require(all(path.stat().st_size <= MAX_LOG for path in paths), 'owned output exceeded bound')
        except BaseException as error:
            receipt['error'] = str(error)
            raise
        finally:
            try:
                receipt['preCleanupOwnedPids'] = sorted(job.members())
                if child is not None and receipt['preCleanupOwnedPids']:
                    job.terminate_and_reap(child, 10)
                receipt['remainingOwnedPids'] = sorted(job.members())
                require(not receipt['remainingOwnedPids'], 'owned cleanup incomplete')
                receipt['cleanupCompletedUtc'] = utc().isoformat()
                require(all(path.stat().st_size <= MAX_LOG for path in paths), 'owned output exceeded bound after cleanup')
            except BaseException as error:
                receipt['cleanupError'] = str(error)
                raise
            finally:
                job.close()
                receipt['finishedUtc'] = utc().isoformat()
                (out / (label + '.owned.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    # No retained descendant can append to the accepted compiler output snapshot.
    return child.returncode, paths[0].read_text(encoding='utf-8', errors='replace')


def prepared_binaries(output, target):
    found = {}
    for line in output.splitlines():
        row = json.loads(line)
        if row.get('reason') != 'compiler-artifact' or not row.get('executable'):
            continue
        name = row['target']['name'].replace('_', '-')
        require(name in (MOSS, FLUX) and row['target']['kind'] == ['lib'] and row['profile']['test'] is True, 'unexpected executable target')
        path = Path(row['executable']).resolve(strict=True)
        require(path.is_relative_to(target.resolve()) and path.suffix.lower() == '.exe' and name not in found, 'foreign/duplicate test binary')
        found[name] = {'path': str(path), 'sha256': digest(path)}
    require(set(found) == {MOSS, FLUX}, 'missing required prepared binary')
    return found


def checked_binary(entry, target):
    path = Path(entry['path']).resolve(strict=True)
    require(path.is_relative_to(target.resolve()) and digest(path) == entry['sha256'], 'prepared binary changed')
    return str(path)


def listed_tests(text):
    lines = text.splitlines()
    names = [line[:-6] for line in lines if line.endswith(': test')]
    require(len(set(names)) == len(names), 'duplicate libtest names')
    return set(names)


def successful_cases(code, text, expected):
    summaries = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    return code == 0 and summaries == [('ok', str(len(expected)), '0', '0')]


def execute(lease, env, out, *, runner=run_owned, now=utc):
    validate_reservation(lease, env, now())
    env = env.copy()
    env.pop('MOSS_CUDA_ADMISSION', None)
    # Cargo's empty overrides disable config wrappers too; do not share a compiler service across owned Jobs.
    env.update(RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='')
    for name in ('hf', 'hub', 'tmp'):
        (out / name).mkdir()
    env.update(CUDA_VISIBLE_DEVICES='-1', CUDA_COMPUTE_CAP='120', HF_HOME=str(out / 'hf'),
               HF_HUB_CACHE=str(out / 'hub'), HUGGINGFACE_HUB_CACHE=str(out / 'hub'), HF_HUB_OFFLINE='1',
               TMP=str(out / 'tmp'), TEMP=str(out / 'tmp'), TMPDIR=str(out / 'tmp'), RUST_TEST_THREADS='1')
    target = Path(env['CARGO_TARGET_DIR'])
    remaining = lambda: (timestamp(lease['expiresUtc']) - now()).total_seconds()
    command = ['cargo', 'test', '--locked', '--lib', '-p', MOSS, '-p', FLUX, '--features', 'cuda', '--no-run', '--message-format=json']
    code, output = runner(command, env, out, 'compile', remaining(), finalize_compile_descendants=True)
    require(code == 0, 'test compilation failed')
    binaries = prepared_binaries(output, target)
    (out / 'prepared.json').write_text(json.dumps(binaries, indent=2), encoding='utf-8')
    # These libtest listings execute no test and keep CUDA hidden. Coverage cannot silently shrink.
    for label, package, filters, expected in ARMS:
        binary = checked_binary(binaries[package], target)
        for ignored in (False, True):
            args = [binary, *filters, '--list'] + (['--ignored'] if ignored else [])
            args[0] = checked_binary(binaries[package], target)
            code, output = runner(args, env, out, label + ('-ignored' if ignored else '-list'), remaining())
            require(code == 0 and listed_tests(output) == (set() if ignored else expected), 'missing/extra/ignored numerical cases')
    results = []
    collector = str(ROOT / 'scripts/ci/windows-gpu1-diagnostic.ps1')
    for label, package, filters, expected in ARMS:
        validate_reservation(lease, env, now())
        binary = checked_binary(binaries[package], target)
        phase_start = now()
        evidence_dir, final_dir = out / (label + '-evidence'), out / (label + '-final')
        args = ['powershell', '-NoProfile', '-NonInteractive', '-File', collector, '-OutputDirectory', str(evidence_dir)]
        code, _ = runner(args, env, out, label + '-collector', min(300, remaining()))
        require(code == 0, 'fresh collector failed')
        evidence = read_json(evidence_dir / 'evidence.json')
        validate_evidence(evidence, lease, env, phase_start, now())
        # Verify binaries before the final refresh; no compiler or test listing follows it.
        binary = checked_binary(binaries[package], target)
        args = ['powershell', '-NoProfile', '-NonInteractive', '-File', collector, '-OutputDirectory', str(final_dir), '-PreviousEvidencePath', str(evidence_dir / 'evidence.json')]
        code, _ = runner(args, env, out, label + '-refresh', min(60, remaining()))
        require(code == 0, 'final identity refresh failed')
        final = read_json(final_dir / 'evidence.json')
        validate_final(final, evidence, lease, env, now())
        binary = checked_binary(binaries[package], target)
        seconds = lease['testTimeoutSeconds']
        require(remaining() >= seconds, 'reservation does not cover invocation budget')
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=lease['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID', SCENEWORKS_GPU_ID='1', REQUIRE_SM120='1', CARGO_NET_OFFLINE='true')
        code, output = runner([binary, *filters, '--test-threads', '1', '--nocapture'], gpu_env, out, label + '-tests', seconds)
        results.append({'arm': label, 'binarySha256': binaries[package]['sha256'], 'expectedTests': sorted(expected),
                        'returncode': code, 'passed': successful_cases(code, output, expected),
                        'collectorSha256': digest(evidence_dir / 'evidence.json'), 'finalSnapshotSha256': digest(final_dir / 'evidence.json')})
        (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    require(all(row['passed'] for row in results), 'one or more numerical arms failed')
    return results


def main():
    require(os.name == 'nt', 'Windows-only numerical transport')
    env = os.environ.copy()
    raw = env['MOSS_CUDA_ADMISSION']
    require(len(raw.encode('utf-8')) <= 48 * 1024, 'reservation JSON exceeded bound')
    lease = json.loads(raw, object_pairs_hook=unique_fields)
    validate_reservation(lease, env, utc())
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, encoding='utf-8').strip() == lease['source'], 'checkout mismatch')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, encoding='utf-8').strip(), 'dirty source')
    out = Path(env['RUNNER_TEMP']).resolve() / f"moss-edit-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    out.mkdir(exist_ok=False)
    receipt = {'source': lease['source'], 'reservation': lease, 'completed': False, 'passed': False}
    try:
        receipt['results'] = execute(lease, env, out)
        receipt['passed'] = True
    except BaseException as error:
        receipt['error'] = str(error)
        raise
    finally:
        receipt['completed'] = True
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
