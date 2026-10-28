"""Manual, synthetic MOSS CUDA gate. Actions schedules execution on the selected runner.

Only the fixed libtest cases below may execute. Compilation/listing hides CUDA;
Each numerical arm verifies the pinned physical GPU1 identity.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
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
        'transformer::edit::tests::frozen_edit_strict_f32_sampler_cuda'}, {
        'transformer::edit::tests::frozen_edit_oracle_cuda',
        'transformer::edit::tests::frozen_edit_strict_f32_sampler_cuda'}),
    ('layout', MOSS, ['transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts', '--exact'], {
        'transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts'}, {
        'transformer::attention_layout_tests::production_attention_cuda_accepts_ordinary_and_dmd_layouts'}),
    ('encoder', FLUX, ['vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda', '--exact'], {
        'vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda'}, {
        'vae::moss_edit_encoder_tests::frozen_moss_edit_encoder_cuda'}),
)
MAX_LOG = 16 * 1024 * 1024
GPU_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'
GPU_PCI = '00000000:C1:00.0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_identity(text):
    rows = [line.strip().split(',') for line in text.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 3, 'expected exactly one physical GPU1 identity')
    index, uuid, pci = [field.strip() for field in rows[0]]
    require(index == '1' and uuid == GPU_UUID and pci.upper() == GPU_PCI, 'physical GPU1 identity mismatch')
    return {'index': 1, 'uuid': uuid, 'pciBusId': pci}


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
                # Completed Cargo descendants may be finalized through this retained Job.
                if finalize_compile_descendants:
                    require(observed < started + seconds, 'owned command deadline exceeded during finalization drain')
                    if observed >= drain_deadline or attempt == 20:
                        if members:
                            receipt['jobDrainAnomaly'] = 'owned descendants survived root exit drain'
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
                try:
                    job.close()
                    receipt['jobCloseSucceeded'] = True
                except BaseException as error:
                    receipt['jobCloseSucceeded'] = False
                    receipt['jobCloseError'] = str(error)
                    raise
                finally:
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


def source_snapshot(env):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8', timeout=30)
    require(git('rev-parse', 'HEAD').strip() == env['GITHUB_SHA'], 'source revision changed')
    require(not git('status', '--porcelain').strip(), 'source checkout changed')
    return {name: digest(ROOT / name) for name in git('ls-files', '-z').split('\0') if name}


def execute(env, out, *, runner=run_owned, source_check=None):
    source_check = source_check or (lambda: source_snapshot(env))
    source_before = source_check()
    (out / 'source.json').write_text(json.dumps(source_before, indent=2), encoding='utf-8')
    env = env.copy()
    # Cargo's empty overrides disable config wrappers too; do not share a compiler service across owned Jobs.
    env.update(RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='', CARGO_BUILD_JOBS='4')
    for name in ('hf', 'hub', 'tmp'):
        (out / name).mkdir()
    env.update(CUDA_VISIBLE_DEVICES='-1', CUDA_COMPUTE_CAP='120', HF_HOME=str(out / 'hf'),
               HF_HUB_CACHE=str(out / 'hub'), HUGGINGFACE_HUB_CACHE=str(out / 'hub'), HF_HUB_OFFLINE='1',
               TMP=str(out / 'tmp'), TEMP=str(out / 'tmp'), TMPDIR=str(out / 'tmp'), RUST_TEST_THREADS='1')
    target = Path(env['CARGO_TARGET_DIR'])
    command = ['cargo', 'test', '--locked', '--lib', '-p', MOSS, '-p', FLUX, '--features', 'cuda', '--no-run', '--message-format=json']
    code, output = runner(command, env, out, 'compile', 3600, finalize_compile_descendants=True)
    require(code == 0, 'test compilation failed')
    binaries = prepared_binaries(output, target)
    (out / 'prepared.json').write_text(json.dumps(binaries, indent=2), encoding='utf-8')
    # These libtest listings execute no test and keep CUDA hidden. Coverage cannot silently shrink.
    for label, package, filters, expected, expected_ignored in ARMS:
        binary = checked_binary(binaries[package], target)
        for ignored in (False, True):
            args = [binary, *filters, '--list'] + (['--ignored'] if ignored else [])
            args[0] = checked_binary(binaries[package], target)
            code, output = runner(args, env, out, label + ('-ignored' if ignored else '-list'), 60)
            require(code == 0 and listed_tests(output) == (expected_ignored if ignored else expected), 'missing/extra/ignored numerical cases')
    results = []
    for label, package, filters, expected, expected_ignored in ARMS:
        require(source_check() == source_before, 'source changed before numerical invocation')
        binary = checked_binary(binaries[package], target)
        # --id=1 restricts this query to the permitted physical device; never query GPU0.
        code, output = runner(['nvidia-smi', '--id=1', '--query-gpu=index,uuid,pci.bus_id',
                               '--format=csv,noheader,nounits'], env, out, label + '-device', 30)
        require(code == 0, 'physical GPU1 identity query failed')
        identity = device_identity(output)
        binary = checked_binary(binaries[package], target)
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=GPU_UUID, CUDA_DEVICE_ORDER='PCI_BUS_ID', SCENEWORKS_GPU_ID='1', REQUIRE_SM120='1', CARGO_NET_OFFLINE='true')
        execution_flags = ['--ignored']
        code, output = runner([binary, *filters, *execution_flags, '--test-threads', '1', '--nocapture'], gpu_env, out, label + '-tests', 600)
        results.append({'arm': label, 'binarySha256': binaries[package]['sha256'], 'expectedTests': sorted(expected), 'declaredIgnoredTests': sorted(expected_ignored),
                        'returncode': code, 'passed': successful_cases(code, output, expected),
                        'device': identity})
        (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        checked_binary(binaries[package], target)
        require(source_check() == source_before, 'source changed during numerical invocation')
        require(results[-1]['passed'], 'numerical arm failed')
    require(all(row['passed'] for row in results), 'one or more numerical arms failed')
    return results


def main():
    require(os.name == 'nt', 'Windows-only numerical transport')
    env = os.environ.copy()
    require(env['GITHUB_REPOSITORY'] == 'SceneWorks/inference' and env['GITHUB_EVENT_NAME'] == 'workflow_dispatch', 'manual inference workflow required')
    require(re.fullmatch('[0-9a-f]{40}', env['GITHUB_SHA']), 'exact source revision required')
    require(env['GITHUB_RUN_ID'].isdigit() and int(env['GITHUB_RUN_ATTEMPT']) > 0, 'invalid Actions run identity')
    out = Path(env['RUNNER_TEMP']).resolve() / f"moss-edit-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    out.mkdir(exist_ok=False)
    receipt = {'source': env['GITHUB_SHA'], 'runId': env['GITHUB_RUN_ID'], 'attempt': env['GITHUB_RUN_ATTEMPT'], 'completed': False, 'passed': False}
    try:
        receipt['results'] = execute(env, out)
        receipt['passed'] = True
    except BaseException as error:
        receipt['error'] = str(error)
        raise
    finally:
        receipt['completed'] = True
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
