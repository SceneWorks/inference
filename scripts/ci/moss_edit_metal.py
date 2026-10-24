"""Run the exact synthetic Metal oracle directly on an Actions NAX runner.

Compilation/listing do not allocate a Metal device. Source, runtime, binary and
libtest membership remain bound across preparation and numerical execution.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TEST = 'transformer::edit_fixture_tests::frozen_edit_oracle_on_metal'
MAX_LOG = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)

def utc(): return dt.datetime.now(dt.timezone.utc)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result

def read(path):
    require(path.stat().st_size <= MAX_LOG, 'oversized JSON evidence')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_fields)
def operation_timeout(maximum, deadline=None):
    if deadline is None: return maximum
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'absolute preparation operation deadline expired')
    return min(maximum, remaining)

def metadata(command, deadline=None):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=operation_timeout(30, deadline), encoding='utf-8')
    operation_timeout(30, deadline)
    require(result.returncode == 0, 'metadata command failed: ' + command[0])
    require(len(result.stdout.encode()) <= MAX_LOG, 'oversized metadata output')
    return result.stdout

def inventory(deadline=None):
    require(metadata(['git', 'rev-parse', 'HEAD'], deadline).strip() == os.environ['GITHUB_SHA'], 'source revision mismatch')
    require(not metadata(['git', 'status', '--porcelain'], deadline).strip(), 'source checkout is dirty')
    files = metadata(['git', 'ls-files', '-z'], deadline).split('\0')
    result = {}
    for name in files:
        operation_timeout(30, deadline)
        if name: result[name] = sha(ROOT / name)
    operation_timeout(30, deadline)
    return result

def paths(create=False):
    require(os.environ['GITHUB_REPOSITORY'] == 'SceneWorks/inference', 'wrong repository')
    require(os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch', 'manual workflow required')
    require(re.fullmatch('[0-9a-f]{40}', os.environ['GITHUB_SHA']), 'exact source required')
    require(os.environ['GITHUB_RUN_ID'].isdigit() and int(os.environ['GITHUB_RUN_ATTEMPT']) > 0, 'invalid Actions run identity')
    out = Path(os.environ['RUNNER_TEMP']).resolve() / f"moss-edit-metal-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    if create: out.mkdir(mode=0o700, exist_ok=False)
    return out

def child_env(out):
    env = {key: value for key, value in os.environ.items() if not any(word in key.upper() for word in ('TOKEN', 'API_KEY', 'SECRET', 'PASSWORD'))}
    env.update(CARGO_TARGET_DIR=str(out / 'target'), CARGO_BUILD_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2', RUSTUP_TOOLCHAIN='1.96.0', RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='', CARGO_NET_OFFLINE='true', CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0', CARGO_PROFILE_TEST_DEBUG='0', CARGO_PROFILE_DEV_STRIP='none', CARGO_PROFILE_TEST_STRIP='none', RUSTFLAGS='-C strip=none', RUST_TEST_THREADS='1', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', HF_HOME=str(out / 'hf'), HF_HUB_CACHE=str(out / 'hub'), HUGGINGFACE_HUB_CACHE=str(out / 'hub-legacy'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', SCENEWORKS_WHISPER_SMOKE_CACHE=str(out / 'whisper'), SCENEWORKS_WHISPER_SMOKE_FRESH='0', TMPDIR=str(out / 'tmp'), TMP=str(out / 'tmp'), TEMP=str(out / 'tmp'), SCENEWORKS_API_URL='http://127.0.0.1:1', SCENEWORKS_ACCESS_TOKEN='source-fixture-deny', RUSTUP_AUTO_INSTALL='0', PYTHONDONTWRITEBYTECODE='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='', MACOSX_DEPLOYMENT_TARGET='26.2')
    return env

def runtime_identity(env, deadline=None):
    rustc = Path(metadata(['rustup', 'which', '--toolchain', '1.96.0', 'rustc'], deadline).strip()).resolve(strict=True)
    version = subprocess.check_output([str(rustc), '--version'], env=env, text=True, timeout=operation_timeout(30, deadline), encoding='utf-8').strip()
    operation_timeout(30, deadline)
    require(version.startswith('rustc 1.96.0 '), 'wrong native Rust version')
    return {'rustc': str(rustc), 'rustcSha256': sha(rustc), 'version': version, 'python': str(Path(sys.executable).resolve()), 'pythonSha256': sha(Path(sys.executable).resolve())}


def group_rows(pgid, deadline):
    """Read only our spawned process group, never a host-wide process census."""
    result = subprocess.run(['ps', '-g', str(pgid), '-o', 'pid=,pgid=,stat=,lstart=,comm='],
                            capture_output=True, text=True, encoding='utf-8',
                            timeout=operation_timeout(2, deadline))
    require(result.returncode == 0 or (result.returncode == 1 and not result.stdout), 'owned group query failed')
    operation_timeout(2, deadline)
    require(len(result.stdout.encode()) <= MAX_LOG, 'owned group metadata exceeded bound')
    records = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 8)
        require(len(fields) == 9 and int(fields[1]) == pgid, 'invalid owned group identity')
        records[int(fields[0])] = {'state': fields[2], 'birth': ' '.join(fields[3:8]), 'comm': fields[8]}
    return records


def run_owned(command, env, out, label, timeout):
    end = time.monotonic() + timeout
    receipt = {'command': command, 'startedUtc': utc().isoformat(), 'observed': {}, 'sentSignals': []}
    process = None
    log_path = out / (label + '.log')
    with log_path.open('w', encoding='utf-8') as log:
        def observe(deadline):
            current = group_rows(process.pid, deadline)
            # Do not poll/reap the leader: its retained child identity pins the
            # PID/PGID even after exit, until all descendants have drained.
            require(process.pid in current, 'unreaped owned leader missing')
            for pid, identity in current.items():
                previous = receipt['observed'].get(str(pid))
                require(previous is None or previous['birth'] == identity['birth'], 'owned process identity changed')
                receipt['observed'][str(pid)] = identity
            return current
        def send(pid, identity, sig, deadline):
            current = observe(deadline)
            if pid in current and current[pid]['birth'] == identity['birth'] and not current[pid]['state'].startswith('Z'):
                try: os.kill(pid, sig)
                except ProcessLookupError: pass
                else: receipt['sentSignals'].append({'pid': pid, **identity, 'signal': int(sig)})
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt['pid'] = process.pid
            while True:
                current = observe(end)
                require(log_path.stat().st_size <= MAX_LOG, 'phase log bound exceeded')
                if current[process.pid]['state'].startswith('Z'): break
                operation_timeout(1, end)
                time.sleep(min(.2, max(0, end - time.monotonic())))
        except BaseException as error:
            receipt['error'] = str(error)
            raise
        finally:
            cleanup_end = time.monotonic() + 10
            try:
                if process is not None:
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        current = observe(cleanup_end)
                        send(process.pid, current[process.pid], sig, cleanup_end)
                        for pid, identity in current.items():
                            if pid != process.pid: send(pid, identity, sig, cleanup_end)
                        until = min(cleanup_end, time.monotonic() + 3)
                        while time.monotonic() < until:
                            current = observe(cleanup_end)
                            if set(current) == {process.pid} and current[process.pid]['state'].startswith('Z'): break
                            time.sleep(.05)
                        current = observe(cleanup_end)
                        if set(current) == {process.pid} and current[process.pid]['state'].startswith('Z'): break
                    receipt['remaining'] = {pid: row for pid, row in current.items() if pid != process.pid or not row['state'].startswith('Z')}
                    require(not receipt['remaining'], 'owned cleanup incomplete')
                else:
                    receipt.update(exitCode=None, remaining={})
                require(log_path.stat().st_size <= MAX_LOG, 'phase log bound exceeded after cleanup')
            except BaseException as error:
                receipt['cleanupError'] = str(error)
                raise
            finally:
                try:
                    if process is not None:
                        receipt['exitCode'] = process.wait(timeout=operation_timeout(2, cleanup_end))
                finally:
                    receipt['finishedUtc'] = utc().isoformat()
                    write(out / (label + '-process.json'), receipt)
    require(receipt['exitCode'] == 0, 'owned phase failed: ' + label)
    return log_path.read_text(encoding='utf-8')


def listed_tests(log):
    return re.findall(r'^(.+): test$', log, re.M)


def prepare():
    require(sys.platform == 'darwin', 'Metal requires macOS')
    out = paths(True)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write('evidence=' + str(out) + '\n')
    for name in ('target', 'tmp', 'hf', 'hub', 'hub-legacy', 'whisper'): (out / name).mkdir()
    source = inventory(); write(out / 'source.json', source)
    env = child_env(out)
    runtime = runtime_identity(env); write(out / 'runtime.json', runtime)
    command = ['cargo', 'test', '--locked', '-p', 'mlx-gen-moss-image', '--lib', '--no-run', '--message-format=json']
    log = run_owned(command, env, out, 'compile', 1200)
    binaries = []
    for line in log.splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get('reason') == 'compiler-artifact' and event.get('executable'):
            require(event.get('target', {}).get('name') == 'mlx_gen_moss_image' and event['target']['kind'] == ['lib'] and event.get('profile', {}).get('test') is True, 'unexpected executable target')
            binaries.append(Path(event['executable']).resolve(strict=True))
    require(len(binaries) == 1 and binaries[0].is_relative_to(out / 'target'), 'missing or escaping prepared binary')
    binary = binaries[0]
    binary_hash = sha(binary)
    for ignored in (False, True):
        listed = run_owned([str(binary), TEST, '--exact', '--list'] + (['--ignored'] if ignored else []), env, out, 'list-ignored' if ignored else 'list', 60)
        require(listed_tests(listed) == [TEST], 'missing or extra oracle membership')
        require(sha(binary) == binary_hash, 'prepared binary changed while listing')
    require(runtime_identity(env) == runtime, 'build runtime changed')
    require(inventory() == source, 'source changed during preparation')
    prepared = {'source': os.environ['GITHUB_SHA'], 'runId': int(os.environ['GITHUB_RUN_ID']), 'attempt': int(os.environ['GITHUB_RUN_ATTEMPT']), 'host': platform.node(), 'runner': os.environ['RUNNER_NAME'], 'binary': str(binary), 'binarySha256': binary_hash, 'controllerSha256': sha(Path(__file__)), 'sourceInventorySha256': sha(out / 'source.json'), 'runtimeSha256': sha(out / 'runtime.json'), 'members': [TEST], 'preparedUtc': utc().isoformat()}
    write(out / 'prepared.json', prepared)
    print(json.dumps(prepared))


def execute():
    out = paths()
    result = {'passed': False, 'source': os.environ['GITHUB_SHA']}
    try:
        prepared = read(out / 'prepared.json')
        require(prepared['source'] == os.environ['GITHUB_SHA'] and prepared['runId'] == int(os.environ['GITHUB_RUN_ID']) and prepared['attempt'] == int(os.environ['GITHUB_RUN_ATTEMPT']), 'prepared Actions identity changed')
        require(prepared['host'] == platform.node() and prepared['runner'] == os.environ['RUNNER_NAME'], 'prepared binary moved to another runner')
        require(prepared['members'] == [TEST] and prepared['controllerSha256'] == sha(Path(__file__)), 'prepared test/controller identity changed')
        require(sha(out / 'source.json') == prepared['sourceInventorySha256'] and sha(out / 'runtime.json') == prepared['runtimeSha256'], 'prepared manifests changed')
        source, runtime = read(out / 'source.json'), read(out / 'runtime.json')
        binary = Path(prepared['binary']).resolve(strict=True)
        require(binary.is_relative_to(out / 'target') and sha(binary) == prepared['binarySha256'], 'prepared binary changed')
        env = child_env(out)
        require(inventory() == source and runtime_identity(env) == runtime, 'source/runtime changed before execution')
        env['MOSS_NATIVE_TEST_ADMITTED'] = '1'
        log = run_owned([str(binary), TEST, '--exact', '--ignored', '--nocapture', '--test-threads=1'], env, out, 'oracle', 1200)
        require(re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', log) == [('ok', '1', '0', '0')], 'oracle did not execute exactly one test')
        require(inventory() == source and sha(binary) == prepared['binarySha256'] and runtime_identity(env) == runtime, 'source/binary/runtime changed during oracle')
        result.update(passed=True, binarySha256=prepared['binarySha256'], expectedTests=[TEST], actualPassed=1, scope='synthetic full Metal graph; not public RGB or real weights')
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        write(out / 'result.json', result)


if __name__ == '__main__':
    def interrupted(signum, _frame): raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    {'prepare': prepare, 'execute': execute}[sys.argv[1]]()
