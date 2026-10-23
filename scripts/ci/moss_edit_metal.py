"""Exact synthetic MOSS Metal gate with a separately published execution lease.

Preparation and queueing allocate no Metal device. Root classification travels
through authenticated Actions artifacts; it is never inferred from an idle port.
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
REVIEWER = 'michaeltrefry'
POLL_SECONDS = 30


class AdmissionDeferred(ValueError):
    pass


class CapacityUnavailable(ValueError):
    """Only a valid measured capacity shortage may wait or retry preparation."""
    def __init__(self, predicate, sample):
        super().__init__(predicate)
        self.predicate, self.sample = predicate, sample


COMPILERS = {'rustc', 'rustdoc', 'clippy-driver', 'clang', 'clang++', 'cc', 'c++', 'gcc', 'g++', 'cc1', 'cc1plus', 'cmake', 'ninja', 'make', 'nvcc', 'ld', 'ld64', 'metal', 'metallib'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def utc(): return dt.datetime.now(dt.timezone.utc)
def stamp(value): return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
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

def rows(deadline=None):
    result = []
    for line in metadata(['ps', '-axo', 'pid,ppid,pgid,lstart,comm'], deadline).splitlines()[1:]:
        fields = line.strip().split(None, 8)
        if len(fields) == 9:
            result.append((int(fields[0]), int(fields[1]), int(fields[2]), ' '.join(fields[3:8]), fields[8]))
    return result

def process_records(current):
    return [{'pid': pid, 'parent': parent, 'group': group, 'birth': birth, 'comm': comm}
            for pid, parent, group, birth, comm in current]


def descendants(records, roots):
    by_pid = {row['pid']: row for row in records}
    owned = {row['pid'] for row in roots if by_pid.get(row['pid']) == row}
    while True:
        children = {row['pid'] for row in records if row['parent'] in owned}
        if children <= owned: return owned
        owned |= children


def relevant_processes(records):
    # Executable identity only. Include native test/model launchers before they
    # create an AGX client or TCP listener, plus their currently live descendants.
    pattern = re.compile(r'(python|cargo|rustc|clippy|mlx|candle|sceneworks|runtime[_-]|worker|runner[.]|benchmark|ollama|llama|comfy|pmetal|pytest)', re.I)
    seeds = [row for row in records if pattern.search(Path(row['comm']).name) or '/target/' in row['comm']]
    relevant = descendants(records, seeds)
    return [row for row in records if row['pid'] in relevant]


def job_owner(records, pid):
    by_pid = {row['pid']: row for row in records}
    seen = set()
    while pid in by_pid and pid not in seen:
        seen.add(pid); row = by_pid[pid]
        if Path(row['comm']).name == 'Runner.Worker': return row
        pid = row['parent']
    raise ValueError('controller is not descended from an identified Actions Runner.Worker')


def token_path():
    return Path.home() / '.codex' / 'locks' / 'native-metal-physical-host.lock'


def try_execution_token(lock):
    import fcntl
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def release_execution_token(lock):
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_UN)


def token_identity(path):
    info = path.stat()
    return {'path': str(path.resolve()), 'device': info.st_dev, 'inode': info.st_ino}


def snapshot(deadline=None):
    import plistlib
    forest = plistlib.loads(subprocess.check_output(['/usr/sbin/ioreg', '-a', '-l', '-r', '-c', 'AGXAccelerator'], timeout=operation_timeout(30, deadline)))
    require(isinstance(forest, list) and len(forest) == 1, 'expected one physical Apple GPU')
    device = forest[0]
    identity = {key: device[key] for key in ('IORegistryEntryID', 'IORegistryEntryName', 'IOObjectClass')}
    all_processes = process_records(rows(deadline))
    processes = {row['pid']: (row['birth'], row['comm']) for row in all_processes}
    clients, device_clients = [], []
    def visit(node):
        is_client = any('AGXDeviceUserClient' in str(node.get(key, '')) for key in ('IORegistryEntryName', 'IOObjectClass'))
        if is_client:
            device_clients.append(node['IORegistryEntryID'])
            require('IOUserClientCreator' in node, 'AGX client lacks owner properties')
        if 'IOUserClientCreator' in node:
            owner = node['IOUserClientCreator']
            match = re.fullmatch(r'pid (\d+), (.+)', owner)
            require(match is not None, 'unrecognized GPU client identity')
            pid = int(match.group(1))
            require(pid in processes, 'GPU client lacks a current process identity')
            birth, comm = processes[pid]
            clients.append({'entry_id': node['IORegistryEntryID'], 'pid': pid, 'birth': birth, 'comm': comm, 'owner': owner, 'usage_present': 'AppUsage' in node, 'usage': node.get('AppUsage')})
        for child in node.get('IORegistryEntryChildren', []): visit(child)
    visit(device)
    require(device_clients and set(device_clients) <= {row['entry_id'] for row in clients}, 'incomplete physical GPU client census')
    listeners, current = {}, None
    listing = subprocess.run(['/usr/sbin/lsof', '-nP', '-iTCP', '-sTCP:LISTEN', '-Fpcn'], capture_output=True, text=True, timeout=operation_timeout(30, deadline), encoding='utf-8')
    operation_timeout(30, deadline)
    require(listing.returncode == 0 or (listing.returncode == 1 and not listing.stdout and not listing.stderr), 'listener census failed')
    for line in listing.stdout.splitlines():
        if line.startswith('p'):
            current = int(line[1:]); require(current in processes, 'listener process identity disappeared')
            birth, comm = processes[current]
            listeners.setdefault(current, {'pid': current, 'birth': birth, 'comm': comm, 'endpoints': []})
        elif line.startswith('n'):
            require(current is not None, 'listener lacks a process')
            listeners[current]['endpoints'].append(line[1:])
    for row in listeners.values(): row['endpoints'] = sorted(set(row['endpoints']))
    return {'utc': utc().isoformat(), 'host': platform.node(), 'runner': os.environ['RUNNER_NAME'], 'device': identity, 'device_client_count': len(device_clients), 'clients': sorted(clients, key=lambda row: row['entry_id']), 'listeners': sorted(listeners.values(), key=lambda row: row['pid']), 'processes': relevant_processes(all_processes), 'controllerJobOwner': job_owner(all_processes, os.getpid())}

def inventory(deadline=None):
    require(metadata(['git', 'rev-parse', 'HEAD'], deadline).strip() == os.environ['MOSS_METAL_SOURCE'] == os.environ['GITHUB_SHA'], 'source revision mismatch')
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
    require(os.environ['GITHUB_ACTOR'] == REVIEWER, 'unreviewed dispatcher')
    require(os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch' and os.environ['GITHUB_RUN_ATTEMPT'] == '1', 'manual first attempt required')
    require(re.fullmatch('[0-9a-f]{40}', os.environ['MOSS_METAL_SOURCE']), 'exact source required')
    out = Path(os.environ['RUNNER_TEMP']).resolve() / f"moss-edit-metal-{os.environ['GITHUB_RUN_ID']}-1"
    if create: out.mkdir(mode=0o700, exist_ok=False)
    return out

def child_env(out):
    env = {key: value for key, value in os.environ.items() if not any(word in key.upper() for word in ('TOKEN', 'API_KEY', 'SECRET', 'PASSWORD'))}
    env.update(CARGO_TARGET_DIR=str(out / 'target'), CARGO_BUILD_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4', RUSTUP_TOOLCHAIN='1.96.0', RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='', CARGO_NET_OFFLINE='true', CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0', CARGO_PROFILE_TEST_DEBUG='0', CARGO_PROFILE_DEV_STRIP='none', CARGO_PROFILE_TEST_STRIP='none', RUSTFLAGS='-C strip=none', RUST_TEST_THREADS='1', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', HF_HOME=str(out / 'hf'), HF_HUB_CACHE=str(out / 'hub'), HUGGINGFACE_HUB_CACHE=str(out / 'hub-legacy'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', SCENEWORKS_WHISPER_SMOKE_CACHE=str(out / 'whisper'), SCENEWORKS_WHISPER_SMOKE_FRESH='0', TMPDIR=str(out / 'tmp'), TMP=str(out / 'tmp'), TEMP=str(out / 'tmp'), SCENEWORKS_API_URL='http://127.0.0.1:1', SCENEWORKS_ACCESS_TOKEN='source-fixture-deny', RUSTUP_AUTO_INSTALL='0', PYTHONDONTWRITEBYTECODE='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='', MACOSX_DEPLOYMENT_TARGET='26.2')
    return env

def run_owned(command, env, out, label, timeout, monitor=None, prelaunch=None, absolute_deadline=None):
    receipt = {'command': command, 'startedUtc': utc().isoformat(), 'observed': {}, 'sentSignals': []}
    observed = receipt['observed']
    # Preparation's final ten seconds are reserved for owned cleanup. The end
    # is fixed BEFORE slow prelaunch metadata and never reset after admission.
    end = min(time.monotonic() + timeout, absolute_deadline - 10) if absolute_deadline is not None else None
    if absolute_deadline is not None: receipt.update(absoluteDeadline=absolute_deadline, workDeadline=end)
    with (out / (label + '.log')).open('w', encoding='utf-8') as log:
        try:
            if prelaunch: receipt['prelaunch'] = prelaunch()
            if end is not None: operation_timeout(timeout, end)
        except BaseException as error:
            receipt.update(error=str(error), launched=False, remaining=[], exitCode=None, finishedUtc=utc().isoformat())
            if isinstance(error, CapacityUnavailable): receipt['capacityUnavailable'] = {'predicate': error.predicate, 'sample': error.sample}
            write(out / (label + '-process.json'), receipt)
            raise
        # The ordinary oracle retains its final admission callback directly
        # before Popen; no source scan or capacity query is inserted here.
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['pid'] = process.pid; receipt['launched'] = True
        def observe(limit=None):
            current = rows(limit) if limit is not None else rows()
            owned = {process.pid} if process.poll() is None else set()
            for pid, parent, group, birth, comm in current:
                if observed.get(str(pid), {}).get('birth') == birth: owned.add(pid)
            while True:
                children = {pid for pid, parent, group, birth, comm in current if parent in owned or (owned and group == process.pid)}
                if children <= owned: break
                owned |= children
            living = []
            for pid, parent, group, birth, comm in current:
                if pid in owned:
                    observed[str(pid)] = {'birth': birth, 'comm': comm, 'pgid': group}; living.append((pid, birth))
            return living
        def cleanup():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                living = observe(absolute_deadline)
                if not living: return
                for pid, birth in living:
                    current = rows(absolute_deadline) if absolute_deadline is not None else rows()
                    if any(p == pid and b == birth for p, parent, group, b, comm in current):
                        try: os.kill(pid, sig)
                        except ProcessLookupError: pass
                        else: receipt['sentSignals'].append({'pid': pid, 'birth': birth, 'signal': int(sig)})
                until = time.monotonic() + 3
                if absolute_deadline is not None: until = min(until, absolute_deadline)
                while observe(absolute_deadline) and time.monotonic() < until: process.poll(); time.sleep(max(0, min(.1, until - time.monotonic())))
        if end is None: end = time.monotonic() + timeout
        try:
            while process.poll() is None:
                operation_timeout(timeout, end)
                observe(end if absolute_deadline is not None else None); write(out / (label + '-process.json'), receipt)
                operation_timeout(timeout, end)
                require((out / (label + '.log')).stat().st_size <= MAX_LOG, 'phase log bound exceeded')
                if monitor: monitor(observed)
                operation_timeout(timeout, end)
                time.sleep(max(0, min(.5, end - time.monotonic())))
        except BaseException as error:
            receipt['error'] = str(error)
            if isinstance(error, CapacityUnavailable):
                receipt['capacityUnavailable'] = {'predicate': error.predicate, 'sample': error.sample}
                receipt['capacityInterrupted'] = process.poll() is None
            raise
        finally:
            try:
                cleanup(); receipt['remaining'] = observe(absolute_deadline)
                receipt['exitCode'] = process.wait(timeout=operation_timeout(10, absolute_deadline)) if absolute_deadline is not None else process.wait()
            finally:
                receipt['finishedUtc'] = utc().isoformat(); write(out / (label + '-process.json'), receipt)
            require(not receipt['remaining'], 'owned cleanup incomplete')
    require(not receipt['remaining'] and receipt['exitCode'] == 0, 'owned phase failed: ' + label)
    return (out / (label + '.log')).read_text(encoding='utf-8')

def capacity(out, prelaunch=False, high_since=None, deadline=None):
    cores = os.cpu_count()
    require(cores is not None and cores >= 8, 'four-job build needs a reviewed host with at least eight logical cores')
    memory = metadata(['/usr/bin/memory_pressure', '-Q'], deadline)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', memory)
    require(match is not None, 'unparseable memory capacity')
    active = [{'pid': pid, 'birth': birth, 'comm': comm} for pid, parent, group, birth, comm in rows(deadline) if Path(comm).name in COMPILERS]
    disk = os.statvfs(out)
    sample = {'utc': utc().isoformat(), 'logicalCores': cores, 'actualCompilers': active, 'load1m': os.getloadavg()[0], 'memoryFreePercent': int(match.group(1)), 'memoryPressureRaw': memory, 'diskFreeBytes': disk.f_bavail * disk.f_frsize}
    write(out / 'capacity.json', sample)
    history = out / 'capacity-history.log'
    encoded = json.dumps(sample) + '\n'
    require((history.stat().st_size if history.exists() else 0) + len(encoded.encode('utf-8')) <= MAX_LOG, 'capacity history bound exceeded')
    with history.open('a', encoding='utf-8') as stream: stream.write(encoded)
    # Compiler process count is telemetry, not a proxy for measured CPU capacity.
    if sample['memoryFreePercent'] < 30: raise CapacityUnavailable('memory free percentage below thirty', sample)
    if sample['diskFreeBytes'] < 80 * 1024**3: raise CapacityUnavailable('disk free bytes below eighty GiB', sample)
    if prelaunch and sample['load1m'] > cores - 4: raise CapacityUnavailable('prelaunch CPU capacity exceeded', sample)
    if sample['load1m'] > cores - 2:
        high_since = high_since if high_since is not None else time.monotonic()
        if time.monotonic() - high_since >= 30: raise CapacityUnavailable('CPU capacity exceeded for thirty seconds', sample)
    else: high_since = None
    return high_since


def runtime_identity(env, deadline=None):
    rustc = Path(metadata(['rustup', 'which', '--toolchain', '1.96.0', 'rustc'], deadline).strip()).resolve(strict=True)
    version = subprocess.check_output([str(rustc), '--version'], env=env, text=True, timeout=operation_timeout(30, deadline), encoding='utf-8').strip()
    operation_timeout(30, deadline)
    require(version.startswith('rustc 1.96.0 '), 'wrong native Rust version')
    return {'rustc': str(rustc), 'rustcSha256': sha(rustc), 'version': version, 'python': str(Path(sys.executable).resolve()), 'pythonSha256': sha(Path(sys.executable).resolve())}


def validate_lease(lease):
    fields = 'schemaVersion source actor preparationRunId preparationAttempt host runner device binarySha256 metadataSha256 sourceInventorySha256 classificationUtc validFromUtc expiresUtc exclusiveMetalExecutionAuthorized clients listeners processes runnerNamesOnPhysicalHost computeJobsToDrain'
    require(type(lease) is dict and set(lease) == set(fields.split()) and lease.get('schemaVersion') == 1, 'lease schema mismatch')
    require(lease['actor'] == REVIEWER and lease['exclusiveMetalExecutionAuthorized'] is True, 'reviewed execution authority missing')
    require(type(lease['preparationRunId']) is int and lease['preparationRunId'] > 0 and lease['preparationAttempt'] == 1, 'invalid preparation identity')
    require(re.fullmatch('[0-9a-f]{40}', lease['source']), 'invalid lease source')
    for key in ('binarySha256', 'metadataSha256', 'sourceInventorySha256'):
        require(re.fullmatch('[0-9a-f]{64}', lease[key]), 'invalid lease digest: ' + key)
    require(lease['runner'] in ('nax-macos', 'nax-macos-2') and type(lease['host']) is str and lease['host'], 'invalid physical runner identity')
    start, end = stamp(lease['validFromUtc']), stamp(lease['expiresUtc'])
    require(start.utcoffset() == end.utcoffset() == dt.timedelta(0) and 1200 <= (end - start).total_seconds() <= 7200, 'invalid execution window')
    require(stamp(lease['classificationUtc']) <= start, 'future ownership classification')
    for key, identity in (('clients', 'entry_id'), ('listeners', 'pid')):
        require(type(lease[key]) is list and len(lease[key]) == len({row[identity] for row in lease[key]}), 'duplicate ownership classifications')
        for row in lease[key]:
            require(type(row['mustBeAbsent']) is bool, 'missing competing-owner classification')
            if key == 'clients':
                require(set(row) == {'entry_id', 'mustBeAbsent', 'allowBackgroundActivity'} and type(row['entry_id']) is int and type(row['allowBackgroundActivity']) is bool, 'invalid GPU client rule')
            else:
                require(set(row) == {'pid', 'mustBeAbsent', 'disposition'} and type(row['pid']) is int and row['pid'] > 0 and row['disposition'] in ('explicitly-reviewed-no-compute', 'competing-compute'), 'invalid listener rule')
                require(row['mustBeAbsent'] or row['disposition'] == 'explicitly-reviewed-no-compute', 'idle listener cannot authorize compute')
    require(type(lease['processes']) is list and len(lease['processes']) == len({row['pid'] for row in lease['processes']}), 'duplicate process classification')
    for row in lease['processes']:
        require(set(row) == {'pid', 'disposition'} and type(row['pid']) is int and row['pid'] > 0 and row['disposition'] in ('must-drain', 'reviewed-no-compute'), 'invalid pipeline classification')
    host_runners = lease['runnerNamesOnPhysicalHost']
    require(type(host_runners) is list and lease['runner'] in host_runners and len(host_runners) == len(set(host_runners)) and set(host_runners) <= {'nax-macos', 'nax-macos-2'}, 'unreviewed physical-host runner mapping')
    require(type(lease['computeJobsToDrain']) is list, 'missing cross-repository classification')
    for job in lease['computeJobsToDrain']:
        require(job['repository'] in ('SceneWorks/inference', 'SceneWorks/SceneWorks') and type(job['runId']) is int and type(job['attempt']) is int, 'unreviewed compute repository or run')


def prepare_compile(out, env, source, runtime, deadline):
    """Wait only for measured capacity; retain each interrupted private build."""
    target = out / 'target'
    require(target.is_dir() and not target.is_symlink() and Path(env['CARGO_TARGET_DIR']) == target, 'private preparation target missing or changed')
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    history = []
    def state(status, **details):
        history.append(dict(utc=utc().isoformat(), status=status, **details))
        require(len(history) <= 200, 'preparation history bound exceeded')
        write(out / 'preparation-queue.json', {'deadlineMonotonic': deadline, 'events': history})
    def budget(reserve=0):
        left = deadline - time.monotonic()
        if left <= reserve:
            state('deadline-expired')
            raise ValueError('bounded preparation deadline expired')
        return left - reserve
    def revalidate():
        budget(70)
        require(target.is_dir() and not target.is_symlink() and (target.stat().st_dev, target.stat().st_ino) == target_identity, 'private preparation target missing or changed')
        require(inventory(deadline - 70) == source, 'source changed during preparation queue')
        require(runtime_identity(env, deadline - 70) == runtime, 'runtime changed during preparation queue')
        budget(70)
    def cooldown():
        state('waiting')
        time.sleep(min(POLL_SECONDS, budget(70)))
        budget(70)
    attempt = 0
    while True:
        consecutive = 0
        while consecutive < 2:
            revalidate()
            try: capacity(out, prelaunch=True, deadline=deadline - 70)
            except CapacityUnavailable as error:
                consecutive = 0
                state('capacity-unavailable', predicate=error.predicate, sample=error.sample)
            else:
                consecutive += 1
                state('capacity-pass', consecutive=consecutive)
            if consecutive < 2: cooldown()
        state('capacity-restored')
        revalidate()
        attempt += 1; label = f'compile-{attempt:03d}'
        high_since = None
        def monitor_build(_owned):
            nonlocal high_since
            try: high_since = capacity(out, high_since=high_since, deadline=deadline - 70)
            except CapacityUnavailable:
                state('owned-draining', attempt=attempt)
                raise
        def prelaunch():
            budget(70)
            return capacity(out, prelaunch=True, deadline=deadline - 70)
        state('retrying' if attempt > 1 else 'compiling', attempt=attempt)
        try:
            log = run_owned(['cargo', 'test', '--locked', '--lib', '-p', 'mlx-gen-moss-image', '--no-run', '--message-format=json'], env, out, label, budget(70), monitor_build, prelaunch, absolute_deadline=deadline - 60)
        except CapacityUnavailable:
            receipt = read(out / (label + '-process.json'))
            require(receipt.get('capacityUnavailable') and receipt['remaining'] == [], 'capacity interruption lacks complete owned cleanup')
            sent_root_signal = any(row['pid'] == receipt.get('pid') and row['birth'] == receipt.get('observed', {}).get(str(receipt.get('pid')), {}).get('birth') and row['signal'] == -receipt.get('exitCode', 0) for row in receipt.get('sentSignals', []))
            require((receipt.get('launched') is False and receipt['exitCode'] is None) or (receipt.get('capacityInterrupted') is True and sent_root_signal and receipt['exitCode'] in (-signal.SIGTERM, -signal.SIGKILL)), 'not a controller-initiated capacity interruption')
            state('owned-drained', attempt=attempt, receipt=label + '-process.json')
            cooldown()
            continue
        revalidate()
        state('compiled', attempt=attempt)
        return log


def prepare():
    deadline = time.monotonic() + 1200
    require(sys.platform == 'darwin' and os.environ['RUNNER_NAME'] in ('nax-macos', 'nax-macos-2'), 'known NAX runner required')
    out = paths(True)
    for name in ('target', 'tmp', 'hf', 'hub', 'hub-legacy', 'whisper'): (out / name).mkdir()
    source = inventory(deadline - 70); write(out / 'source.json', source)
    env = child_env(out)
    runtime = runtime_identity(env, deadline - 70); write(out / 'runtime.json', runtime)
    log = prepare_compile(out, env, source, runtime, deadline)
    binaries = []
    for line in log.splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get('reason') == 'compiler-artifact' and event.get('target', {}).get('name') == 'mlx_gen_moss_image' and event.get('profile', {}).get('test') and event.get('executable'):
            binaries.append(Path(event['executable']).resolve())
    require(len(binaries) == 1 and binaries[0].is_relative_to(out / 'target'), 'missing or escaping prepared binary')
    binary = binaries[0]
    remaining = deadline - time.monotonic() - 10
    require(remaining > 0, 'preparation deadline expired before listing')
    listed = run_owned([str(binary), TEST, '--exact', '--list', '--ignored'], env, out, 'list', min(60, remaining), absolute_deadline=deadline)
    require(re.findall(r'^(.+): test$', listed, re.M) == [TEST], 'missing or extra ignored oracle membership')
    require(runtime_identity(env, deadline) == runtime, 'build runtime changed')
    require(time.monotonic() < deadline, 'preparation deadline expired before metadata')
    current = snapshot(deadline); write(out / 'metadata.json', current)
    token = token_path(); token.parent.mkdir(parents=True, exist_ok=True)
    with token.open('a', encoding='utf-8'): pass
    require(inventory(deadline) == source, 'source changed during preparation')
    prepared = {'source': os.environ['GITHUB_SHA'], 'actor': os.environ['GITHUB_ACTOR'], 'runId': int(os.environ['GITHUB_RUN_ID']), 'attempt': 1, 'host': current['host'], 'runner': current['runner'], 'binary': str(binary), 'binarySha256': sha(binary), 'controllerSha256': sha(Path(__file__)), 'jobOwner': current['controllerJobOwner'], 'tokenIdentity': token_identity(token), 'sourceInventorySha256': sha(out / 'source.json'), 'runtimeSha256': sha(out / 'runtime.json'), 'metadataSha256': sha(out / 'metadata.json'), 'members': [TEST], 'preparedUtc': utc().isoformat()}
    require(time.monotonic() < deadline, 'preparation deadline expired before publication')
    write(out / 'prepared.json', prepared)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output: output.write('evidence=' + str(out) + '\n')
    print(json.dumps(prepared))

def api(path): return json.loads(metadata(['gh', 'api', path]))

def publish_lease():
    lease = json.loads(os.environ['MOSS_METAL_LEASE'], object_pairs_hook=unique_fields)
    validate_lease(lease)
    require(lease['schemaVersion'] == 1 and lease['source'] == os.environ['MOSS_METAL_SOURCE'] == os.environ['GITHUB_SHA'], 'lease source mismatch')
    require(lease['actor'] == os.environ['GITHUB_ACTOR'], 'lease publisher actor mismatch')
    require(stamp(lease['validFromUtc']) <= utc() < stamp(lease['expiresUtc']) and 1200 <= (stamp(lease['expiresUtc']) - stamp(lease['validFromUtc'])).total_seconds() <= 7200, 'invalid lease window')
    parent = api(f"repos/SceneWorks/inference/actions/runs/{int(lease['preparationRunId'])}")
    require(parent['head_sha'] == lease['source'] and parent['run_attempt'] == lease['preparationAttempt'] == 1 and parent['actor']['login'] == lease['actor'] and parent['event'] == 'workflow_dispatch', 'preparation run provenance mismatch')
    require(lease['exclusiveMetalExecutionAuthorized'] is True, 'root execution authority missing')
    require(re.fullmatch('[0-9a-f]{64}', lease['binarySha256']) and re.fullmatch('[0-9a-f]{64}', lease['metadataSha256']), 'binary or metadata digest missing')
    require(inventory(), 'empty source checkout')
    out = paths(True); write(out / 'lease.json', lease)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output: output.write('evidence=' + str(out) + '\n'); output.write('name=' + f"moss-metal-lease-{lease['preparationRunId']}-1" + '\n')

def fetch_lease(out, prepared, cache):
    runs = api('repos/SceneWorks/inference/actions/workflows/ci.yml/runs?event=workflow_dispatch&status=completed&per_page=50')['workflow_runs']
    name = f"moss-metal-lease-{prepared['runId']}-1"
    for run in runs:
        if run['head_sha'] != prepared['source'] or run['actor']['login'] != prepared['actor'] or run['conclusion'] != 'success' or run['run_attempt'] != 1: continue
        if run['id'] in cache['not_publishers']: continue
        artifacts = api(f"repos/SceneWorks/inference/actions/runs/{run['id']}/artifacts")['artifacts']
        if not any(row['name'] == name and not row['expired'] for row in artifacts):
            cache['not_publishers'].add(run['id']); continue
        dest = out / ('lease-publisher-' + str(run['id']))
        if not dest.exists():
            dest.mkdir(); metadata(['gh', 'run', 'download', str(run['id']), '--repo', 'SceneWorks/inference', '--name', name, '--dir', str(dest)])
        lease = read(dest / 'lease.json'); validate_lease(lease)
        if stamp(lease['expiresUtc']) <= utc() + dt.timedelta(seconds=1200): continue
        return lease, run['id']
    return None

def authenticated_waiter(repository, run, job, prepared, out, cache):
    # A matching name is insufficient: source/controller, successful preparation
    # artifact, exact physical token inode and Runner.Worker identity must agree.
    if repository != 'SceneWorks/inference' or run['event'] != 'workflow_dispatch' or run['run_attempt'] != 1 or run['head_sha'] != prepared['source'] or run['actor']['login'] != prepared['actor'] or job['name'] != 'moss-edit-metal': return None
    waiting_step = 'Wait for the reviewed lease, recheck physical ownership, and run one exact oracle'
    if not any(step['name'] == waiting_step and step['status'] == 'in_progress' for step in job.get('steps', [])): return None
    if run['id'] not in cache['waiters']:
        name = f"moss-metal-prepared-{run['id']}-1"
        artifacts = api(f"repos/SceneWorks/inference/actions/runs/{run['id']}/artifacts")['artifacts']
        if not any(row['name'] == name and not row['expired'] for row in artifacts): return None
        dest = out / ('waiter-' + str(run['id']))
        if not dest.exists():
            dest.mkdir(); metadata(['gh', 'run', 'download', str(run['id']), '--repo', repository, '--name', name, '--dir', str(dest)])
        cache['waiters'][run['id']] = read(dest / 'prepared.json')
    other = cache['waiters'][run['id']]
    for key in ('source', 'actor', 'host', 'controllerSha256', 'tokenIdentity'):
        if other.get(key) != prepared[key]: return None
    if other.get('runId') != run['id'] or other.get('attempt') != 1 or other.get('members') != [TEST]: return None
    return other['jobOwner']


def competing_jobs(lease, prepared, out, cache):
    active, waiters = [], []
    for repository in ('SceneWorks/inference', 'SceneWorks/SceneWorks'):
        response = api(f'repos/{repository}/actions/runs?status=in_progress&per_page=100')
        require(response['total_count'] <= 100, 'active-run census exceeds bound')
        for run in response['workflow_runs']:
            jobs = api(f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100")
            require(jobs['total_count'] <= 100, 'active-job census exceeds bound')
            for job in jobs['jobs']:
                if job['status'] != 'in_progress' or job['runner_name'] not in lease['runnerNamesOnPhysicalHost']: continue
                if repository == 'SceneWorks/inference' and run['id'] == prepared['runId'] and run['run_attempt'] == prepared['attempt']: continue
                waiter = authenticated_waiter(repository, run, job, prepared, out, cache)
                if waiter is not None: waiters.append(waiter)
                else: active.append({'repository': repository, 'runId': run['id'], 'attempt': run['run_attempt'], 'jobId': job['id'], 'runner': job['runner_name']})
    return active, waiters


def ownership(lease, old, current, owned=None, authenticated_jobs=()):
    owned = owned or {}
    require(current['host'] == old['host'] == lease['host'] and current['runner'] == old['runner'] == lease['runner'] and current['device'] == old['device'] == lease['device'], 'physical Metal host/device changed')
    require(lease['classificationUtc'] == old['utc'], 'ownership classification timestamp mismatch')
    require(current['controllerJobOwner'] == old['controllerJobOwner'], 'own Actions job incarnation changed')
    before_processes = {row['pid']: row for row in old['processes']}
    rules = {row['pid']: row for row in lease['processes']}
    require(set(before_processes) == set(rules), 'lease did not classify every prepared local pipeline')
    # These are exact authenticated controller job trees, not a general runner or
    # interpreter exemption. All use the same physical token while executing.
    trusted = descendants(current['processes'], [old['controllerJobOwner'], *authenticated_jobs])
    for row in current['processes']:
        if row['pid'] in trusted or owned.get(str(row['pid']), {}).get('birth') == row['birth']: continue
        require(row['pid'] in before_processes and row == before_processes[row['pid']], 'new or changed local compute pipeline')
        require(rules[row['pid']]['disposition'] == 'reviewed-no-compute', 'current local compute pipeline must drain')

    previous = {row['entry_id']: row for row in old['clients']}
    permitted = {row['entry_id']: row for row in lease['clients']}
    require(set(previous) == set(permitted), 'lease did not classify all prepared GPU clients')
    for row in current['clients']:
        if owned.get(str(row['pid']), {}).get('birth') == row['birth']: continue
        require(row['entry_id'] in previous, 'unreviewed GPU client; remain queued')
        rule, before = permitted[row['entry_id']], previous[row['entry_id']]
        require(not rule['mustBeAbsent'], 'classified competing GPU client remains; remain queued')
        for key in ('pid', 'birth', 'comm', 'owner'): require(row[key] == before[key], 'GPU client incarnation changed')
        require(type(rule['allowBackgroundActivity']) is bool, 'unclassified background activity')
        if not rule['allowBackgroundActivity']: require(row['usage_present'] == before['usage_present'] and row['usage'] == before['usage'], 'unapproved GPU activity')
    permitted_listeners = {row['pid']: row for row in lease['listeners']}
    before_listeners = {row['pid']: row for row in old['listeners']}
    require(set(permitted_listeners) == {row['pid'] for row in old['listeners']}, 'lease did not classify all prepared listeners')
    for row in current['listeners']:
        require(row['pid'] in permitted_listeners, 'unreviewed listener; remain queued')
        rule = permitted_listeners[row['pid']]
        require(not rule['mustBeAbsent'], 'competing listener remains; remain queued')
        for key in ('birth', 'comm', 'endpoints'): require(row[key] == before_listeners[row['pid']][key], 'listener incarnation changed')
        require(rule['disposition'] == 'explicitly-reviewed-no-compute', 'idle listener is not admission')

def final_admission(lease, prepared, old, source, binary, out, cache):
    # Slow source/capacity work happens first. The complete current job and local
    # ownership census follows it, directly at the launch boundary.
    try:
        capacity(out, prelaunch=True)
        require(inventory() == source and sha(binary) == prepared['binarySha256'], 'source/binary changed before execution')
        require(runtime_identity(child_env(out)) == read(out / 'runtime.json'), 'runtime changed before execution')
        for job in lease['computeJobsToDrain']:
            current_job = api(f"repos/{job['repository']}/actions/runs/{int(job['runId'])}")
            require(current_job['run_attempt'] == job['attempt'] and current_job['status'] == 'completed', 'reviewed compute run still active')
        jobs, waiters = competing_jobs(lease, prepared, out, cache)
        require(not jobs, 'another Actions compute job owns this physical host')
        current = snapshot(); ownership(lease, old, current, authenticated_jobs=waiters)
        require(stamp(lease['validFromUtc']) <= utc() and utc() + dt.timedelta(seconds=1200) <= stamp(lease['expiresUtc']), 'execution lease is not current')
        return {'utc': utc().isoformat(), 'freshSnapshot': current, 'authenticatedWaiters': waiters, 'competingJobs': jobs, 'historicalClassification': old['utc']}
    except (ValueError, OSError, KeyError) as error:
        raise AdmissionDeferred(str(error)) from error


def execute():
    out = paths(); prepared = read(out / 'prepared.json'); old = read(out / 'metadata.json'); source = read(out / 'source.json'); binary = Path(prepared['binary'])
    require(inventory() == source and sha(binary) == prepared['binarySha256'] and sha(out / 'metadata.json') == prepared['metadataSha256'] and sha(out / 'source.json') == prepared['sourceInventorySha256'], 'prepared source/binary changed')
    require(runtime_identity(child_env(out)) == read(out / 'runtime.json') and sha(out / 'runtime.json') == prepared['runtimeSha256'], 'prepared runtime changed')
    token = token_path()
    require(token_identity(token) == prepared['tokenIdentity'], 'physical token changed')
    end = time.monotonic() + 5400
    cache = {'not_publishers': set(), 'waiters': {}}
    with token.open('a', encoding='utf-8') as lock:
        while True:
            require(time.monotonic() < end, 'bounded Metal queue expired without admission')
            if not try_execution_token(lock): time.sleep(POLL_SECONDS); continue
            try:
                require(token_identity(token) == prepared['tokenIdentity'], 'physical token changed')
                candidate = fetch_lease(out, prepared, cache)
                require(candidate is not None, 'awaiting authenticated root lease artifact')
                lease, publisher = candidate
                for key, value in {'source': prepared['source'], 'actor': prepared['actor'], 'preparationRunId': prepared['runId'], 'preparationAttempt': 1, 'binarySha256': prepared['binarySha256'], 'metadataSha256': prepared['metadataSha256'], 'sourceInventorySha256': prepared['sourceInventorySha256']}.items(): require(lease[key] == value, 'lease preparation binding mismatch: ' + key)
                require(lease['schemaVersion'] == 1 and lease['exclusiveMetalExecutionAuthorized'] is True, 'root execution authority missing')
            except (ValueError, OSError, KeyError) as error:
                write(out / 'queue-state.json', {'utc': utc().isoformat(), 'status': 'waiting-not-admitted', 'reason': str(error)})
                release_execution_token(lock); time.sleep(POLL_SECONDS); continue
            env = child_env(out); env['MOSS_NATIVE_TEST_ADMITTED'] = '1'
            high_since, next_jobs_check, waiters = None, 0.0, []
            def prelaunch():
                nonlocal waiters, next_jobs_check
                admission = final_admission(lease, prepared, old, source, binary, out, cache)
                waiters = admission['authenticatedWaiters']; next_jobs_check = time.monotonic() + POLL_SECONDS
                admission['publisherRunId'] = publisher
                return admission
            def monitor_oracle(owned):
                nonlocal high_since, next_jobs_check, waiters
                require(utc() < stamp(lease['expiresUtc']), 'execution lease expired')
                high_since = capacity(out, high_since=high_since)
                if time.monotonic() >= next_jobs_check:
                    jobs, waiters = competing_jobs(lease, prepared, out, cache)
                    write(out / 'competing-jobs.json', jobs)
                    require(not jobs, 'another Actions compute job claimed this physical host')
                    next_jobs_check = time.monotonic() + POLL_SECONDS
                ownership(lease, old, snapshot(), owned, waiters)
            try:
                log = run_owned([str(binary), TEST, '--exact', '--ignored', '--nocapture', '--test-threads=1'], env, out, 'oracle', 1200, monitor=monitor_oracle, prelaunch=prelaunch)
            except AdmissionDeferred as error:
                write(out / 'queue-state.json', {'utc': utc().isoformat(), 'status': 'waiting-not-admitted', 'reason': str(error)})
                release_execution_token(lock); time.sleep(POLL_SECONDS); continue
            require(re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', log) == [('ok', '1', '0', '0')], 'oracle did not execute exactly one test')
            require(inventory() == source and sha(binary) == prepared['binarySha256'], 'source/binary changed during oracle')
            ownership(lease, old, snapshot(), authenticated_jobs=waiters)
            write(out / 'result.json', {'passed': True, 'source': prepared['source'], 'binarySha256': prepared['binarySha256'], 'expectedTests': [TEST], 'actualPassed': 1, 'scope': 'synthetic full Metal graph; not public RGB or real weights'})
            return


if __name__ == '__main__':
    def interrupted(signum, _frame): raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    {'prepare': prepare, 'execute': execute, 'publish-lease': publish_lease}[sys.argv[1]]()
