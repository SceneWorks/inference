"""Explicit two-UUID source role binding; leases are NOT bounded monitoring or NN provenance."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from types import SimpleNamespace

from scripts.ci import model_proof_cuda_primitives as primitives

UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
CASE = {'package': 'candle-llm', 'target': 'shard', 'target_kind': 'test',
        'fqn': 'real::shards_across_two_gpus_and_matches_single', 'source': 'crates/llm/candle-llm/tests/shard.rs'}
FIELDS = frozenset(('cuda_device_uuids', 'additional_cuda_policies'))
GPU_POLICY_FIELDS = frozenset(('cudaDeviceUuid', 'gpuFreeReserveBytes', 'childGpuCapBytes'))


def validate_request(value: dict, backend: str) -> tuple[str, ...]:
    if not FIELDS.intersection(value):
        return ()
    if not FIELDS <= set(value) or backend != 'candle-cuda':
        raise ValueError('explicit multi-GPU resources require CUDA ordered UUIDs and secondary policies together')
    uuids = value['cuda_device_uuids']
    extra = value['additional_cuda_policies']
    if (not isinstance(uuids, list) or len(uuids) != 2
            or any(not isinstance(uuid, str) or not UUID.fullmatch(uuid) for uuid in uuids)
            or len({uuid.casefold() for uuid in uuids}) != 2
            or not isinstance(extra, list) or len(extra) != 1 or not isinstance(extra[0], dict)):
        raise ValueError('source CUDA0/1 requires exactly two distinct explicit approved UUID policies')
    primary = value['policy']
    secondary = extra[0]
    if (primary.get('cudaDeviceUuid') != uuids[0] or secondary.get('cudaDeviceUuid') != uuids[1]
            or primary.get('backend') != 'windows-cuda' or secondary.get('backend') != 'windows-cuda'
            or {key: item for key, item in primary.items() if key not in GPU_POLICY_FIELDS}
               != {key: item for key, item in secondary.items() if key not in GPU_POLICY_FIELDS}
            or not GPU_POLICY_FIELDS <= set(secondary)):
        raise ValueError('ordered UUID policies must bind role0/1 and identical explicit host/tree/log/deadline settings')
    return tuple(uuids)


def validate_source(row: dict, *, required: bool = False) -> bool:
    source_case = all(row.get(key) == value for key, value in CASE.items())
    ordinals = row.get('required_cuda_ordinals')
    if ordinals is None and not required:
        return source_case
    if (not source_case or row.get('backend') != 'candle-cuda' or ordinals != [0, 1]
            or any(type(ordinal) is not int for ordinal in ordinals)
            or not {'cuda', 'candle-llm/cuda'}.intersection(row.get('cargo_features', []))):
        raise ValueError('multi-GPU source binding requires exact declared shard case CUDA roles0/1, never inferred physical indices')
    return True


def bind_rows(rows: tuple[dict, ...], value: dict, backend: str) -> tuple[str, ...]:
    uuids = validate_request(value, backend)
    needs = [validate_source(row, required=all(row.get(key) == item for key, item in CASE.items())) for row in rows]
    if any(needs) and not uuids:
        raise ValueError('source two-GPU work requires two explicit approved UUIDs before any NN launch')
    if uuids and (not rows or not all(needs)):
        raise ValueError('two-GPU request is limited to source-declared CUDA0/1 work, not arbitrary device expansion')
    return uuids


def validate_driver_binding(binding: dict, uuids: tuple[str, ...], environment: dict[str, str]) -> None:
    packet = binding.get('probe', {})
    if (type(packet.get('schema_version')) is not int or packet['schema_version'] != 1
            or packet.get('context_created') is not False
            or packet.get('abi_source') != {'crate': 'cudarc', 'version': '0.19.7',
                'checksum': '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'}
            or packet.get('visibility') != {'CUDA_VISIBLE_DEVICES': ','.join(uuids),
                                             'CUDA_DEVICE_ORDER': environment.get('CUDA_DEVICE_ORDER')}):
        raise ValueError('multi-GPU ordinal mapping did not use exact actual context-free visibility/order')
    actual = tuple(str(item.get('uuid', '')).casefold() for item in binding.get('mapping', []))
    if actual != tuple(uuid.casefold() for uuid in uuids):
        raise ValueError('actual ordered CUDA0/1 UUID mapping differs from explicit approved roles')


def validate_case_evidence(row: dict, value: dict, case: dict) -> None:
    uuids = bind_rows((row,), value, 'candle-cuda')
    if not uuids:
        return
    physical = case.get('physical_execution', {})
    binding = physical.get('actual_ordinal_binding', {})
    order = case.get('configuration_environment', {}).get('CUDA_DEVICE_ORDER', {})
    validate_driver_binding(binding, uuids, {'CUDA_DEVICE_ORDER': order.get('value') if order.get('present') else None})
    mapping = binding.get('mapping', [])
    post = physical.get('post_idle_devices', [])
    if (physical.get('approved_uuid_roles') != list(uuids) or physical.get('post_execution_verified') is not True
            or len(mapping) != len(uuids)
            or any(item.get('cuda_ordinal') != role or str(item.get('uuid', '')).casefold() != uuid.casefold()
                   for role, (item, uuid) in enumerate(zip(mapping, uuids)))
            or len(post) != len(uuids) or {item.get('uuid') for item in post} != set(uuids)
            or any(item.get('verified') is not True for item in post)):
        raise ValueError('multi-GPU case lacks every approved ordered role/lease/post-idle completion')
    monitors = case.get('cuda_devices', [])
    policies = [value['policy'], *value['additional_cuda_policies']]
    if not isinstance(monitors, list) or len(monitors) != len(uuids):
        raise ValueError('multi-GPU case lacks actual bounded monitor evidence for every approved UUID')
    for role, (monitor, uuid, policy) in enumerate(zip(monitors, uuids, policies)):
        digest = hashlib.sha256((json.dumps(policy, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
        peak = monitor.get('peakGpuBytes')
        if (monitor.get('role') != role or monitor.get('cudaDeviceUuid') != uuid or monitor.get('policySha256') != digest
                or not isinstance(monitor.get('admission'), dict) or not isinstance(monitor.get('samples'), (list, tuple))
                or peak is not None and (type(peak) is not int or peak <= 0)
                or peak is None and not monitor.get('peakGpuUnknownReason')):
            raise ValueError('multi-GPU monitor role/policy/unknown-peak evidence differs from explicit approved budget')
    commands = case.get('commands', [])
    if (not commands or json.dumps(commands[-1].get('supervision', {}).get('cuda_devices'), sort_keys=True)
            != json.dumps(monitors, sort_keys=True)):
        raise ValueError('multi-GPU monitor evidence differs from actual owned-tree supervision result')


def _secure_namespace(public: Path, uuids: tuple[str, ...]) -> set[tuple[int, int]]:
    if not public.is_absolute():
        raise ValueError('PUBLIC must be the existing absolute host-global CUDA lock namespace')
    directory = public / 'SceneWorks/gpu-locks'
    for candidate in (public, public / 'SceneWorks', directory):
        if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
            raise ValueError('physical CUDA namespace cannot be a link/reparse fallback')
    directory.mkdir(parents=True, exist_ok=True)
    identities = set()
    for uuid in uuids:
        path = directory / (uuid + '.lock')
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('physical CUDA lock cannot be linked')
        with path.open('a+b') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('physical CUDA lock must be a unique regular file')
            identities.add((info.st_dev, info.st_ino))
            if handle.seek(0, 2) == 0:
                handle.write(b'0'); handle.flush()
    return identities


@contextmanager
def execution_claim(row: dict, policies: tuple, value: dict, output: Path, env: dict[str, str]):
    """Claim only caller-approved UUIDs; role order is independent of deterministic lock order."""
    uuids = bind_rows((row,), value, 'candle-cuda')
    if os.name != 'nt' or len(policies) != len(uuids) or tuple(policy.cuda_device_uuid for policy in policies) != uuids:
        raise ValueError('multi-GPU claim requires exact registered Windows policy/role binding')
    if env.get('CUDA_VISIBLE_DEVICES') != ','.join(uuids):
        raise ValueError('model visibility must preserve explicit approved source role order')
    import msvcrt
    public = Path(env['PUBLIC'])
    identities = _secure_namespace(public, uuids)
    def locking(descriptor, operation, count):
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) not in identities or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('physical CUDA lock inode changed; no namespace fallback')
        msvcrt.locking(descriptor, operation, count)
    locker = SimpleNamespace(LK_NBLCK=msvcrt.LK_NBLCK, LK_UNLCK=msvcrt.LK_UNLCK, locking=locking)
    deadline = time.monotonic() + policies[0].deadline_seconds
    guard = primitives.component('veda_cuda_guard.py', primitives.GUARD_SHA)
    original_command = guard.command
    def bounded_command(argv, requested_deadline=None):
        return original_command(argv, min(deadline, requested_deadline) if requested_deadline is not None else deadline)
    def bounded_sleep(seconds):
        if time.monotonic() + seconds > deadline:
            raise TimeoutError('approved multi-GPU admission/post-idle budget exhausted')
        time.sleep(seconds)
    guard.command = bounded_command
    guard.time = SimpleNamespace(monotonic=time.monotonic, sleep=bounded_sleep)
    devices = []; handles = []; baselines = {}; record = None; body_error = None
    try:
        devices, handles, baselines, waited = primitives.claim_uuids(
            uuids, guard, output, deadline, public, locker, sleep=bounded_sleep)
        # Fresh context-free driver mapping in EXACT actual child visibility/order, after both leases.
        metadata_helper = Path(__file__).with_name('windows_cuda_metadata.py')
        if hashlib.sha256(metadata_helper.read_bytes()).hexdigest() != '0eaa7a4f32a0c0d686544b9736505584875a6d2f8c9319a5e9c37b4881a235cd':
            raise ValueError('multi-GPU mapping primitive differs from verified context-free source')
        binding = primitives.resolve_binding(guard, output, deadline, env, cuda_ordinals='0,1')
        validate_driver_binding(binding, uuids, env)
        record = {'mechanism': 'existing PUBLIC UUID leases; sorted lock order, preserved CUDA source role order',
                  'approved_uuid_roles': list(uuids), 'lock_order': [device['uuid'] for device in devices],
                  'actual_ordinal_binding': binding, 'admission_baselines': baselines,
                  'wait_seconds': waited, 'post_execution_verified': False, 'post_idle_devices': []}
        try:
            yield record
        except BaseException as error:
            body_error = error
            raise
    except RuntimeError as error:
        body_error = error
        raise ValueError('explicit multi-GPU admission/binding failed: ' + str(error)) from error
    finally:
        failures = []
        # Even when one post-census fails, attempt EVERY approved UUID. No complete/accepted seal otherwise.
        for device in devices:
            try:
                guard.verify_idle(device, output, 'model-proof-release-' + device['uuid'], baselines[device['uuid']])
                if record is not None:
                    record['post_idle_devices'].append({'uuid': device['uuid'], 'verified': True})
            except BaseException as error:
                failures.append(str(error))
                if record is not None:
                    record['post_idle_devices'].append({'uuid': device['uuid'], 'verified': False, 'error': str(error)})
        try:
            primitives.release_all(handles, locker)
        except BaseException as error:
            failures.append('physical lease release failed: ' + str(error))
            if body_error is None:
                raise
        finally:
            if record is not None:
                record['post_execution_verified'] = len(devices) == len(uuids) and not failures
        if failures and body_error is None:
            raise ValueError('not every approved CUDA UUID passed post-idle verification: ' + '; '.join(failures))
