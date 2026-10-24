"""Reuse existing execution coordination and supervisor policies; never submission/job-count gates."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


CUDA_METADATA_HELPER_SHA256 = '0eaa7a4f32a0c0d686544b9736505584875a6d2f8c9319a5e9c37b4881a235cd'
CUDA_METADATA_ABI_CHECKSUM = '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'


def validate_child_visibility(row: dict) -> None:
    fixed = {'package': 'candle-gen-iris', 'target': 'integration', 'target_kind': 'test',
             'fqn': 'train_distributed::a_rank_without_its_cuda_device_is_refused_unless_sharing_is_opted_in',
             'source': 'crates/media/candle-gen/candle-gen-iris/tests/train_distributed.rs'}
    source_case = all(row.get(key) == value for key, value in fixed.items())
    declaration = row.get('child_cuda_visibility')
    if declaration is None and not source_case and 'child_cuda_visibility' not in row:
        return
    if (not source_case or row.get('backend') != 'candle-cuda'
            or not {'cuda', 'candle-gen-iris/cuda'}.intersection(row.get('cargo_features', []))):
        raise ValueError('child CUDA visibility is limited to its exact source-declared Iris case')
    if (not isinstance(declaration, dict) or set(declaration) != {'source_identity', 'visible_devices', 'device_order', 'ordinal'}
            or declaration['visible_devices'] != '0' or declaration['device_order'] != 'inherit'
            or type(declaration['ordinal']) is not int or declaration['ordinal'] != 0
            or not isinstance(declaration['source_identity'], str) or not declaration['source_identity'].startswith(row['source'] + '::')):
        raise ValueError('child CUDA binding must retain exact source visibility/order/ordinal facts')


def child_visibility_binding(row: dict, env: dict[str, str], approved_uuid: str, remaining: float) -> dict:
    """Resolve the original child reset in a FRESH context-free metadata process, inside its UUID lease."""
    validate_child_visibility(row)
    declaration = row['child_cuda_visibility']
    if remaining <= 0:
        raise ValueError('approved deadline expired before child CUDA visibility mapping')
    helper = Path(__file__).with_name('windows_cuda_metadata.py')
    if hashlib.sha256(helper.read_bytes()).hexdigest() != CUDA_METADATA_HELPER_SHA256:
        raise ValueError('context-free CUDA metadata primitive differs from verified App source')
    child_env = dict(env)
    child_env['CUDA_VISIBLE_DEVICES'] = declaration['visible_devices']
    # CUDA_DEVICE_ORDER is inherited exactly, including UNSET; never infer NVML index from an ordinal.
    try:
        result = subprocess.run([sys.executable, '-I', str(helper)], cwd=helper.parents[2],
                                env=child_env, capture_output=True, text=True, timeout=min(30, remaining), check=True, encoding="utf-8")
    except subprocess.SubprocessError as error:
        raise ValueError('context-free child CUDA metadata process failed before model execution') from error
    if len(result.stdout.encode('utf-8')) > 16384:
        raise ValueError('child CUDA metadata exceeds its fixed metadata-only budget')
    packet = json.loads(result.stdout)
    expected_visibility = {'CUDA_VISIBLE_DEVICES': child_env.get('CUDA_VISIBLE_DEVICES'),
                           'CUDA_DEVICE_ORDER': child_env.get('CUDA_DEVICE_ORDER')}
    if (not isinstance(packet, dict) or type(packet.get('schema_version')) is not int
            or packet['schema_version'] != 1 or packet.get('context_created') is not False
            or packet.get('abi_source') != {'crate': 'cudarc', 'version': '0.19.7', 'checksum': CUDA_METADATA_ABI_CHECKSUM}
            or packet.get('visibility') != expected_visibility):
        raise ValueError('child effective visibility/context-free CUDA ABI is unverified')
    devices = packet.get('devices')
    if (not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict)
            or type(devices[0].get('ordinal')) is not int or devices[0]['ordinal'] != declaration['ordinal']
            or not isinstance(devices[0].get('uuid'), str)
            or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', devices[0]['uuid'])):
        raise ValueError('source child CUDA ordinal mapping is absent or ambiguous')
    if devices[0]['uuid'].casefold() != approved_uuid.casefold():
        raise ValueError('original child visibility selects a UUID other than the approved leased device')
    return {'source_declaration': declaration, 'metadata_primitive_sha256': CUDA_METADATA_HELPER_SHA256,
            'metadata': packet, 'approved_leased_uuid': approved_uuid,
            'binding': 'exact child-effective CUDA ordinal UUID equals approved leased NVML UUID; no index inference'}


def metadata_policy_payload(policy: dict) -> dict:
    """Compilation/discovery does not execute a model or require a fabricated CUDA sample."""
    result = dict(policy)
    if result['backend'] == 'windows-cuda':
        result['backend'] = 'windows-cpu'
    elif result['backend'] == 'linux-cuda':
        result['backend'] = 'linux-cpu'
    if result['backend'] in ('windows-cpu', 'linux-cpu'):
        for key in ('cudaDeviceUuid', 'gpuFreeReserveBytes', 'childGpuCapBytes'):
            result.pop(key, None)
    return result


@contextmanager
def execution_claim(backend: str, policy, output: Path, env: dict[str, str], *, constraints: dict | None = None):
    """Claim only during execution. CUDA reuses Veda's host-global UUID lock namespace.

    A hosted Mac/Linux worker is an isolated Actions execution environment. The provisioned
    Mac route claims the shared host-global cooperative POSIX lease in addition to its runner
    route. Labels/queues are not a physical mutex or a global inventory assertion.
    """
    if backend != 'candle-cuda':
        hosted = env.get('MODEL_PROOF_RUNNER_ENVIRONMENT') == 'github-hosted'
        if not hosted and env.get('RUNNER_NAME') != 'nax-macos-2':
            raise ValueError('wrong physical execution route')
        if hosted:
            record = {'mechanism': 'isolated hosted runner', 'runner': env.get('RUNNER_NAME'),
                      'global_org_inventory_claimed': False, 'post_execution_verified': False}
            try:
                yield record
            finally:
                # The caller's existing supervisor returns/raises only after owned-tree cleanup.
                record['post_execution_verified'] = True
        else:
            from scripts.ci.metal_execution_lock import execution_lease
            with execution_lease(policy.deadline_seconds) as lease:
                from scripts.ci.yue2_precision_proof import metal_census, physical_busy_message
                try:
                    raw, busy = metal_census()
                except RuntimeError as error:
                    raise ValueError('Metal census failed: ' + str(error)) from error
                if busy:
                    raise ValueError(physical_busy_message(raw, busy, 'unwrapped Metal worker detected'))
                record = dict(lease.record(), runner=env.get('RUNNER_NAME'), global_org_inventory_claimed=False,
                              census='existing yue2_precision_proof.metal_census; no universal process-coverage claim',
                              post_execution_verified=False)
                previous_fd = env.get('SCENEWORKS_METAL_EXECUTION_LOCK_FD')
                env.update(lease.environment(env))
                try:
                    yield record
                finally:
                    if previous_fd is None:
                        env.pop('SCENEWORKS_METAL_EXECUTION_LOCK_FD', None)
                    else:
                        env['SCENEWORKS_METAL_EXECUTION_LOCK_FD'] = previous_fd
                    try:
                        raw, busy = metal_census()
                        if busy:
                            raise ValueError(physical_busy_message(raw, busy, 'post-execution Metal census refused'))
                    except RuntimeError as error:
                        raise ValueError('post-execution Metal census failed: ' + str(error)) from error
                    record['post_execution_verified'] = True
        return
    if os.name != 'nt':
        raise ValueError('CUDA physical lock requires the registered Windows CUDA route')
    import msvcrt
    # EXACT existing namespace/locking protocol from scripts/release/veda_cuda_guard.py.
    # Public is host-global across repository workspaces/listeners; runner.temp is not.
    uuid = policy.cuda_device_uuid
    constraints = constraints or {}
    validate_child_visibility(constraints)
    required_uuid = constraints.get('required_cuda_device_uuid')
    if required_uuid is not None and uuid != required_uuid:
        raise ValueError('policy UUID violates this case fixed physical-device restriction')
    directory = Path(env['PUBLIC']) / 'SceneWorks' / 'gpu-locks'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid + '.lock')
    if path.is_symlink():
        raise ValueError('physical lock may not be a symlink')
    handle = path.open('a+b')
    claimed = False
    try:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('physical lock must be a regular unique file')
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        started = time.monotonic()
        while not claimed:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                claimed = True
            except OSError:
                if time.monotonic() - started >= policy.deadline_seconds:
                    raise ValueError('approved wait deadline expired before physical-device claim')
                time.sleep(policy.poll_millis / 1000)
        from scripts.release import veda_cuda_guard
        devices, processes = veda_cuda_guard.inventory(output, 'model-proof-claimed-device')
        matched = [device for device in devices if device['uuid'] == uuid]
        if len(matched) != 1 or not veda_cuda_guard.idle(matched[0], processes):
            raise ValueError('claimed CUDA UUID is absent or occupied; no model execution')
        required_index = constraints.get('required_cuda_physical_index')
        if required_index is not None and matched[0]['index'] != required_index:
            raise ValueError('claimed UUID violates this case fixed physical-index restriction')
        record = {'mechanism': 'existing PUBLIC/SceneWorks/gpu-locks UUID lock', 'device_uuid': uuid,
                  'physical_index': matched[0]['index'], 'runner': env.get('RUNNER_NAME')}
        if constraints.get('child_cuda_visibility') is not None:
            record['child_visibility_binding'] = child_visibility_binding(
                constraints, env, uuid, policy.deadline_seconds - (time.monotonic() - started))
        # Reuse the existing stable-driver baseline, not the broader choose-any-device API.
        try:
            selected, baseline = veda_cuda_guard.sample_baseline(matched[0], output, 'model-proof-admission')
        except RuntimeError as error:
            raise ValueError('CUDA physical admission failed: ' + str(error)) from error
        record['post_execution_verified'] = False
        record['admission_baseline'] = baseline
        (output / 'physical-execution-claim.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        try:
            yield record
        finally:
            try:
                veda_cuda_guard.verify_idle(selected, output, 'model-proof-release', baseline)
            except RuntimeError as error:
                raise ValueError('CUDA post-idle verification failed: ' + str(error)) from error
            record['post_execution_verified'] = True
    finally:
        if claimed:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()
