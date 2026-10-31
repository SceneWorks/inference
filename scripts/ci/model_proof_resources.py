"""Reuse existing execution coordination and supervisor policies; never submission/job-count gates."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from scripts.release.veda_cuda_guard import MAX_BASELINE_UTILIZATION_PERCENT


CUDA_METADATA_HELPER_SHA256 = '0eaa7a4f32a0c0d686544b9736505584875a6d2f8c9319a5e9c37b4881a235cd'
CUDA_METADATA_ABI_CHECKSUM = '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'
STRICT_CUDA_ACTIVITY = 'strict-zero-utilization'
IRIS_CUDA_CORRECTNESS_ACTIVITY = 'non-isolated-correctness-low-activity'
SHARED_CUDA_HEADROOM_ACTIVITY = 'non-isolated-correctness-headroom'


class CudaClaimRejected(ValueError):
    def __init__(self, diagnostic: dict):
        super().__init__('claimed CUDA UUID is absent or occupied; no model execution')
        self.diagnostic = diagnostic


class CudaReleaseRejected(ValueError):
    def __init__(self, message: str, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


def proof_cuda_contexts(uuid: str, processes: list[list[str]], *, activity_profile: str = STRICT_CUDA_ACTIVITY) -> list[tuple[str, str, str]]:
    types = (('C', 'C+G', 'G') if activity_profile in
             (IRIS_CUDA_CORRECTNESS_ACTIVITY, SHARED_CUDA_HEADROOM_ACTIVITY) else ('C', 'C+G'))
    return sorted((row[0], row[1], row[3]) for row in processes
                  if len(row) == 4 and row[0] == uuid and row[3] in types)


def proof_cuda_idle(device: dict, processes: list[list[str]], policy,
                    *, activity_profile: str = STRICT_CUDA_ACTIVITY) -> bool:
    """Admit exact-card correctness work only under its source-bound activity policy."""
    mib = 1024**2
    total, used, free, utilization = (device.get(key) for key in
                                      ('total_mib', 'used_mib', 'free_mib', 'utilization_percent'))
    reserve, cap = policy.gpu_free_reserve_bytes, policy.child_gpu_cap_bytes
    if activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY:
        # Foreign utilization/contexts are observations, not ownership or safety prerequisites.
        return (all(type(value) is int for value in (total, used, free, reserve, cap))
                and total > 0 and 0 <= used <= total and 0 <= free <= total
                and used + free <= total + 2 and reserve > 0 and cap > 0
                and free * mib >= reserve + cap)
    limit = (MAX_BASELINE_UTILIZATION_PERCENT if activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY
             else 1 if activity_profile == STRICT_CUDA_ACTIVITY else None)
    if (limit is None or any(type(value) is not int for value in (total, used, free, utilization, reserve, cap))
            or total <= 0 or used < 0 or free < 0 or used > total or free > total
            or used + free > total + 2 or utilization < 0 or utilization >= limit or reserve <= 0 or cap <= 0
            or free * mib < reserve + cap or any(len(row) != 4 for row in processes)):
        return False
    selected = [row for row in processes if row and row[0] == device['uuid']]
    allowed_types = ('G', 'C+G') if utilization > 0 else ('G', 'C+G', 'C')
    if utilization > 0 and not selected:
        return False
    return all(len(row) == 4 and row[1].isdecimal() and 0 < int(row[1]) <= 2**32 - 1
               and row[3] in allowed_types for row in selected)


def proof_cuda_shared_release(device: dict, uuid: str, physical_index: int, total_mib: int) -> bool:
    """Post-owned-exit primary metadata check; admission headroom is diagnostic only."""
    total, used, free = (device.get(key) for key in ('total_mib', 'used_mib', 'free_mib'))
    return (device.get('uuid') == uuid and type(device.get('index')) is int
            and device['index'] == physical_index and type(total_mib) is int
            and all(type(value) is int for value in (total, used, free))
            and total == total_mib and total > 0 and 0 <= used <= total
            and 0 <= free <= total and used + free <= total + 2)


def shared_owned_completion(record: dict) -> bool:
    """Validate observations retained from run_guarded's existing owned Job completion."""
    observed = record.get('owned_execution_completion')
    if (isinstance(observed, dict) and set(observed) == {'scope', 'creation_attempted'}
            and observed.get('scope') == 'supervisor-no-child-created'
            and observed.get('creation_attempted') is False):
        return True
    return (isinstance(observed, dict)
            and set(observed) == {'scope', 'root_pid', 'root_returncode', 'waited_returncode',
                                  'job_member_pids', 'owned_pids'}
            and observed['scope'] == 'supervisor-owned-windows-job'
            and type(observed['root_pid']) is int and 0 < observed['root_pid'] <= 2**32 - 1
            and type(observed['root_returncode']) is int
            and type(observed['waited_returncode']) is int
            and observed['waited_returncode'] == observed['root_returncode']
            and observed['job_member_pids'] == [] and observed['owned_pids'] == [])


def validate_shared_release_record(record: dict, commands: list[dict], constraints: dict | None = None) -> None:
    """Join a shared release seal to the actual supervised body's retained observations."""
    recipe = (constraints or {}).get('selected_configuration')
    expected_shared = isinstance(recipe, dict) and recipe.get('cuda_activity_profile') == SHARED_CUDA_HEADROOM_ACTIVITY
    if not expected_shared and record.get('activity_profile') != SHARED_CUDA_HEADROOM_ACTIVITY:
        return
    if constraints is not None and (not expected_shared
            or record.get('activity_profile') != SHARED_CUDA_HEADROOM_ACTIVITY
            or record.get('device_uuid') != constraints.get('required_cuda_device_uuid')
            or type(record.get('physical_index')) is not int
            or record['physical_index'] != constraints.get('required_cuda_physical_index')):
        raise ValueError('shared CUDA release differs from its exact source-bound device/profile')
    body = [item.get('supervision', {}) for item in commands
            if '--list' not in item.get('argv', [])]
    if (len(body) != 1 or not shared_owned_completion(record)
            or record.get('lease_released') is not True or record.get('post_execution_verified') is not True
            or record['owned_execution_completion'].get('scope') != 'supervisor-owned-windows-job'
            or body[0].get('owned_completion') != record['owned_execution_completion']
            or body[0].get('pid') != record['owned_execution_completion']['root_pid']
            or body[0].get('returncode') != 0
            or type(record.get('claim_deadline_monotonic')) not in (int, float)
            or not math.isfinite(record['claim_deadline_monotonic'])
            or body[0].get('admission', {}).get('originalClaimDeadlineMonotonic') != record.get('claim_deadline_monotonic')):
        raise ValueError('shared CUDA release lacks its actual owned Job completion/unlock seal')


def bound_cuda_activity_profile(constraints: dict) -> str:
    """Only exact source-owned registry recipes may select non-default admission."""
    recipe = constraints.get('selected_configuration')
    profile = recipe.get('cuda_activity_profile') if isinstance(recipe, dict) else None
    if profile is None:
        return STRICT_CUDA_ACTIVITY
    from scripts.ci import model_inference_proofs as proofs
    if profile == SHARED_CUDA_HEADROOM_ACTIVITY:
        proofs.apply_configuration(constraints, {})
        registered = next((row for row in proofs.load_registry()
                           if proofs.identity(row) == proofs.identity(constraints)), None)
        expected = proofs.configured_case(registered, recipe['name']) if registered is not None else None
        if constraints.get('training_comparison_stage'):
            from scripts.ci.training_comparison import stage_row
            expected = stage_row(expected, constraints['training_comparison_stage']) if expected is not None else None
        if (profile != proofs.SHARED_CUDA_HEADROOM_PROFILE or registered is None
                or constraints != expected):
            raise ValueError('shared CUDA headroom profile requires its exact registered source recipe')
        return profile
    binding = '/'.join(str(constraints.get(key)) for key in ('package', 'target', 'fqn'))
    if (profile != IRIS_CUDA_CORRECTNESS_ACTIVITY or profile != proofs.IRIS_CUDA_CORRECTNESS_PROFILE
            or binding not in proofs.IRIS_CUDA_CORRECTNESS_PROOFS
            or constraints.get('source') != 'crates/media/candle-gen/candle-gen-iris/tests/quant_real_weights.rs'
            or recipe.get('name') not in ('source_selected_cuda', 'source_selected_cuda_gpu0')):
        raise ValueError('CUDA low-activity profile is limited to the exact Iris correctness recipes')
    proofs.apply_configuration(constraints, {})
    return profile


def cuda_claim_diagnostic(uuid: str, devices: list[dict], processes: list[list[str]], policy,
                          *, activity_profile: str = STRICT_CUDA_ACTIVITY) -> dict:
    """Bounded facts from the already-read driver snapshot; never process command lines."""
    matched = [device for device in devices if device['uuid'] == uuid]
    selected = matched[0] if len(matched) == 1 else None
    rows = [row for row in processes if row and row[0] == uuid]
    def safe_uuid(value: str) -> str:
        return value if re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value) else 'invalid-uuid'
    types = [row[3] if len(row) == 4 and row[3] in ('G', 'C+G', 'C', 'unknown') else 'malformed-or-other' for row in rows]
    low_activity = activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY
    shared_headroom = activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY
    predicate_name = ('proof_shared_correctness_headroom' if shared_headroom else
                      'proof_correctness_profile_headroom_and_stable_contexts' if low_activity
                      else 'proof_policy_headroom_and_inactive_contexts')
    utilization = selected.get('utilization_percent') if selected is not None else None
    processes_unknown = selected is not None and selected.get('process_inventory_status') == 'unknown/not-collected-diagnostic-only'
    allowed_types = ('G', 'C+G') if low_activity and type(utilization) is int and utilization > 0 else ('G', 'C+G', 'C')
    return {'requested_uuid': safe_uuid(uuid), 'matched_uuid_count': len(matched),
            'activity_profile': activity_profile,
            'observed_uuids': [safe_uuid(device['uuid']) for device in devices[:8]],
            'observed_gpu_count': len(devices),
            'selected_card': (None if selected is None else {
                key: selected.get(key) for key in ('index', 'total_mib', 'used_mib', 'free_mib', 'utilization_percent')}),
            **({'headroom_limits': {'minimum_free_bytes': policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes,
                                    'background_activity_is_diagnostic': True}}
               if shared_headroom else
               {'activity_limits': {'minimum_free_bytes': policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes,
                                    'selected_utilization_percent_exclusive': MAX_BASELINE_UTILIZATION_PERCENT,
                                    'allowed_process_types_at_observed_utilization': list(allowed_types)}}
               if low_activity else
               {'idle_limits': {'minimum_free_bytes': policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes,
                                'selected_utilization_percent': 0,
                                'allowed_process_types': ['G', 'C+G', 'C']}}),
            'process_inventory_status': 'unknown/not-collected-diagnostic-only' if processes_unknown else 'observed',
            'selected_process_count': None if processes_unknown else len(rows),
            'selected_process_type_counts': None if processes_unknown else {kind: types.count(kind) for kind in ('G', 'C+G', 'C', 'unknown', 'malformed-or-other')},
            'selected_processes': [{'pid': row[1] if len(row) == 4 and row[1].isdecimal() and len(row[1]) <= 10 else 'invalid',
                                    'driver_type': kind} for row, kind in zip(rows[:64], types[:64])],
            'process_list_truncated': len(rows) > 64,
            'predicates': {'unique_uuid': len(matched) == 1,
                           predicate_name: selected is not None and proof_cuda_idle(
                               selected, processes, policy, activity_profile=activity_profile),
                           'free_memory_above_policy_headroom': selected is not None and
                               type(selected.get('free_mib')) is int and
                               selected['free_mib'] * 1024**2 >= policy.gpu_free_reserve_bytes + policy.child_gpu_cap_bytes,
                           **({} if shared_headroom else {
                           ('selected_gpu_below_profile_utilization_limit' if low_activity else 'selected_gpu_zero_utilization'):
                               type(utilization) is int and 0 <= utilization < (MAX_BASELINE_UTILIZATION_PERCENT if low_activity else 1),
                           'selected_process_types_allowed': all(len(row) == 4 and row[3] in allowed_types and row[1]
                                                                 for row in rows)})}}


def cuda_sample_diagnostic(output: Path, phase: str, uuid: str, physical_index: int, policy, baseline: dict,
                           *, activity_profile: str = STRICT_CUDA_ACTIVITY) -> dict:
    """Retain only bounded predicates from samples already collected by the guard."""
    path = output / f'{phase}-samples.json'
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            return {'status': 'missing-or-over-budget'}
        with path.open('rb') as stream:
            raw = stream.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            return {'status': 'missing-or-over-budget'}
        samples = json.loads(raw.decode('utf-8'))
        if not isinstance(samples, list) or not 1 <= len(samples) <= 5:
            raise ValueError('invalid release sample count')
        facts = []
        for sample in samples:
            if (not isinstance(sample, dict) or sample.get('sample_number') != len(facts) + 1
                    or type(sample.get('sample_number')) is not int
                    or not isinstance(sample.get('devices'), list) or not isinstance(sample.get('processes'), list)):
                raise ValueError('malformed release sample')
            devices, processes = sample['devices'], sample['processes']
            if any(not isinstance(device, dict) for device in devices) or any(
                    not isinstance(row, list) or len(row) != 4 or any(not isinstance(cell, str) for cell in row)
                    for row in processes):
                raise ValueError('malformed release inventory')
            observed = cuda_claim_diagnostic(uuid, devices, processes, policy, activity_profile=activity_profile)
            card = observed['selected_card']
            matched = [device for device in devices if device.get('uuid') == uuid]
            contexts_match = (None if observed['process_inventory_status'] != 'observed' else
                              proof_cuda_contexts(uuid, processes, activity_profile=activity_profile) == baseline['process_identities'])
            index_matches = card is not None and card.get('index') == physical_index
            facts.append({'sample_number': sample.get('sample_number'),
                          'selected_card': card,
                          'selected_index_matches_admission': index_matches,
                          'process_inventory_status': observed['process_inventory_status'],
                          'selected_process_count': observed['selected_process_count'],
                          'selected_process_type_counts': observed['selected_process_type_counts'],
                          'context_identities_match_admission': contexts_match,
                          'transient_utilization_only': (activity_profile != SHARED_CUDA_HEADROOM_ACTIVITY
                              and len(matched) == 1 and index_matches and contexts_match
                              and type(card.get('utilization_percent')) is int and card['utilization_percent'] > 0
                              and proof_cuda_idle({**matched[0], 'utilization_percent': 0}, processes, policy,
                                                  activity_profile=activity_profile)),
                          'predicates': observed['predicates']})
        return {'status': 'retained-existing-samples', 'activity_profile': activity_profile,
                'sample_count': len(facts), 'samples': facts}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return {'status': 'missing-or-malformed'}


def cuda_release_diagnostic(output: Path, uuid: str, physical_index: int, policy, baseline: dict,
                            *, activity_profile: str = STRICT_CUDA_ACTIVITY) -> dict:
    facts = cuda_sample_diagnostic(output, 'model-proof-release', uuid, physical_index, policy, baseline,
                                   activity_profile=activity_profile)
    if activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY:
        facts['release_policy'] = 'verified-owned-Job-exit; exact selected primary metadata; headroom/activity diagnostic only'
        facts['admission_headroom_is_diagnostic_only'] = True
        for sample in facts.get('samples', []):
            card = sample.get('selected_card') or {}
            sample['post_owned_exit_primary_metadata_valid'] = proof_cuda_shared_release(
                {**card, 'uuid': uuid} if sample['predicates']['unique_uuid'] else card,
                uuid, physical_index, baseline.get('admitted_total_mib'))
    return facts


def accepted_utilization_samples(output: Path, phase: str, uuid: str, physical_index: int,
                                 policy, baseline: dict, activity_profile: str) -> list[int | None]:
    """Keep the five actual selected-card readings, not an isolated benchmark claim."""
    facts = cuda_sample_diagnostic(output, phase, uuid, physical_index, policy, baseline,
                                   activity_profile=activity_profile)
    values = [(sample.get('selected_card') or {}).get('utilization_percent') for sample in facts.get('samples', [])]
    if (facts.get('status') != 'retained-existing-samples' or facts.get('sample_count') != 5
            or len(values) != 5 or any(type(value) is not int
                and not (value is None and activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY) for value in values)):
        raise ValueError('missing exact-card utilization readings for correctness proof')
    return values


def settled_cuda_samples(probe, output: Path, phase: str, uuid: str, physical_index: int,
                         policy, baseline: dict, deadline: float, *, activity_profile: str = STRICT_CUDA_ACTIVITY):
    """Require five fresh samples; strict profiles may retry bounded transient utilization."""
    failed_attempts = []
    for attempt in range(5):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'CUDA settle deadline expired during {phase}')
        try:
            result = probe()
            if time.monotonic() >= deadline:
                raise RuntimeError(f'CUDA settle deadline expired during {phase}')
            return result
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            diagnostic = cuda_sample_diagnostic(output, phase, uuid, physical_index, policy, baseline,
                                                activity_profile=activity_profile)
            failed_attempts.append(diagnostic)
            samples = diagnostic.get('samples', [])
            predicate_name = ('proof_correctness_profile_headroom_and_stable_contexts'
                              if activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY
                              else 'proof_policy_headroom_and_inactive_contexts')
            transient = (activity_profile != SHARED_CUDA_HEADROOM_ACTIVITY
                         and str(error) == f'physical GPU occupied during {phase}' and samples
                         and all(item['predicates'][predicate_name]
                                 and item['selected_index_matches_admission']
                                 and item['context_identities_match_admission'] for item in samples[:-1])
                         and samples[-1]['transient_utilization_only'])
            if not transient or attempt == 4 or time.monotonic() + 0.5 >= deadline:
                error.cuda_settle_attempts = failed_attempts
                raise
            time.sleep(0.5)


def cuda_settle_refusal(error: Exception, diagnostic: dict) -> dict:
    attempts = getattr(error, 'cuda_settle_attempts', None)
    if attempts:
        return {**diagnostic, 'settle_attempt_count': len(attempts), 'settle_attempts': attempts}
    return diagnostic


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


def _shared_admission_wait(reader, output: Path, uuid: str, physical_index: int,
                           policy, deadline: float, observation: dict):
    """Wait only for valid selected-card cap+reserve headroom before any body exists."""
    while True:
        if time.monotonic() >= deadline:
            raise CudaClaimRejected({**observation.get('last_diagnostic', {}),
                'admission_error': 'original CUDA claim deadline expired waiting for headroom',
                'admission_wait': dict(observation)})
        try:
            devices, processes = reader(output, 'model-proof-claimed-device', deadline)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            raise CudaClaimRejected({'requested_uuid': uuid, 'admission_error': str(error),
                                     'admission_wait': dict(observation)}) from error
        observation['primary_polls'] += 1
        diagnostic = cuda_claim_diagnostic(uuid, devices, processes, policy,
                                           activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY)
        observation['last_diagnostic'] = diagnostic
        matches = [device for device in devices if device.get('uuid') == uuid]
        # Identity/counter sanity only: the unchanged cap+reserve check below
        # remains the sole readiness decision; release policy never admits work.
        if len(matches) != 1 or not proof_cuda_shared_release(
                matches[0], uuid, physical_index,
                observation.get('admitted_total_mib', matches[0].get('total_mib'))):
            raise CudaClaimRejected({**diagnostic, 'admission_error': 'invalid selected primary identity/metadata',
                                     'admission_wait': dict(observation)})
        observation.setdefault('admitted_total_mib', matches[0]['total_mib'])
        if time.monotonic() >= deadline:
            raise CudaClaimRejected({**diagnostic, 'admission_error': 'original CUDA claim deadline expired after primary query',
                                     'admission_wait': dict(observation)})
        if proof_cuda_idle(matches[0], processes, policy, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY):
            return devices, processes
        observation['headroom_shortfalls'] += 1
        observation.setdefault('first_shortfall', diagnostic)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        pause = min(policy.poll_millis / 1000, remaining)
        time.sleep(pause)
        observation['headroom_wait_seconds'] += pause


def _shared_admission_samples(probe, reader, output: Path, uuid: str, physical_index: int,
                              policy, baseline: dict, deadline: float, observation: dict):
    """Restart five fresh samples only for valid headroom loss, within the same deadline."""
    while True:
        try:
            return settled_cuda_samples(probe, output, 'model-proof-admission', uuid, physical_index,
                policy, baseline, deadline, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            facts = cuda_sample_diagnostic(output, 'model-proof-admission', uuid, physical_index,
                                           policy, baseline, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY)
            samples = facts.get('samples', [])
            last = samples[-1] if samples else {}
            card = last.get('selected_card') or {}
            valid = (last.get('predicates', {}).get('unique_uuid') is True
                     and last.get('selected_index_matches_admission') is True
                     and proof_cuda_shared_release({**card, 'uuid': uuid}, uuid, physical_index,
                                                   baseline['admitted_total_mib']))
            if (str(error) != 'physical GPU occupied during model-proof-admission' or not valid
                    or last.get('predicates', {}).get('free_memory_above_policy_headroom') is not False):
                raise
            observation['stable_sample_restarts'] += 1
            _shared_admission_wait(reader, output, uuid, physical_index, policy, deadline, observation)


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
    record = None
    shared_release_verified = False
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
        activity_profile = bound_cuda_activity_profile(constraints)
        shared_headroom = activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY
        inventory_reader = (lambda out, label, deadline: veda_cuda_guard.inventory(
            out, label, min(deadline, started + policy.deadline_seconds), selected_uuid=uuid)) if shared_headroom else veda_cuda_guard.inventory
        admission_wait = {'primary_polls': 0, 'headroom_shortfalls': 0,
                          'stable_sample_restarts': 0, 'headroom_wait_seconds': 0.0}
        devices, processes = (_shared_admission_wait(inventory_reader, output, uuid,
            constraints['required_cuda_physical_index'], policy, started + policy.deadline_seconds, admission_wait)
            if shared_headroom else veda_cuda_guard.inventory(output, 'model-proof-claimed-device'))
        matched = [device for device in devices if device['uuid'] == uuid]
        required_index = constraints.get('required_cuda_physical_index')
        if len(matched) == 1 and required_index is not None and matched[0]['index'] != required_index:
            raise ValueError('claimed UUID violates this case fixed physical-index restriction')
        idle_check = lambda device, rows: (device.get('uuid') == uuid and
                                           (required_index is None or device.get('index') == required_index) and
                                           proof_cuda_idle(device, rows, policy, activity_profile=activity_profile))
        if (len(matched) != 1 or not idle_check(matched[0], processes)
                and not (type(matched[0].get('utilization_percent')) is int
                         and matched[0]['utilization_percent'] > 0
                         and idle_check({**matched[0], 'utilization_percent': 0}, processes))):
            raise CudaClaimRejected(cuda_claim_diagnostic(uuid, devices, processes, policy,
                                                          activity_profile=activity_profile))
        record = {'mechanism': 'existing PUBLIC/SceneWorks/gpu-locks UUID lock', 'device_uuid': uuid,
                  'physical_index': matched[0]['index'], 'runner': env.get('RUNNER_NAME'),
                  'activity_profile': activity_profile, 'claim_utilization_percent': matched[0]['utilization_percent']}
        if shared_headroom:
            record['activity_policy'] = ('non-isolated correctness; exact-card identity and policy cap plus reserve free; '
                                         'background utilization and contexts are diagnostic only')
            record['isolated_benchmark_claim'] = False
            record['lease_released'] = False
            record['claim_deadline_monotonic'] = started + policy.deadline_seconds
            record['admission_wait'] = admission_wait
            record['release_policy'] = 'verified-owned-Job-exit; exact selected primary metadata; headroom/activity diagnostic only'
        elif activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY:
            record['activity_policy'] = ('non-isolated correctness, stable selected-card G/C+G contexts when utilization '
                                         'is nonzero; C also permitted at zero; below '
                                         f'{MAX_BASELINE_UTILIZATION_PERCENT}% utilization, policy cap plus reserve free')
            record['isolated_benchmark_claim'] = False
        else:
            record['idle_policy'] = 'stable typed contexts, zero selected-card utilization, policy cap plus reserve free'
        if constraints.get('child_cuda_visibility') is not None:
            record['child_visibility_binding'] = child_visibility_binding(
                constraints, env, uuid, policy.deadline_seconds - (time.monotonic() - started))
        # Reuse five fresh samples, with this proof's actual free reserve and child cap.
        context_baseline = {'uuid': uuid, 'reserved_baseline_mib': matched[0]['used_mib'],
                            'process_identities': proof_cuda_contexts(uuid, processes, activity_profile=activity_profile)}
        if shared_headroom:
            context_baseline['admitted_total_mib'] = matched[0]['total_mib']
        identity_key = lambda selected_uuid, rows: proof_cuda_contexts(
            selected_uuid, rows, activity_profile=activity_profile)
        settle_deadline = (started + policy.deadline_seconds if shared_headroom
                           else min(started + policy.deadline_seconds, time.monotonic() + 30))
        probe = lambda: veda_cuda_guard.sample_baseline(
            matched[0], output, 'model-proof-admission', context_baseline, idle_check=idle_check,
            identity_key=identity_key, permit_bounded_growth=True,
            require_stable_contexts=not shared_headroom, inventory_reader=inventory_reader)
        try:
            if shared_headroom:
                selected, baseline = _shared_admission_samples(probe, inventory_reader, output, uuid,
                    matched[0]['index'], policy, context_baseline, settle_deadline, admission_wait)
            else:
                selected, baseline = settled_cuda_samples(probe, output, 'model-proof-admission', uuid,
                    matched[0]['index'], policy, context_baseline, settle_deadline, activity_profile=activity_profile)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            diagnostic = cuda_sample_diagnostic(output, 'model-proof-admission', uuid, matched[0]['index'], policy,
                                                context_baseline, activity_profile=activity_profile)
            raise CudaClaimRejected({**cuda_claim_diagnostic(uuid, devices, processes, policy,
                                                              activity_profile=activity_profile),
                                     'admission_samples': cuda_settle_refusal(error, diagnostic),
                                     'admission_error': str(error),
                                     **({'admission_wait': admission_wait} if shared_headroom else {})}) from error
        record['post_execution_verified'] = False
        record['admission_baseline'] = baseline
        if activity_profile in (IRIS_CUDA_CORRECTNESS_ACTIVITY, SHARED_CUDA_HEADROOM_ACTIVITY):
            record['admission_utilization_percent'] = accepted_utilization_samples(
                output, 'model-proof-admission', uuid, matched[0]['index'], policy, baseline, activity_profile)
        if shared_headroom:
            record['admission_background_activity'] = cuda_sample_diagnostic(
                output, 'model-proof-admission', uuid, matched[0]['index'], policy, baseline,
                activity_profile=activity_profile)
        (output / 'physical-execution-claim.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        try:
            yield record
        finally:
            try:
                if shared_headroom and not shared_owned_completion(record):
                    raise RuntimeError('supervisor-owned Windows Job completion is unverified; no release seal')
                release_check = (lambda device, rows: proof_cuda_shared_release(
                    device, uuid, matched[0]['index'], baseline['admitted_total_mib'])) if shared_headroom else idle_check
                settled_cuda_samples(
                    lambda: veda_cuda_guard.verify_idle(
                        selected, output, 'model-proof-release', baseline, idle_check=release_check,
                        identity_key=identity_key, permit_bounded_growth=True,
                        require_stable_contexts=not shared_headroom, inventory_reader=inventory_reader),
                    output, 'model-proof-release', uuid, matched[0]['index'], policy, baseline,
                    min(started + policy.deadline_seconds, time.monotonic() + 30),
                    activity_profile=activity_profile)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                raise CudaReleaseRejected('CUDA post-idle verification failed: ' + str(error),
                                          cuda_settle_refusal(
                                              error, cuda_release_diagnostic(output, uuid, matched[0]['index'], policy,
                                                                             baseline, activity_profile=activity_profile))) from error
            if activity_profile in (IRIS_CUDA_CORRECTNESS_ACTIVITY, SHARED_CUDA_HEADROOM_ACTIVITY):
                record['release_utilization_percent'] = accepted_utilization_samples(
                    output, 'model-proof-release', uuid, matched[0]['index'], policy, baseline, activity_profile)
            if shared_headroom:
                record['release_background_activity'] = cuda_release_diagnostic(
                    output, uuid, matched[0]['index'], policy, baseline, activity_profile=activity_profile)
            if shared_headroom:
                shared_release_verified = True
            else:
                record['post_execution_verified'] = True
    finally:
        shared_record = record is not None and record.get('activity_profile') == SHARED_CUDA_HEADROOM_ACTIVITY
        if not shared_record:
            # Preserve the original release behavior for all other profiles.
            if claimed:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            handle.close()
        else:
            try:
                if claimed:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    record['lease_released'] = True
            finally:
                handle.close()
            if shared_release_verified:
                record['post_execution_verified'] = True
