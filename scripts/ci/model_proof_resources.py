"""Reuse existing execution coordination and supervisor policies; never submission/job-count gates."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
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
# Post-run release sampling gets its own fixed per-query budget. It never inherits the admission
# claim deadline or the guard's five-second admission window (run 44431476698: a passing proof
# was rejected when one nvidia-smi query on a busy shared host got 2.7 s).
RELEASE_QUERY_SECONDS = 30
RELEASE_QUERY_ATTEMPTS = 3
RELEASE_SAMPLING_SECONDS = 5 * (RELEASE_QUERY_SECONDS * RELEASE_QUERY_ATTEMPTS + 1)


# A job that states no device estimate needs its child cap free (plus the reserve), as before
# claims stated estimates; claims recorded before then carry no estimate and replay the same way.
LEGACY_CAP_CLAIM_ESTIMATE_SOURCE = 'child-gpu-cap'


def claim_gpu_estimate(policy) -> tuple[str, int]:
    """The device estimate a CUDA claim needs free beyond the reserve.

    The caller's stated estimate (`policy.claim_gpu_estimate`, from reviewed peaks of that lane),
    else the child cap: a job with no recorded peak may really fill its cap. A small-cap job still
    admits on a busy card; the child cap and live reserve watchdog bound every run.
    """
    cap = policy.child_gpu_cap_bytes
    stated = getattr(policy, 'claim_gpu_estimate', None)
    if stated is None:
        return LEGACY_CAP_CLAIM_ESTIMATE_SOURCE, cap
    source, value = stated
    if (not isinstance(source, str) or not source or type(value) is not int
            or type(cap) is not int or not 0 < value <= cap):
        raise ValueError('CUDA claim estimate must name its source and fit under the child GPU cap')
    return source, value


def claim_required_free_bytes(policy) -> int:
    return policy.gpu_free_reserve_bytes + claim_gpu_estimate(policy)[1]


def claim_estimate_record(policy) -> dict:
    source, value = claim_gpu_estimate(policy)
    return {'source': source, 'estimate_bytes': value, 'reserve_bytes': policy.gpu_free_reserve_bytes,
            'required_free_bytes': policy.gpu_free_reserve_bytes + value}


def replay_claim_policy(policy, claim: dict):
    """The policy a verifier replays a claim's headroom checks under: the estimate the claim
    recorded, or the child cap for a claim recorded before claims stated one."""
    import dataclasses
    recorded = claim.get('claim_gpu_estimate')
    if recorded is None:
        stated = (LEGACY_CAP_CLAIM_ESTIMATE_SOURCE, policy.child_gpu_cap_bytes)
    else:
        if (not isinstance(recorded, dict)
                or set(recorded) != {'source', 'estimate_bytes', 'reserve_bytes', 'required_free_bytes'}
                or recorded['reserve_bytes'] != policy.gpu_free_reserve_bytes):
            raise ValueError('malformed recorded CUDA claim estimate')
        stated = (recorded['source'], recorded['estimate_bytes'])
    replay = (dataclasses.replace(policy, claim_gpu_estimate=stated) if dataclasses.is_dataclass(policy)
              else type(policy)(**{**vars(policy), 'claim_gpu_estimate': stated}))
    if recorded is not None and claim_estimate_record(replay) != recorded:
        raise ValueError('recorded CUDA claim estimate contradicts its reserve or cap')
    return replay


class CudaClaimRejected(ValueError):
    def __init__(self, diagnostic: dict, message='claimed CUDA UUID is absent or occupied; no model execution'):
        super().__init__(message)
        self.diagnostic = diagnostic


class CudaCardTaken(CudaClaimRejected):
    """The claim gave up waiting for its card because the caller can run on another one."""


class _AdmissionInventoryTimeout(TimeoutError):
    """The selected GPU query exceeded its bounded five-sample admission window."""


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
    required = claim_required_free_bytes(policy)
    if activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY:
        # Foreign utilization/contexts are observations, not ownership or safety prerequisites.
        return (all(type(value) is int for value in (total, used, free, reserve, cap))
                and total > 0 and 0 <= used <= total and 0 <= free <= total
                and used + free <= total + 2 and reserve > 0 and cap > 0
                and free * mib >= required)
    limit = (MAX_BASELINE_UTILIZATION_PERCENT if activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY
             else 1 if activity_profile == STRICT_CUDA_ACTIVITY else None)
    if (limit is None or any(type(value) is not int for value in (total, used, free, utilization, reserve, cap))
            or total <= 0 or used < 0 or free < 0 or used > total or free > total
            or used + free > total + 2 or utilization < 0 or utilization >= limit or reserve <= 0 or cap <= 0
            or free * mib < required or any(len(row) != 4 for row in processes)):
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


def _valid_shared_admission_card(device: dict, uuid: str, physical_index: int, total_mib: int) -> bool:
    # The selected inventory reader maps an explicit [N/A] to None.
    utilization = device.get('utilization_percent')
    return (proof_cuda_shared_release(device, uuid, physical_index, total_mib)
            and 'utilization_percent' in device
            and (utilization is None or type(utilization) is int and 0 <= utilization <= 100))


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
    # Headroom admission is the default for every single-UUID CUDA row; multi-UUID rows
    # (required_cuda_ordinals) carry their own lease record.
    expected_shared = (constraints is not None and constraints.get('backend') == 'candle-cuda'
                       and constraints.get('required_cuda_ordinals') is None)
    if not expected_shared and record.get('activity_profile') != SHARED_CUDA_HEADROOM_ACTIVITY:
        return
    required_uuid = (constraints or {}).get('required_cuda_device_uuid')
    required_index = (constraints or {}).get('required_cuda_physical_index')
    if constraints is not None and (not expected_shared
            or record.get('activity_profile') != SHARED_CUDA_HEADROOM_ACTIVITY
            or (required_uuid is not None and record.get('device_uuid') != required_uuid)
            or type(record.get('physical_index')) is not int
            or (required_index is not None and record['physical_index'] != required_index)):
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
    """Every CUDA claim admits on free-memory headroom held stable under the UUID lock.

    A declared registry profile is still authenticated against its exact source recipe, but
    no profile selects idle (zero-utilization / no-foreign-context) admission any more.
    """
    recipe = constraints.get('selected_configuration')
    profile = recipe.get('cuda_activity_profile') if isinstance(recipe, dict) else None
    if profile is None:
        return SHARED_CUDA_HEADROOM_ACTIVITY
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
    return SHARED_CUDA_HEADROOM_ACTIVITY


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
            **({'headroom_limits': {'minimum_free_bytes': claim_required_free_bytes(policy),
                                    'background_activity_is_diagnostic': True}}
               if shared_headroom else
               {'activity_limits': {'minimum_free_bytes': claim_required_free_bytes(policy),
                                    'selected_utilization_percent_exclusive': MAX_BASELINE_UTILIZATION_PERCENT,
                                    'allowed_process_types_at_observed_utilization': list(allowed_types)}}
               if low_activity else
               {'idle_limits': {'minimum_free_bytes': claim_required_free_bytes(policy),
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
                               selected['free_mib'] * 1024**2 >= claim_required_free_bytes(policy),
                           **({} if shared_headroom else {
                           ('selected_gpu_below_profile_utilization_limit' if low_activity else 'selected_gpu_zero_utilization'):
                               type(utilization) is int and 0 <= utilization < (MAX_BASELINE_UTILIZATION_PERCENT if low_activity else 1),
                           'selected_process_types_allowed': all(len(row) == 4 and row[3] in allowed_types and row[1]
                                                                 for row in rows)})}}


def cuda_sample_diagnostic(output: Path, phase: str, uuid: str, physical_index: int, policy, baseline: dict,
                           *, activity_profile: str = STRICT_CUDA_ACTIVITY, allow_empty: bool = False) -> dict:
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
        if not isinstance(samples, list) or not (0 if allow_empty else 1) <= len(samples) <= 5:
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


def release_inventory_reader(guard, uuid: str):
    """Selected-card release reads with a fixed budget per nvidia-smi query and a bounded retry."""
    def read(out, label, _window_deadline=None):
        for attempt in range(RELEASE_QUERY_ATTEMPTS):
            try:
                return guard.inventory(out, label, time.monotonic() + RELEASE_QUERY_SECONDS, selected_uuid=uuid)
            except (subprocess.TimeoutExpired, TimeoutError):
                if attempt == RELEASE_QUERY_ATTEMPTS - 1:
                    raise
    return read


def release_window_overrun(phase: str) -> str:
    """The guard's message when its five-sample window outlasts MAX_SAMPLING_SECONDS. A
    selected-UUID reader already refuses an absent card, so for it this means only time."""
    return f'physical GPU identity or sampling deadline changed during {phase}'


def release_check_inconclusive(error: Exception, diagnostic: dict, phase: str = 'model-proof-release') -> bool:
    """A query timeout or sampling-time overrun is not evidence against the release.

    It is inconclusive only when at least one sample was retained after owned exit and every
    retained sample shows valid exact-card metadata; with none (a hung driver) nothing was checked
    and the release is refused. The owned-Job completion check (`shared_owned_completion`) stays
    the proof that this proof's process exited.
    """
    timing = (isinstance(error, (subprocess.TimeoutExpired, TimeoutError))
              or (type(error) is RuntimeError and str(error) in (
                  release_window_overrun(phase), f'CUDA settle deadline expired during {phase}')))
    samples = diagnostic.get('samples') or []
    return timing and bool(samples) and all(sample.get('post_owned_exit_primary_metadata_valid') is True
                                            for sample in samples)


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


WAIT_NOTICE_SECONDS = 60


def _wait_notice(observation: dict, reason: str) -> None:
    """Log every refusal: a new kind of reason at once, a repeated one at most once per minute.

    Numbers inside a reason (free MiB, utilization percent) do not make it a new kind, so a
    steady refusal cannot flood the log, and a wait is never silent.
    """
    now = time.monotonic()
    kind = re.sub(r'\d+', 'N', reason)
    if (kind != observation.get('_last_kind')
            or now - observation.get('_last_notice', float('-inf')) >= WAIT_NOTICE_SECONDS):
        observation['_last_notice'] = now
        observation['_last_kind'] = kind
        print('CUDA admission waiting: ' + reason, file=sys.stderr, flush=True)


# The Windows CUDA host's two RTX PRO 6000 cards: UUID -> nvidia-smi physical index. GPU 1 also
# drives the display. A proof that names neither card may run on whichever one fits.
REGISTERED_CUDA_CARDS = {
    'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0': 0,
    'GPU-e4b79931-7be6-f216-460a-f5405cfafffe': 1,
}
CARD_SELECTION_RULE = 'most-free-card-covering-estimate-plus-reserve-v1'


def card_free_policy(payload: dict) -> bool:
    """A Windows CUDA request policy that names no card: the proof runs on whichever card fits."""
    return isinstance(payload, dict) and payload.get('backend') == 'windows-cuda' and 'cudaDeviceUuid' not in payload


# The single driver read a switch makes before giving up its card (no polling, its own budget).
SWITCH_READ_SECONDS = 30


def select_cuda_card(policies: dict, deadline: float, *, reader=None, pause=time.sleep,
                     clock=time.monotonic, single_read: bool = False, notice: dict | None = None) -> tuple[str, dict]:
    """Choose the card a proof that names none runs on, from `policies` ({UUID: its policy}).

    Each poll reads every candidate's free memory. A card is refused (and the reason logged, at most
    once a minute per kind) when it is absent, has moved index, or has less free than its reserve
    plus the claim estimate. Among the cards that fit, the one with the most free memory is chosen
    at once: the proof never waits on a short card while another fits. Only when no card fits does
    it wait, polling every card, until `deadline`. `single_read` makes one read with its own
    SWITCH_READ_SECONDS driver budget and refuses at once when nothing fits (a switch never waits).
    The chosen card is then claimed normally (lock, stable window).
    """
    from scripts.release import veda_cuda_guard
    if not policies or any(uuid not in REGISTERED_CUDA_CARDS or policy.cuda_device_uuid != uuid
                           for uuid, policy in policies.items()):
        raise ValueError('CUDA card selection is limited to the registered host cards')
    reader = reader or (lambda uuid, until: veda_cuda_guard.inventory(
        None, 'model-proof-card-selection', until, selected_uuid=uuid)[0])
    notice = {} if notice is None else notice
    polls, started = 0, clock()
    while True:
        polls += 1
        cards = []
        read_until = time.monotonic() + SWITCH_READ_SECONDS if single_read else deadline
        for uuid in sorted(policies, key=REGISTERED_CUDA_CARDS.get):
            policy = policies[uuid]
            source, estimate = claim_gpu_estimate(policy)
            required = claim_required_free_bytes(policy)
            entry = {'uuid': uuid, 'index': REGISTERED_CUDA_CARDS[uuid], 'estimate_source': source,
                     'estimate_bytes': estimate, 'required_free_bytes': required}
            try:
                devices = reader(uuid, read_until)
            except (RuntimeError, OSError, subprocess.SubprocessError, TimeoutError) as error:
                devices, entry['refusal'] = [], f'inventory failed: {str(error)[:200]}'
            matched = [device for device in devices if device.get('uuid') == uuid]
            if 'refusal' in entry:
                pass
            elif len(matched) != 1:
                entry['refusal'] = 'card absent from the driver inventory'
            elif matched[0].get('index') != REGISTERED_CUDA_CARDS[uuid]:
                entry['refusal'] = f"card moved to index {matched[0].get('index')}"
            else:
                card = matched[0]
                entry.update({key: card.get(key) for key in ('total_mib', 'used_mib', 'free_mib')})
                if type(card.get('free_mib')) is not int or card['free_mib'] * 1024**2 < required:
                    entry['refusal'] = (f"{card.get('free_mib')} MiB free < {-(-required // 1024**2)} MiB "
                                        f"required (reserve {policy.gpu_free_reserve_bytes} + estimate "
                                        f'{estimate} bytes, {source})')
            cards.append(entry)
        fits = [entry for entry in cards if 'refusal' not in entry]
        for entry in cards:
            if 'refusal' in entry:
                _wait_notice(notice, f"card {entry['index']} {entry['uuid']} refused: {entry['refusal']}"
                             + ('' if fits else '; no card fits' + ('' if single_read else ', waiting')))
        if fits:
            chosen = max(fits, key=lambda entry: (entry['free_mib'], -entry['index']))
            print(f"CUDA card selected: index {chosen['index']} {chosen['uuid']} "
                  f"({chosen['free_mib']} MiB free, the most of {len(fits)} fitting card(s))",
                  file=sys.stderr, flush=True)
            return chosen['uuid'], {'rule': CARD_SELECTION_RULE, 'chosen_uuid': chosen['uuid'],
                                    'chosen_index': chosen['index'], 'polls': polls,
                                    'wait_seconds': clock() - started, 'cards': cards}
        remaining = deadline - clock()
        if single_read or remaining <= 0:
            raise CudaClaimRejected({'rule': CARD_SELECTION_RULE, 'polls': polls, 'cards': cards,
                                     'admission_error': 'no registered CUDA card had estimate plus reserve free'},
                                    'no registered CUDA card fits this proof; no model execution')
        pause(min(max(min(policy.poll_millis for policy in policies.values()) / 1000, 1.0), remaining))


def planned_card_policy(payload: dict) -> dict:
    """The policy a planner validates: a card-free policy is checked with a registered card filled in."""
    return dict(payload, cudaDeviceUuid=next(iter(REGISTERED_CUDA_CARDS))) if card_free_policy(payload) else payload


def card_candidates(payload: dict, work: Path, uuids, estimate=None) -> dict:
    """The policy each candidate card would run under (with the stated claim estimate, if any)."""
    import dataclasses
    from scripts.media_campaign_supervisor import load_policy
    candidates = {}
    for uuid in sorted(uuids, key=lambda u: REGISTERED_CUDA_CARDS.get(u, -1)):
        if uuid not in REGISTERED_CUDA_CARDS:
            raise ValueError(f'unregistered CUDA card {uuid}')
        path = work / f'card-candidate-{uuid}.json'
        path.write_text(json.dumps(dict(payload, cudaDeviceUuid=uuid), sort_keys=True), encoding='utf-8')
        policy = load_policy(path)
        candidates[uuid] = dataclasses.replace(policy, claim_gpu_estimate=estimate) if estimate else policy
    return candidates


def choose_card(payload: dict, work: Path, *, named=None, estimate=None, deadline_monotonic=None,
                **selector) -> tuple[dict, dict | None]:
    """Resolve a request policy to the policy a proof runs under.

    A policy that names a card is returned unchanged (no selection). A card-free policy runs on the
    registered card (or the one card `named` by its case) with the most free memory covering
    reserve plus `estimate` (source, bytes; default the child cap); the record is the receipt's
    `cuda_card_selection`. Selection waits at most until `deadline_monotonic` (default: the
    policy's deadline from now), the same budget the following claim then uses.
    """
    if not card_free_policy(payload):
        return payload, None
    deadline = time.monotonic() + payload['deadlineSeconds'] if deadline_monotonic is None else deadline_monotonic
    candidates = card_candidates(payload, work, [named] if named else REGISTERED_CUDA_CARDS, estimate)
    chosen, record = select_cuda_card(candidates, deadline, **selector)
    return dict(payload, cudaDeviceUuid=chosen), record


MAX_CARD_SWITCHES = 4


def card_switcher(payload: dict, work: Path, selection: dict, *, estimate=None, **selector):
    """An `abandon_wait` for a card-free proof's first claim: when its card is taken, read the other
    registered card(s) once (own driver budget, no waiting). If one fits now, the switch is recorded
    (`superseded`) in a new selection and the claim gives up its card; if none fits, the claim keeps
    waiting for its card. Refusals are logged at most once a minute per kind across polls. Returns
    (abandon_wait, state) where state['selection'] is the selection now in force."""
    state = {'selection': selection, 'notice': {}}

    def abandon(reason: str) -> bool:
        current = state['selection']
        history = current.get('superseded', [])
        if len(history) >= MAX_CARD_SWITCHES:
            return False
        left = current['chosen_uuid']
        try:
            candidates = card_candidates(payload, work, [uuid for uuid in REGISTERED_CUDA_CARDS if uuid != left], estimate)
            _, record = select_cuda_card(candidates, time.monotonic(), single_read=True,
                                         notice=state['notice'], **selector)
        except CudaClaimRejected:
            return False
        earlier = {key: value for key, value in current.items() if key != 'superseded'}
        record['superseded'] = [*history, {'reason': reason, 'left_uuid': left,
                                           'switched_to': record['chosen_uuid'], 'selection': earlier}]
        print(f"CUDA card switched: {left} is taken ({reason}); running on {record['chosen_uuid']}",
              file=sys.stderr, flush=True)
        state['selection'] = record
        return True

    return abandon, state


@contextmanager
def claim_chosen_card(backend: str, request_payload: dict, payload: dict, selection, output: Path, env: dict,
                      *, claim_policy, rebind=None, record=None, estimate=None, constraints=None,
                      claim_with=None, selector=None, **claim):
    """Claim the card a proof chose and yield the claim record.

    For a card-free request (a recorded `selection`), a card taken before the claim holds it (lock
    held by another job, or short of estimate plus reserve) is switched for another registered card
    that fits now: `rebind(new_payload)` re-binds the caller's policy, hardware and environment, and
    `record['cuda_card_selection']` takes the new selection. A named card (no selection) waits as
    before. `claim_policy(payload)` and `constraints(payload)` give the claim's policy and
    constraints for the card in force; each switched attempt claims in its own subdirectory.
    `claim_with` is the caller's execution_claim (default this module's); `selector` passes
    select_cuda_card options (a test's driver reader) to the switch.
    """
    claim_with = execution_claim if claim_with is None else claim_with
    abandon, state = (card_switcher(request_payload, output, selection, estimate=estimate, **(selector or {}))
                      if selection is not None and backend == 'candle-cuda' else (None, None))
    attempt = 1
    while True:
        stack = ExitStack()
        target = output if attempt == 1 else output / f'switch-{attempt}'
        target.mkdir(exist_ok=True)
        try:
            options = dict(claim)
            if abandon is not None:
                options['abandon_wait'] = abandon
            bound = constraints(payload) if callable(constraints) else constraints
            if bound is not None:
                options['constraints'] = bound
            held = stack.enter_context(claim_with(backend, claim_policy(payload), target, env, **options))
        except CudaCardTaken:
            stack.close()
            if state is None:
                raise  # the caller's own abandon_wait: it switches
            payload = dict(request_payload, cudaDeviceUuid=state['selection']['chosen_uuid'])
            if record is not None:
                record['cuda_card_selection'] = state['selection']
            if rebind is not None:
                rebind(payload)
            attempt += 1
            continue
        break
    with stack:
        yield held


def _cards_read(record, payload: dict, expected) -> str:
    """The card one selection chose, after checking it read exactly `expected` and chose the most
    free card that fit."""
    if not isinstance(record, dict) or record.get('rule') != CARD_SELECTION_RULE:
        raise ValueError('card-free proof lacks its recorded most-free registered card selection')
    cards, chosen = record.get('cards'), record.get('chosen_uuid')
    uuids = [card.get('uuid') for card in cards] if isinstance(cards, list) else None
    fitting = [card for card in cards if 'refusal' not in card] if uuids else []
    reserve = payload.get('gpuFreeReserveBytes')
    if (uuids != sorted(expected, key=REGISTERED_CUDA_CARDS.get)
            or chosen not in uuids or record.get('chosen_index') != REGISTERED_CUDA_CARDS[chosen]
            or any(card.get('index') != REGISTERED_CUDA_CARDS[card['uuid']]
                   or type(card.get('estimate_bytes')) is not int
                   or not 0 < card['estimate_bytes'] <= payload.get('childGpuCapBytes', 0)
                   or card.get('required_free_bytes') != reserve + card['estimate_bytes'] for card in cards)
            or any(type(card.get('free_mib')) is not int
                   or card['free_mib'] * 1024**2 < card['required_free_bytes'] for card in fitting)
            or chosen != max(fitting, key=lambda card: (card['free_mib'], -card['index']))['uuid']):
        raise ValueError('card-free proof lacks its recorded most-free registered card selection')
    return chosen


SELECTION_REFUSAL = 'cuda_card_selection_refusal'


def record_selection_refusal(receipt: dict, error: BaseException) -> bool:
    """Keep a card selection's refusal (no registered card fit before the deadline) in the receipt,
    as claim refusals keep theirs, so the unsuccessful run still publishes verifiable evidence."""
    diagnostic = getattr(error, 'diagnostic', None)
    if (isinstance(error, CudaClaimRejected) and isinstance(diagnostic, dict)
            and diagnostic.get('rule') == CARD_SELECTION_RULE and 'chosen_uuid' not in diagnostic):
        receipt[SELECTION_REFUSAL] = diagnostic
        return True
    return False


def executed_card_policy(payload: dict, selection, *, named=None, estimate=None, refusal=None) -> dict | None:
    """The policy a verified proof ran under: the request's own when it names a card (and then no
    selection may be recorded), else the request's policy with the card its recorded selection
    chose. The first selection must have read every registered card (or exactly the one card its
    case names); each switch (`superseded`) must have read only the other card(s), chosen one that
    fit, and been made from the card the previous selection chose."""
    if not card_free_policy(payload):
        if selection is not None or refusal is not None:
            raise ValueError('a named-card proof records no card selection')
        return payload
    if selection is None and refusal is not None:
        # No card fit before the deadline: nothing ran. The refusal must have read every candidate
        # and refused each one.
        cards = refusal.get('cards') if isinstance(refusal, dict) else None
        expected = sorted([named] if named else REGISTERED_CUDA_CARDS, key=REGISTERED_CUDA_CARDS.get)
        if (refusal.get('rule') != CARD_SELECTION_RULE or not isinstance(cards, list)
                or [card.get('uuid') for card in cards] != expected
                or any(not isinstance(card.get('refusal'), str) or not card['refusal'] for card in cards)):
            raise ValueError('card-free proof records an invalid card selection refusal')
        return None
    if refusal is not None:
        raise ValueError('a proof that selected a card records no selection refusal')
    superseded = selection.get('superseded', []) if isinstance(selection, dict) else None
    if not isinstance(superseded, list) or (named and superseded):
        raise ValueError('card-free proof records an invalid card re-selection')
    first = [named] if named else list(REGISTERED_CUDA_CARDS)
    previous, expected = None, first
    for earlier in superseded:
        if (not isinstance(earlier, dict) or not isinstance(earlier.get('reason'), str) or not earlier['reason']
                or _cards_read(earlier.get('selection'), payload, expected) != earlier.get('left_uuid')
                or earlier['left_uuid'] == earlier.get('switched_to')):
            raise ValueError('card-free proof records an invalid card re-selection')
        previous = earlier
        expected = [uuid for uuid in REGISTERED_CUDA_CARDS if uuid != earlier['left_uuid']]
    chosen = _cards_read(selection, payload, expected)
    if previous is not None and previous['switched_to'] != chosen:
        raise ValueError('card-free proof records an invalid card re-selection')
    every = [selection, *(earlier['selection'] for earlier in superseded)]
    if estimate is not None and any((card['estimate_source'], card['estimate_bytes']) != tuple(estimate)
                                    for record in every for card in record['cards']):
        raise ValueError('card-free proof selected on a different estimate than it states')
    return dict(payload, cudaDeviceUuid=chosen)


def require_claimed_card(claim, executed: dict, estimate=None) -> None:
    """A claim record must hold the card the verified policy names, and (when the lane states an
    estimate) record that estimate; a claim from before claims recorded one carries none."""
    if not isinstance(claim, dict) or claim.get('device_uuid') != executed.get('cudaDeviceUuid'):
        raise ValueError('claim holds a different card than the verified policy or selection')
    recorded = claim.get('claim_gpu_estimate')
    if estimate is not None and recorded is not None and (recorded.get('source'), recorded.get('estimate_bytes')) != tuple(estimate):
        raise ValueError('claim recorded a different device estimate than its lane states')


def _shared_admission_wait(reader, output: Path, uuid: str, physical_index: int | None,
                           policy, deadline: float, observation: dict, notice: dict | None = None,
                           pause_unlocked=time.sleep):
    """Wait only for valid selected-card estimate-plus-reserve headroom before any body exists.

    `pause_unlocked` releases the UUID lock while waiting so other CI jobs that fit can run.
    """
    notice = {} if notice is None else notice
    while True:
        if time.monotonic() >= deadline:
            raise CudaClaimRejected({**observation.get('last_diagnostic', {}),
                'admission_error': 'original CUDA claim deadline expired waiting for headroom',
                'admission_wait': dict(observation)})
        try:
            devices, processes = reader(output, 'model-proof-claimed-device', deadline)
        except (subprocess.TimeoutExpired, TimeoutError) as error:
            # A slow driver query is an inventory failure, not a refusal: log it and retry within
            # the claim deadline. Every other inventory error still refuses at once.
            observation['inventory_timeouts'] = observation.get('inventory_timeouts', 0) + 1
            observation['last_inventory_timeout'] = str(error)[:512]
            observation['last_refusal'] = 'nvidia-smi inventory query timed out'
            _wait_notice(notice, f'{uuid}: nvidia-smi inventory query timed out; retrying')
            remaining = deadline - time.monotonic()
            if remaining > 0:
                pause = min(policy.poll_millis / 1000, remaining)
                pause_unlocked(pause)
                observation['inventory_timeout_wait_seconds'] = (
                    observation.get('inventory_timeout_wait_seconds', 0.) + pause)
            continue
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            raise CudaClaimRejected({'requested_uuid': uuid, 'admission_error': str(error),
                                     'admission_wait': dict(observation)}) from error
        observation['primary_polls'] += 1
        diagnostic = cuda_claim_diagnostic(uuid, devices, processes, policy,
                                           activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY)
        observation['last_diagnostic'] = diagnostic
        matches = [device for device in devices if device.get('uuid') == uuid]
        if physical_index is None and len(matches) == 1:
            # No source-pinned index: bind the index observed for this UUID on the first poll.
            physical_index = matches[0].get('index')
        # Identity/counter sanity only: the unchanged cap+reserve check below
        # remains the sole readiness decision; release policy never admits work.
        if len(matches) != 1 or not _valid_shared_admission_card(
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
        card = diagnostic.get('selected_card') or {}
        source, estimate = claim_gpu_estimate(policy)
        reason = (f"{card.get('free_mib')} MiB free < "
                  f"{-(-claim_required_free_bytes(policy) // 1024**2)} MiB required (reserve "
                  f"{policy.gpu_free_reserve_bytes} + estimate {estimate} bytes, {source}); "
                  'foreign activity is not a refusal reason')
        observation['last_refusal'] = reason
        _wait_notice(notice, f'{uuid}: {reason}')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        pause = min(policy.poll_millis / 1000, remaining)
        pause_unlocked(pause)
        observation['headroom_wait_seconds'] += pause


def _shared_admission_samples(probe, reader, output: Path, uuid: str, physical_index: int,
                              policy, baseline: dict, deadline: float, observation: dict,
                              pause_unlocked=time.sleep, window_reason=lambda: None):
    """Restart five fresh samples after headroom loss or an unstable card, within the same deadline.

    The lock is released for one poll interval before each restart.
    """
    notice = {}
    while True:
        try:
            return settled_cuda_samples(probe, output, 'model-proof-admission', uuid, physical_index,
                policy, baseline, deadline, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            timed_out = isinstance(error, _AdmissionInventoryTimeout)
            facts = cuda_sample_diagnostic(output, 'model-proof-admission', uuid, physical_index,
                policy, baseline, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY, allow_empty=timed_out)
            samples = facts.get('samples', [])
            def valid_identity(sample):
                card = sample.get('selected_card') or {}
                return (sample.get('predicates', {}).get('unique_uuid') is True
                        and sample.get('selected_index_matches_admission') is True
                        and _valid_shared_admission_card(dict(card, uuid=uuid), uuid, physical_index,
                                                         baseline['admitted_total_mib']))
            if timed_out and facts.get('status') == 'retained-existing-samples' and all(map(valid_identity, samples)):
                observation['inventory_window_timeouts'] = observation.get('inventory_window_timeouts', 0) + 1
                observation['last_inventory_window_timeout'] = {'error': str(error), 'discarded_samples': len(samples)}
                reason = f'incomplete admission window: {error}'
                if time.monotonic() >= deadline:
                    message = 'original CUDA claim deadline expired during admission inventory sampling'
                    raise CudaClaimRejected({'requested_uuid': uuid, 'admission_error': message,
                        'admission_samples': facts, 'admission_wait': dict(observation)}, message) from error
            elif str(error) == 'physical GPU occupied during model-proof-admission' and samples and valid_identity(samples[-1]):
                reason = window_reason() or 'free memory fell below reserve + estimate'
            else:
                raise
            observation['stable_sample_restarts'] += 1
            observation['last_refusal'] = reason
            _wait_notice(notice, f'{uuid}: restarting stable sampling: {reason}')
            remaining = deadline - time.monotonic()
            if remaining > 0:
                pause = min(policy.poll_millis / 1000, remaining)
                pause_unlocked(pause)
                key = 'inventory_timeout_wait_seconds' if timed_out else 'headroom_wait_seconds'
                observation[key] = observation.get(key, 0.) + pause
            devices, processes = _shared_admission_wait(reader, output, uuid, physical_index, policy, deadline,
                                                        observation, notice, pause_unlocked)
            current = next(device for device in devices if device['uuid'] == uuid)
            baseline.update(reserved_baseline_mib=current['used_mib'], process_identities=proof_cuda_contexts(
                uuid, processes, activity_profile=SHARED_CUDA_HEADROOM_ACTIVITY))


def effective_claim_deadline(started: float, policy, original_deadline_monotonic: float | None = None) -> float:
    deadline = started + policy.deadline_seconds
    if original_deadline_monotonic is not None:
        if type(original_deadline_monotonic) not in (int, float) or not math.isfinite(original_deadline_monotonic):
            raise ValueError('original comparison deadline must be finite')
        deadline = min(deadline, original_deadline_monotonic)
    return deadline


@contextmanager
def execution_claim(backend: str, policy, output: Path, env: dict[str, str], *, constraints: dict | None = None,
                    original_deadline_monotonic: float | None = None, require_owned_completion: bool = False,
                    abandon_wait=None):
    """Claim only during execution. CUDA reuses Veda's host-global UUID lock namespace.

    CUDA admission: the policy's reserve plus the claim's device estimate (claim_gpu_estimate)
    free on the exact UUID in five stable
    samples taken under the lock; display/compute co-tenants and utilization are diagnostic.
    `require_owned_completion` makes the release seal also require the supervisor's owned-Job
    completion record (the registry runner retains it); other callers' supervisors reap their
    own trees before the claim exits.

    A busy or short card is a queueing condition: the lock, headroom and quiet-window waits
    continue until the claim deadline, logging each refusal reason, and the deadline refusal
    records the last one.

    A hosted Mac/Linux worker is an isolated Actions execution environment. The provisioned
    Mac route claims the shared host-global cooperative POSIX lease in addition to its runner
    route. Labels/queues are not a physical mutex or a global inventory assertion.

    `abandon_wait(reason)` (CUDA, optional) is asked before every wait for this card (its lock is
    held by another job, or it lacks estimate plus reserve free). When it returns true the claim
    raises CudaCardTaken before any body runs, so a caller that can use another card switches
    instead of waiting.
    """
    if backend != 'candle-cuda':
        hosted = env.get('MODEL_PROOF_RUNNER_ENVIRONMENT') == 'github-hosted'
        # A provisioned Mac route (model_inference_proofs.NAX_ROUTES) is any self-hosted Mac its
        # labels select; the runner name is recorded, never compared to a machine.
        if not hosted and env.get('MODEL_PROOF_RUNNER_ENVIRONMENT') != 'self-hosted':
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
        claim_deadline = effective_claim_deadline(started, policy, original_deadline_monotonic)
        if started >= claim_deadline:
            raise ValueError('original comparison deadline expired before physical-device claim')
        lock_notice = {}
        def acquire():
            nonlocal claimed
            while not claimed:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    claimed = True
                except OSError:
                    if time.monotonic() >= claim_deadline:
                        raise ValueError('approved wait deadline expired before physical-device claim')
                    reason = f'{uuid}: UUID lock held by another CI job'
                    if abandon_wait is not None and abandon_wait(reason):
                        raise CudaCardTaken({'requested_uuid': uuid, 'abandoned_wait': reason},
                                            'claimed card is taken; switching to another card')
                    _wait_notice(lock_notice, reason)
                    time.sleep(policy.poll_millis / 1000)
        admission_wait = {}
        acquire()
        def pause_unlocked(seconds):
            """Release the UUID lock while waiting for this card, then take it back."""
            nonlocal claimed
            if abandon_wait is not None and abandon_wait(f"{uuid}: {admission_wait.get('last_refusal', 'waiting')}"):
                raise CudaCardTaken({'requested_uuid': uuid, 'abandoned_wait': admission_wait.get('last_refusal'),
                                     'admission_wait': dict(admission_wait)},
                                    'claimed card is taken; switching to another card')
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            claimed = False
            time.sleep(seconds)
            try:
                acquire()
            except CudaCardTaken:
                raise
            except ValueError as error:
                raise CudaClaimRejected({'requested_uuid': uuid, 'admission_error': str(error),
                                         'admission_wait': dict(admission_wait)}) from error
        from scripts.release import veda_cuda_guard
        activity_profile = bound_cuda_activity_profile(constraints)
        shared_headroom = activity_profile == SHARED_CUDA_HEADROOM_ACTIVITY
        inventory_reader = (lambda out, label, deadline: veda_cuda_guard.inventory(
            out, label, min(deadline, claim_deadline), selected_uuid=uuid)) if shared_headroom else veda_cuda_guard.inventory
        admission_wait.update({'primary_polls': 0, 'headroom_shortfalls': 0,
                               'stable_sample_restarts': 0, 'headroom_wait_seconds': 0.0})
        devices, processes = (_shared_admission_wait(inventory_reader, output, uuid,
            constraints.get('required_cuda_physical_index'), policy, claim_deadline, admission_wait,
            pause_unlocked=pause_unlocked)
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
                  'claim_gpu_estimate': claim_estimate_record(policy),
                  'activity_profile': activity_profile, 'claim_utilization_percent': matched[0]['utilization_percent']}
        if shared_headroom:
            record['activity_policy'] = ('non-isolated correctness; exact-card identity and estimate plus reserve free '
                                         'on a quiet card (memory.used spread/growth, utilization level/spread bounded); '
                                         'foreign contexts are diagnostic only')
            record['isolated_benchmark_claim'] = False
            record['lease_released'] = False
            record['claim_deadline_monotonic'] = claim_deadline
            record['admission_wait'] = admission_wait
            record['release_policy'] = 'verified-owned-Job-exit; exact selected primary metadata; headroom/activity diagnostic only'
        elif activity_profile == IRIS_CUDA_CORRECTNESS_ACTIVITY:
            record['activity_policy'] = ('non-isolated correctness, stable selected-card G/C+G contexts when utilization '
                                         'is nonzero; C also permitted at zero; below '
                                         f'{MAX_BASELINE_UTILIZATION_PERCENT}% utilization, estimate plus reserve free')
            record['isolated_benchmark_claim'] = False
        else:
            record['idle_policy'] = 'stable typed contexts, zero selected-card utilization, estimate plus reserve free'
        if constraints.get('child_cuda_visibility') is not None:
            record['child_visibility_binding'] = child_visibility_binding(
                constraints, env, uuid, claim_deadline - time.monotonic())
        # Reuse five fresh samples, with this proof's actual free reserve and child cap.
        context_baseline = {'uuid': uuid, 'reserved_baseline_mib': matched[0]['used_mib'],
                            'process_identities': proof_cuda_contexts(uuid, processes, activity_profile=activity_profile)}
        if shared_headroom:
            context_baseline['admitted_total_mib'] = matched[0]['total_mib']
        identity_key = lambda selected_uuid, rows: proof_cuda_contexts(
            selected_uuid, rows, activity_profile=activity_profile)
        settle_deadline = (claim_deadline if shared_headroom
                           else min(claim_deadline, time.monotonic() + 30))
        window = {}
        def read_admission_window(out, label, window_deadline):
            # Only this selected-card inventory query may restart a shared admission window.
            try:
                return inventory_reader(out, label, window_deadline)
            except (TimeoutError, subprocess.TimeoutExpired) as error:
                if not isinstance(error, subprocess.TimeoutExpired) and (type(error) is not TimeoutError
                        or str(error) != 'GPU inventory sampling deadline exhausted'):
                    raise
                raise _AdmissionInventoryTimeout(str(error)) from error

        def probe():
            # Headroom admission also needs a quiet card across the window (fresh per attempt).
            check = idle_check
            if shared_headroom:
                def valid_card(card, rows):
                    if not _valid_shared_admission_card(card, uuid, matched[0]['index'],
                                                        context_baseline['admitted_total_mib']):
                        raise RuntimeError('invalid selected primary identity/metadata during admission sampling')
                    return idle_check(card, rows)
                check, window['state'] = veda_cuda_guard.stable_window(valid_card)
            return veda_cuda_guard.sample_baseline(
                matched[0], output, 'model-proof-admission', dict(context_baseline), idle_check=check,
                identity_key=identity_key, permit_bounded_growth=True,
                require_stable_contexts=not shared_headroom,
                inventory_reader=read_admission_window if shared_headroom else inventory_reader)
        try:
            if shared_headroom:
                selected, baseline = _shared_admission_samples(probe, inventory_reader, output, uuid,
                    matched[0]['index'], policy, context_baseline, settle_deadline, admission_wait,
                    pause_unlocked=pause_unlocked,
                    window_reason=lambda: window.get('state', {}).get('reason'))
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
                if shared_headroom and require_owned_completion and not shared_owned_completion(record):
                    raise RuntimeError('supervisor-owned Windows Job completion is unverified; no release seal')
                release_check = (lambda device, rows: proof_cuda_shared_release(
                    device, uuid, matched[0]['index'], baseline['admitted_total_mib'])) if shared_headroom else idle_check
                settled_cuda_samples(
                    lambda: veda_cuda_guard.verify_idle(
                        selected, output, 'model-proof-release', baseline, idle_check=release_check,
                        identity_key=identity_key, permit_bounded_growth=True,
                        require_stable_contexts=not shared_headroom,
                        inventory_reader=(release_inventory_reader(veda_cuda_guard, uuid)
                                          if shared_headroom else inventory_reader)),
                    output, 'model-proof-release', uuid, matched[0]['index'], policy, baseline,
                    (time.monotonic() + RELEASE_SAMPLING_SECONDS if shared_headroom
                     else min(claim_deadline, time.monotonic() + 30)),
                    activity_profile=activity_profile)
                release_inconclusive = None
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                diagnostic = cuda_settle_refusal(
                    error, cuda_release_diagnostic(output, uuid, matched[0]['index'], policy,
                                                   baseline, activity_profile=activity_profile))
                if not (shared_headroom and release_check_inconclusive(error, diagnostic)):
                    raise CudaReleaseRejected('CUDA post-idle verification failed: ' + str(error), diagnostic) from error
                # Owned-Job exit was verified above; a slow driver query on a shared card is a diagnostic.
                release_inconclusive = {'status': 'inconclusive-driver-query-timing', 'error': str(error),
                                        'diagnostic': diagnostic}
            if shared_headroom:
                record['release_metadata_verification'] = release_inconclusive or {'status': 'verified'}
            if activity_profile in (IRIS_CUDA_CORRECTNESS_ACTIVITY, SHARED_CUDA_HEADROOM_ACTIVITY) and not release_inconclusive:
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
