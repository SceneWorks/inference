"""Source-bound, scalar-only protocol for the one Krea six-update recovery.

This neither executes downloaded code nor upgrades the failed original run. The
coordinator must separately qualify unchanged source/input closure before use.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import struct

FQN = 'trainer_e2e::krea_bounded_acceleration_profile'
CONFIGURATION = 'fixed_cuda_candidate_recovery'
CONTROL = 'KREA_PROFILE_CANDIDATE_RECOVERY'
GUARD = 'candidate-only-six-new-updates;prior-conservative-charge14;hard-cap20;no-reference-replay'
BASELINE_PATH = Path(__file__).with_name('krea-recovery-baseline.json')
ADAPTER_ROLE = 'actual-six-update-candidate-even-on-no-go'


def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding='utf-8'))


def applies(row: dict) -> bool:
    return row.get('package') == 'candle-gen-krea' and row.get('fqn') == FQN


def validate_binding(row: dict) -> bool:
    if not applies(row):
        return False
    recipe = row.get('selected_configuration', {})
    if recipe.get('name') == 'fixed_cuda_paired_comparison':
        from scripts.ci import krea_paired
        krea_paired.validate_binding(row)
        return False
    recovery = recipe.get('name') == CONFIGURATION
    if (row.get('configuration_kind') != 'source-controls'
            or row.get('configuration_controls') != [CONTROL]
            or recipe.get('environment') != {CONTROL: '1' if recovery else None}
            or (recovery and recipe.get('source_guard') != GUARD)):
        raise ValueError('Krea recovery/default binding must retain its exact six/18 source budget selector')
    return recovery


def _record(text: str, prefix: str) -> dict:
    lines = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError('Krea recovery missing/duplicate ' + prefix.strip())
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError('Krea recovery record must be an object')
    return value


def _close(actual: object, expected: float, tolerance: float = 1e-12) -> bool:
    return (type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=tolerance))


def verify_execution(row: dict, text: str) -> dict:
    if not applies(row):
        return {}
    recovery = validate_binding(row)
    if not recovery:
        if 'KREA_PROFILE_RECOVERY_' in text:
            raise ValueError('six-update recovery cannot masquerade as the18-update default')
        return {}
    if re.search(r'^KREA_PROFILE_RESULT ', text, re.M):
        raise ValueError('Krea recovery cannot claim a new18-update result')
    for phase in ('update-start', 'update-committed'):
        records = re.findall(r'^KREA_PROFILE_BOUNDARY phase=' + phase
                             + r' physical_update=(\d+) segment=(\d+) logical_step=(\d+)$', text, re.M)
        if (records != [(str(i), '2', str(i)) for i in range(1, 7)]
                or text.count('KREA_PROFILE_BOUNDARY phase=' + phase + ' ') != 6):
            raise ValueError('Krea recovery requires exactly six candidate ' + phase + ' boundaries')
    steps = re.findall(r'^KREA_PROFILE_STEP physical_update=(\d+)/(\d+) segment=(\d+) logical_step=(\d+) '
                       r'warmup=(true|false) seconds=([^ ]+) loss=([^ ]+) attention_candidate=(\d+) attention_reference=(\d+)$', text, re.M)
    if len(steps) != 6 or text.count('KREA_PROFILE_STEP ') != 6:
        raise ValueError('Krea recovery must have six actual step records')
    gradients = re.findall(r'^KREA_PROFILE_GRADIENTS physical_update=(\d+)/(\d+) finite=(\d+) total=(\d+)$', text, re.M)
    if (gradients != [(str(i), '6', '224', '224') for i in range(1, 7)]
            or text.count('KREA_PROFILE_GRADIENTS ') != 6):
        raise ValueError('Krea recovery requires all224 finite gradients on all six updates')
    if text.splitlines().count('KREA_PROFILE_RECOVERY_ADAPTER changed=true') != 1:
        raise ValueError('Krea recovery requires the actual changed candidate adapter')
    attribution = re.findall(r'^KREA_PROFILE_ATTRIBUTION segment=(\d+) synchronized_inclusive_spans=.+$', text, re.M)
    optimizer = re.findall(r'^KREA_PROFILE_OPTIMIZER segment=(\d+) synchronized_seconds=([^ ]+)$', text, re.M)
    preparation = re.findall(r'^KREA_PROFILE_PREPARATION seconds=([^ ]+)$', text, re.M)
    if (attribution != ['2'] or text.count('KREA_PROFILE_ATTRIBUTION ') != 1
            or len(optimizer) != 1 or text.count('KREA_PROFILE_OPTIMIZER ') != 1 or optimizer[0][0] != '2'
            or len(preparation) != 1 or text.count('KREA_PROFILE_PREPARATION ') != 1
            or any(not math.isfinite(float(value)) or float(value) < 0
                                          for value in [optimizer[0][1], *preparation])):
        raise ValueError('Krea recovery must retain first-warmup attribution/optimizer and separate preparation')
    report = _record(text, 'KREA_PROFILE_RECOVERY_RESULT ')
    execution = _record(text, 'KREA_PROFILE_RECOVERY_EXECUTION ')
    if (not re.fullmatch(r'[0-9a-f]{40}', str(execution.get('source', '')))
            or not re.fullmatch(r'[1-9][0-9]*', str(execution.get('run', '')))
            or not re.fullmatch(r'[1-9][0-9]*', str(execution.get('attempt', '')))
            or execution.get('run') == str(baseline()['run'])):
        raise ValueError('Krea recovery must preserve distinct actual candidate source/run identity')
    expected = {'schema_version': 1, 'new_optimizer_updates': 6, 'new_segment': 2,
                'steady_updates': 4, 'prior_conservative_update_charge': 14,
                'hard_update_cap': 20, 'remaining_updates_after_success': 0,
                'baseline': baseline(), 'saved_adapter_role': ADAPTER_ROLE,
                'shared_adapter_available': False}
    if any(report.get(key) != value or type(report.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError('Krea recovery changed the source-bound original evidence or hard6/20 budget')
    times, losses = report.get('candidate_steady_seconds'), report.get('candidate_losses')
    if (not isinstance(times, list) or len(times) != 4 or not isinstance(losses, list) or len(losses) != 6
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in [*times, *losses])
            or any(x <= 0 for x in times)):
        raise ValueError('Krea recovery requires six finite losses and four actual timed updates')
    dispatches = 0
    for i, step in enumerate(steps, 1):
        update, limit, segment, logical, warmup, seconds, loss, candidate, reference = step
        if (update, limit, segment, logical, warmup) != (str(i), '6', '2', str(i), str(i <= 2).lower()):
            raise ValueError('Krea recovery step budget/segment/warmup mismatch')
        if not math.isfinite(float(seconds)) or float(seconds) <= 0 or not math.isfinite(float(loss)):
            raise ValueError('Krea recovery nonfinite/unmeasured step')
        if struct.pack('!f', float(loss)) != struct.pack('!f', losses[i - 1]):
            raise ValueError('Krea recovery losses differ from actual step logs')
        if i > 2 and not _close(times[i - 3], float(seconds), 0.000000500001):
            raise ValueError('Krea recovery timing differs from actual rounded step log')
        if int(candidate) + int(reference) <= 0:
            raise ValueError('Krea recovery missing actual attention dispatch')
        dispatches += int(candidate)
    old = baseline()
    reference = sum(float(s['seconds']) for s in old['reference_segments'][0]['steps'][2:]) / 4
    shared = sum(float(s['seconds']) for s in old['reference_segments'][1]['steps'][2:]) / 4
    candidate = sum(times) / 4
    uncertainty = old['timing_mean_rounding_uncertainty_seconds']
    parity = all(abs(float(s['loss']) - loss) <= .01 * max(abs(float(s['loss'])), 1)
                 for s, loss in zip(old['reference_segments'][1]['steps'], losses))
    go = dispatches > 0 and parity and candidate < shared - uncertainty
    target_ambiguous = abs(candidate - .75 * reference) <= .75 * uncertainty
    decisions = {'candidate_dispatches': dispatches, 'loss_parity': parity, 'candidate_go': go,
                 'go_rounding_boundary_ambiguous': abs(candidate - shared) <= uncertainty,
                 'target_met': go and not target_ambiguous and candidate <= .75 * (reference - uncertainty),
                 'target_rounding_boundary_ambiguous': target_ambiguous}
    if any(report.get(key) != value or type(report.get(key)) is not type(value) for key, value in decisions.items()):
        raise ValueError('Krea recovery GO/target decision differs from source-bound measured scalars')
    for key, value in {'reference_logged_mean_seconds': reference, 'shared_logged_mean_seconds': shared,
                       'candidate_mean_seconds': candidate,
                       'candidate_reduction_vs_original_logged_mean': 1 - candidate / reference}.items():
        if not _close(report.get(key), value):
            raise ValueError('Krea recovery mean/reduction differs from measured scalar records')
    return {'krea_recovery': report, 'krea_candidate_execution': execution}


def verify_metrics(row: dict, metrics: dict, execution: dict, expected_source: str) -> None:
    if not applies(row) or not validate_binding(row):
        return
    report = execution['krea_recovery']
    identity = execution['krea_candidate_execution']
    config = metrics.get('configuration', {})
    # Rust stores the recipe as f32; serde_json emits that value promoted to
    # f64. Compare its exact wire value, without rounding untrusted metadata
    # (which would also admit other doubles) or weakening the fixed recipe.
    learning_rate = struct.unpack('!f', struct.pack('!f', .001))[0]
    expected_config = {'rank': 16, 'alpha': 16.0, 'learning_rate': learning_rate, 'resolution': 1024,
                       'train_dtype': 'bf16', 'gradient_accumulation': 1,
                       'gradient_checkpointing': False, 'seed': 7, 'sample_every': 500, 'save_every': 0}
    if (metrics.get('completed_optimizer_updates') != 6 or metrics.get('recovery') != report
            or metrics.get('candidate_execution') != identity or identity['source'] != expected_source
            or metrics.get('losses_by_physical_update') != report['candidate_losses']
            or metrics.get('saved_adapter_role') != ADAPTER_ROLE
            or metrics.get('graph_retaining_backward') is not False
            or metrics.get('selected_adapter_final_loss') != report['candidate_losses'][-1]
            or any(config.get(key) != value for key, value in expected_config.items())):
        raise ValueError('Krea recovery metrics differ from actual six-update candidate/source/recipe')
