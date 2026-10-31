"""Bounded manual conditional training comparisons; no model work on import.

The legacy control preserves explicitly listed correctness repairs. Fresh native
children avoid claiming that adapter-only snapshots restore private training state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time

PROTOCOL = 'conditional-v1'
STAGES = ('legacy', 'current', 'minus-e1', 'minus-e2', 'minus-e3',
          'minus-reference-owner', 'plus-streaming')
DEVICES = {
    'gpu0': ('GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0', 0),
    'gpu1': ('GPU-e4b79931-7be6-f216-460a-f5405cfafffe', 1),
}
RECIPES = {f'fixed_cuda_comparison_{name}': name for name in DEVICES}
SHA = re.compile(r'[0-9a-f]{64}')


def case_manifest(row: dict) -> dict:
    manifest = json.loads(Path(__file__).with_name('training-comparison-cases.json').read_text(encoding='utf-8'))
    found = [x for x in manifest['entries'] if all(x[k] == row.get(k) for k in ('package', 'target', 'fqn'))]
    if len(found) != 1 or manifest.get('schema_version') != 1 or manifest.get('protocol') != PROTOCOL:
        raise ValueError('comparison has no exact source-owned provider manifest')
    return found[0]

# Exact existing registered FQNs. N/A is source applicability, never a successful
# zero-dispatch stage. The reference owner is absent from all seventeen callers.
_CASES = {
    'lens_bounded_acceleration_acceptance': ('lens', 'ordinary', True, False, True),
    'z_image_bounded_acceleration_acceptance': ('z_image_turbo', 'ordinary', True, True, False),
    'qwen21_bounded_acceleration_acceptance': ('qwen_image_2_1', 'ordinary', True, True, True),
    'wan_t2v_14b_bounded_acceleration_acceptance': ('wan2_2_t2v_14b', 'wan-dual', True, True, True),
    'wan_ti2v_5b_bounded_acceleration_acceptance': ('wan2_2_ti2v_5b', 'ordinary', True, True, True),
    'ltx23_bounded_acceleration_acceptance': ('ltx_2_3', 'ordinary', True, False, True),
    'ltx25_bounded_acceleration_acceptance': ('ltx_2_5', 'ordinary', True, False, True),
    'sdxl_bounded_acceleration_acceptance': ('sdxl', 'ordinary', True, True, True),
    'kolors_bounded_acceleration_acceptance': ('kolors', 'ordinary', True, True, True),
    'sd35_medium_bounded_acceleration_acceptance': ('sd3_5_medium', 'ordinary', True, True, True),
    'anima_bounded_acceleration_acceptance': ('anima_base', 'ordinary', True, True, True),
    'moss_bounded_acceleration_acceptance': ('moss', 'ordinary', True, True, True),
    'mage_bounded_acceleration_acceptance': ('mage-adapter', 'ordinary', True, True, True),
    'flux3_bounded_acceleration_acceptance': ('flux3', 'ordinary', True, True, True),
    'iris_generation_bounded_acceleration_acceptance': ('iris-generation', 'ordinary', False, True, True),
    'iris_depth_bounded_acceleration_acceptance': ('iris-depth', 'ordinary', False, False, True),
    'iris_restoration_bounded_acceleration_acceptance': ('iris-restoration', 'iris-restoration', False, True, True),
}

# These are consuming branches, not preparation counters. Pre-existing caches,
# fused projections, and constant packed-input VJP repair are deliberately absent.
_WITNESSES = {
    'lens': (['lens:frozen-caption-storage-hit', 'lens:rope-storage-hit;E3-already-fused-native-QKV'], []),
    'z_image_turbo': (['zimage:frozen-caption-and-mask-storage-hit', 'zimage:exact-geometry-mask-storage-hit'], ['zimage:actual-frozen-QKV-pack']),
    'qwen_image_2_1': (['qwen21_frozen_initial_text', 'qwen21_frozen_reference_input', 'qwen21_joint_geometry'], ['qwen21_frozen_qkv_pack']),
    'wan2_2_t2v_14b': (['wan:expert-owned-caption-storage-hit'], ['wan:actual-frozen-projection-pack']),
    'wan2_2_ti2v_5b': (['wan:expert-owned-caption-storage-hit'], ['wan:actual-frozen-projection-pack']),
    'ltx_2_3': (['ltx23:rope-storage-hit'], ['ltx23:actual-dense-pack']),
    'ltx_2_5': (['ltx25:exact-grid-rope-storage-hit'], ['ltx25:actual-compatible-dense-pack']),
    'sdxl': (['sdxl-or-kolors:frozen-cross-base-KV-storage-hit'], ['sdxl:actual-frozen-projection-pack']),
    'kolors': (['sdxl-or-kolors:frozen-cross-base-KV-storage-hit'], ['sdxl:actual-frozen-projection-pack']),
    'sd3_5_medium': (['sd3:frozen-context-storage-hit'], ['sd3:actual-frozen-QKV-pack']),
    'anima_base': (['anima:geometry-storage-hit'], ['anima:frozen-projection-pack']),
    'moss': (['moss:geometry-storage-hit', 'moss:frozen-caption-storage-hit'], ['moss:frozen-projection-pack']),
    'mage-adapter': (['mage:geometry-storage-hit', 'mage:frozen-caption-base-hit-live-residual'], ['mage:frozen-projection-pack']),
    'flux3': (['flux3:geometry-storage-hit', 'flux3:frozen-caption-projection-consumed'], ['flux3:frozen-projection-pack']),
    'iris-generation': (['iris:geometry-storage-hit'], ['iris:frozen-GQA-projection-pack']),
    'iris-depth': (['iris:geometry-storage-hit'], []),
    'iris-restoration': (['iris:geometry-storage-hit'], ['iris:frozen-GQA-projection-pack']),
}


def spec(row: dict) -> dict:
    key = row.get('fqn', '').split('::')[-1]
    if key not in _CASES or not row.get('package', '').startswith('candle-gen-') or 'krea' in row['package']:
        raise ValueError('comparison is limited to the seventeen exact non-Krea training cases')
    model, cadence, e1, e3, e4 = _CASES[key]
    provider = case_manifest(row)
    if provider['model_id'] != model:
        raise ValueError('provider/core comparison model identity mismatch')
    updates, warmups = {'ordinary': (4, 1), 'wan-dual': (6, 2), 'iris-restoration': (8, 2)}[cadence]
    stages = [s for s in STAGES if not (
        s == 'minus-e1' and not e1 or s == 'minus-e3' and not e3
        or s == 'minus-reference-owner' or s == 'plus-streaming' and not e4)]
    seen, distinct = set(), []
    for stage in stages:
        key = (e1 and stage not in ('legacy', 'minus-e1'),
               stage not in ('legacy', 'minus-e2'),
               e3 and stage not in ('legacy', 'minus-e3'),
               e4 and stage == 'plus-streaming')
        if key not in seen:
            seen.add(key)
            distinct.append(stage)
    stages = distinct
    if provider['stages'] != stages or provider['updates_per_stage'] != updates or provider['warmups_per_stage'] != warmups:
        raise ValueError('provider/core stage applicability or physical cadence mismatch')
    # Stable randomized serial order avoids an optimization ladder. It does not
    # erase first-stage/interference bias; three timed samples remain exploratory.
    order = list(stages)
    seed = int.from_bytes(hashlib.sha256((PROTOCOL + ':' + model).encode()).digest()[:8], 'little')
    random.Random(seed).shuffle(order)
    return {'protocol': PROTOCOL, 'model': model, 'cadence': cadence,
            'stages': stages, 'execution_order': order, 'updates_per_stage': updates,
            'warmups_per_stage': warmups, 'max_physical_updates': updates * len(stages),
            'applicability': {'e1_combined_arithmetic_norm': e1, 'e2': True, 'e3': e3,
                              'reference_owner': False, 'streaming_candidate': e4},
            'constant_repairs': provider['constant_repairs'],
            'e1_norm_scope': provider.get('E1_norm_scope', 'Combined arithmetic/norm group where applicable; pre-existing mechanisms remain common'),
            'workload': provider['workload'],
            'case_deadline_seconds': provider.get('case_deadline_seconds', 3600),
            'immutable_frozen_bootstrap': provider.get('immutable_frozen_bootstrap'),
            'source_manifest_sha256': file_digest(Path(__file__).with_name('training-comparison-cases.json')),
            'baseline_scope': 'legacy optimization control with constant correctness repairs; not a historical whole binary',
            'sample_scope': 'zero new quality samples; existing quality evidence remains separate',
            'comparison_scope': 'conditional leave-one-group-out effects; not additive independent contributions; shared-GPU serial timing',
            'noise_scope': 'one counted warmup per owner; three ordinary/restoration timed updates or cycles, two per Wan expert; small sample and order/interference uncertainty'}


def is_recipe(name: object) -> bool:
    return name in RECIPES if isinstance(name, str) else False


def recipe_binding(row: dict, name: str, recipe: dict) -> tuple[str, int]:
    plan = spec(row)
    binding = RECIPES.get(name)
    if binding is None:
        raise ValueError('unknown reviewed comparison physical recipe')
    uuid, index = DEVICES[binding]
    if (recipe.get('required_cuda_device_uuid') != uuid
            or recipe.get('required_cuda_physical_index') != index
            or recipe.get('environment') != {'MODEL_PROOF_COMPARISON_BINDING': binding,
                                             'MODEL_PROOF_COMPARISON_STAGE': None}
            or recipe.get('comparison_protocol') != PROTOCOL):
        raise ValueError('comparison recipe differs from exact source UUID/index/control binding')
    if not plan['stages']:
        raise ValueError('empty source comparison')
    return uuid, index


def validate_request(rows: tuple[dict, ...], value: dict, backend: str) -> dict:
    if backend != 'candle-cuda' or len(rows) != 1 or not is_recipe(value.get('configuration')):
        raise ValueError('comparison requires one exact CUDA training case and reviewed comparison recipe')
    row = rows[0]
    plan = spec(row)
    requested = value.get('training_comparison')
    expected = {k: plan[k] for k in ('protocol', 'stages', 'max_physical_updates')}
    if requested != expected or type(requested.get('max_physical_updates')) is not int:
        raise ValueError('comparison request must affirm the exact source stage list and physical cap')
    uuid, index = recipe_binding(row, value['configuration'], row['selected_configuration'])
    if value['policy'].get('cudaDeviceUuid') != uuid or row.get('required_cuda_device_uuid') != uuid or row.get('required_cuda_physical_index') != index:
        raise ValueError('comparison policy/case physical UUID/index mismatch')
    if (value['policy'].get('childGpuCapBytes') != 91 * 1024**3
            or value['policy'].get('gpuFreeReserveBytes') != 2 * 1024**3
            or value['policy'].get('deadlineSeconds') != plan['case_deadline_seconds']):
        raise ValueError('comparison requires unchanged91GiB+2GiB and its source-declared overall deadline')
    if value.get('additional_cuda_policies') or value.get('cuda_devices'):
        raise ValueError('comparison has exactly one selected physical device')
    return {**plan, 'device_binding': RECIPES[value['configuration']], 'cuda_uuid': uuid, 'physical_index': index}


def stage_row(row: dict, stage: str) -> dict:
    plan = spec(row)
    if stage not in plan['stages'] or not is_recipe(row.get('selected_configuration', {}).get('name')):
        raise ValueError('stage is not an applicable source-owned comparison')
    result = copy.deepcopy(row)
    result['training_comparison_stage'] = stage
    result['training_comparison_plan'] = plan
    if plan['model'] == 'iris-restoration':
        result['optional_environment'] = sorted(set(result.get('optional_environment', [])) | {'MODEL_PROOF_IRIS_FROZEN_CACHE_DIR'})
    provider = case_manifest(row)
    result['output_artifacts'] = [provider['metrics']]
    result.pop('generated_adapter_artifacts', None)
    contract = result['input_contract']
    contract['requirements'] = copy.deepcopy(provider['surviving_input_requirements'])
    if not contract['requirements']:
        raise ValueError('comparison cannot discard all actual training input readers')
    result['output_contract'] = 'Timing-only source supplement: exact bounded fresh-child stage; actual training readers, starts/commits/finite gradients/state fingerprints/branch witnesses and selected-card global peak; no new sample or export lifecycle claim'
    return result


def apply_stage(row: dict, env: dict) -> None:
    stage = row.get('training_comparison_stage')
    if stage is None:
        return
    plan = spec(row)
    if stage not in plan['stages'] or row.get('training_comparison_plan') != plan:
        raise ValueError('altered comparison stage authority')
    uuid, index = recipe_binding(row, row['selected_configuration']['name'], row['selected_configuration'])
    if row.get('required_cuda_device_uuid') != uuid or row.get('required_cuda_physical_index') != index:
        raise ValueError('stage physical mapping differs')
    env['MODEL_PROOF_COMPARISON_STAGE'] = stage
    env['SCENEWORKS_GPU_ID'] = str(index)
    env.pop('MODEL_PROOF_IRIS_FROZEN_CACHE_DIR', None)
    # Existing native asset fixtures may write large recovery exports. Keep each
    # fresh child in its own owned external runtime subdirectory.
    if env.get('BESPOKE_RUNTIME_DIR'):
        parent = Path(env['BESPOKE_RUNTIME_DIR'])
        marker = parent / '.sc-26850-owned-runtime'
        from scripts.ci import model_inference_proofs as proof
        if proof.linked(marker) or not marker.is_file() or marker.stat().st_nlink != 1 or marker.stat().st_size > 4096:
            raise ValueError('comparison runtime requires its actual parent owned marker')
        marker_bytes = marker.read_bytes()
        if marker_bytes.decode('utf-8').strip() != proof.identity(row):
            raise ValueError('comparison runtime parent marker belongs to another exact case')
        if plan['model'] == 'iris-restoration':
            source = env.get('GITHUB_SHA', '')
            run_id, attempt = env.get('GITHUB_RUN_ID', ''), env.get('GITHUB_RUN_ATTEMPT', '')
            if not re.fullmatch(r'[0-9a-f]{40}', source) or not re.fullmatch(r'[0-9]+', run_id) or not re.fullmatch(r'[0-9]+', attempt):
                raise ValueError('Iris frozen cache requires source/run-bound Actions provenance')
            shared = parent / 'comparison-iris-frozen'
            proof.unlinked_path(shared, parent)
            shared.mkdir(exist_ok=True)
            owner = {'schema_version':1,'protocol':PROTOCOL,'binding':proof.identity(row),
                     'source_revision':source,'github_run_id':run_id,'github_run_attempt':attempt,
                     'parent_marker_sha256':hashlib.sha256(marker_bytes).hexdigest(),
                     'cache_scope':'Iris restoration immutable frozen checkpoint decode; constant all stages, not E2'}
            for path, data in ((shared / marker.name, marker_bytes),
                               (shared / 'comparison-cache-owner.json', (json.dumps(owner,sort_keys=True)+'\n').encode())):
                proof.unlinked_path(path, parent)
                if path.exists():
                    if not path.is_file() or path.stat().st_nlink != 1 or path.read_bytes() != data:
                        raise ValueError('Iris frozen cache ownership/source/run binding mismatch')
                else:
                    path.write_bytes(data)
            env['MODEL_PROOF_IRIS_FROZEN_CACHE_DIR'] = str(shared)
        runtime = parent / ('comparison-' + stage)
        runtime.mkdir(exist_ok=False)
        (runtime / marker.name).write_bytes(marker_bytes)
        env['BESPOKE_RUNTIME_DIR'] = str(runtime)


def _finite(value: object, *, positive=False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else True)


def verify_metrics(row: dict, metrics: dict, text: str) -> dict:
    plan = spec(row)
    stage = row.get('training_comparison_stage')
    if stage not in plan['stages']:
        raise ValueError('metrics lack exact requested source stage')
    n, warm = plan['updates_per_stage'], plan['warmups_per_stage']
    comparison = metrics.get('comparison', {})
    if (comparison.get('schema_version') != 1 or comparison.get('model') != plan['model']
            or comparison.get('stage') != stage or comparison.get('cadence') != plan['cadence']
            or comparison.get('state_reset') != 'fresh-native-child-process'
            or comparison.get('cuda_visible_devices') != row['required_cuda_device_uuid']
            or comparison.get('physical_index') != row['required_cuda_physical_index']
            or metrics.get('optimizer_updates') != n
            or comparison.get('physical_starts') != n or comparison.get('physical_commits') != n):
        raise ValueError('stage identity/reset/physical counts mismatch')
    if re.findall(r'^TRAINING_COMPARISON_STAGE model=([^ ]+) stage=([^ ]+) cadence=([^ ]+) updates=(\d+)\s*$', text, re.M) != [(plan['model'], stage, plan['cadence'], str(n))]:
        raise ValueError('missing/duplicate/swapped actual stage body')
    if sum(line.startswith('TRAINING_COMPARISON_STAGE') for line in text.splitlines()) != 1:
        raise ValueError('unknown/truncated additional comparison stage header')
    starts = re.findall(r'^COHORT_UPDATE_START attempted_update=(\d+)\s*$', text, re.M)
    commits = re.findall(r'^COHORT_UPDATE_COMMITTED optimizer_updates=(\d+)\s*$', text, re.M)
    sequence = [str(i) for i in range(1, n + 1)]
    if (starts != sequence or commits != sequence
            or sum(line.startswith('COHORT_UPDATE_START') for line in text.splitlines()) != n
            or sum(line.startswith('COHORT_UPDATE_COMMITTED') for line in text.splitlines()) != n):
        raise ValueError('missing/extra/truncated physical update boundaries')
    initial = comparison.get('initialization')
    tensors = comparison.get('initial_trainable_fingerprints')
    if (not isinstance(initial, list) or not initial or any(not isinstance(x, dict) or not x for x in initial)
            or not isinstance(tensors, list) or not tensors or any(not isinstance(x, str) or not SHA.fullmatch(x) for x in tensors)):
        raise ValueError('missing actual initial state/configuration fingerprints')
    opt = comparison.get('optimizer_initialization')
    if not isinstance(opt, list) or any(not isinstance(x, dict) or not isinstance(x.get('sha256'), str) or not SHA.fullmatch(x['sha256']) for x in opt):
        raise ValueError('invalid actual optimizer initialization receipts')
    if not opt and not any(x.get('kind') == 'actual-named-initial-tensors' and 'optim' in x.get('owner', '').lower() for x in initial):
        raise ValueError('neither shared nor private actual initial optimizer state authenticated')
    steps = metrics.get('steps')
    attempts = comparison.get('actual_shared_optimizer_attempts')
    if type(attempts) is not int or not 0 <= attempts <= n or (opt and attempts != n):
        raise ValueError('shared actual optimizer attempts differ from bounded observed updates')
    if not isinstance(steps, list) or len(steps) != n or not _finite(metrics.get('preparation_including_load_cache_seconds')) or metrics['preparation_including_load_cache_seconds'] < 0:
        raise ValueError('truncated stage/timing preparation data')
    owners = []
    for i, step in enumerate(steps, 1):
        if (step.get('update') != i or step.get('warmup') is not (i <= warm)
                or not _finite(step.get('seconds'), positive=True) or not _finite(step.get('loss'))
                or type(step.get('finite_gradients')) is not int or step['finite_gradients'] <= 0
                or type(step.get('nonzero_gradients')) is not int or not 0 < step['nonzero_gradients'] <= step['finite_gradients']
                or step.get('adapter_changed') is not True or not isinstance(step.get('owner'), str) or not step['owner']):
            raise ValueError('invalid actual step/gradient/timing/owner witness')
        owners.append(step['owner'])
        candidate, reference = step.get('attention_candidate'), step.get('attention_reference')
        critic_only = plan['cadence'] == 'iris-restoration' and i % 2 == 0
        if any(type(x) is not int or x < 0 for x in (candidate, reference)) or (stage != 'plus-streaming' and candidate != 0) or (stage == 'plus-streaming' and not critic_only and candidate <= 0):
            raise ValueError('requested attention mechanism was not actually selected')
        mechanisms = step.get('mechanism_consumption')
        if not isinstance(mechanisms, dict) or any(type(v) is not int or v <= 0 for v in mechanisms.values()):
            raise ValueError('invalid actual mechanism counters')
        if not plan['applicability']['reference_owner'] and any(k.startswith('comparison:reference-owner:') for k in mechanisms):
            raise ValueError('source N/A Reference-owner path actually executed')
        if plan['applicability']['e1_combined_arithmetic_norm']:
            expected = 'legacy' if stage in ('legacy', 'minus-e1') else 'current'
            opposite = 'current' if expected == 'legacy' else 'legacy'
            if not any(k.startswith('comparison:e1-') and k.endswith(':' + expected) for k in mechanisms) or any(k.startswith('comparison:e1-') and k.endswith(':' + opposite) for k in mechanisms):
                raise ValueError('E1 math/owner paths are missing or contradictory')
        elif any(k.startswith('comparison:e1-') for k in mechanisms):
            raise ValueError('source N/A E1 shared path actually executed')
    unique = list(dict.fromkeys(owners))
    provider = case_manifest(row)
    declared_owners = provider.get('owners') or provider.get('owner')
    declared_owners = declared_owners if isinstance(declared_owners, list) else [declared_owners]
    if unique != declared_owners:
        raise ValueError('actual optimizer owner identity differs from source provider manifest')
    if plan['cadence'] == 'ordinary' and len(unique) != 1:
        raise ValueError('ordinary stage changed optimizer owner')
    if plan['cadence'] != 'ordinary':
        expected_each = n // 2
        if len(unique) != 2 or owners[:2] != unique or any(owners.count(o) != expected_each for o in unique) or any(owners[i] != unique[i % 2] for i in range(n)):
            raise ValueError('dual-owner warm/timed update cycles incomplete')
    for group, witnesses in zip(('e2', 'e3'), _WITNESSES[plan['model']], strict=True):
        on = stage not in ('legacy', 'minus-' + group)
        total = sum(sum(step['mechanism_consumption'].get(w, 0) for w in witnesses) for step in steps)
        timed_total = sum(sum(step['mechanism_consumption'].get(w, 0) for w in witnesses) for step in steps if not step['warmup'])
        if not plan['applicability'][group]:
            if total:
                raise ValueError('source N/A mechanism actually executed: ' + group)
            continue
        if (on and timed_total <= 0) or (not on and total != 0):
            raise ValueError('actual consuming branches contradict requested ' + group + ' control')
    timed = [x['seconds'] for x in steps if not x['warmup']]
    owner_timings = {owner: {'warmup_seconds': [x['seconds'] for x in steps if x['owner'] == owner and x['warmup']],
                             'timed_seconds': [x['seconds'] for x in steps if x['owner'] == owner and not x['warmup']]}
                     for owner in unique}
    for timings in owner_timings.values():
        timings['mean_seconds'] = statistics.mean(timings['timed_seconds'])
        timings['sample_stdev_seconds'] = statistics.stdev(timings['timed_seconds'])
    return {'stage': stage, 'physical_updates': n, 'owners': owners,
            'timed_seconds': timed, 'mean_seconds': statistics.mean(timed),
            'median_seconds': statistics.median(timed), 'min_seconds': min(timed),
            'max_seconds': max(timed), 'sample_stdev_seconds': statistics.stdev(timed),
            'sample_stdev_scope': 'Timed physical updates; multi-owner stages mix different operations. Per-owner samples are separate; no statistical confidence claim',
            'per_owner_timings': owner_timings,
            'preparation_seconds': metrics['preparation_including_load_cache_seconds'],
            'warmup_seconds': [x['seconds'] for x in steps if x['warmup']],
            'observed_setup_plus_counted_updates_seconds': metrics['preparation_including_load_cache_seconds'] + sum(x['seconds'] for x in steps),
            'setup_timing_scope': 'Observed model load/cache/preparation includes initial-state proof diagnostics; update spans exclude post-span diagnostics. Excludes compilation/admission/export; not production end-to-end latency',
            'losses': [x['loss'] for x in steps], 'initialization': initial,
            'initial_trainable_fingerprints': tensors, 'optimizer_initialization': opt,
            'finite_gradient_counts': [x['finite_gradients'] for x in steps]}


def verify_matching_states(reference: dict, candidate: dict) -> None:
    for key in ('initialization', 'initial_trainable_fingerprints', 'optimizer_initialization', 'owners', 'finite_gradient_counts'):
        if reference[key] != candidate[key]:
            raise ValueError('cross-stage initial settings/state/owner/gradient surface mismatch: ' + key)


def aggregate(plan: dict, stages: list[dict]) -> dict:
    if [x.get('stage') for x in stages] != plan['execution_order']:
        raise ValueError('aggregate exact stage order/set mismatch')
    by_stage = {x['stage']: x for x in stages}
    current = by_stage['current']
    for item in stages:
        verify_matching_states(current, item)
    def compare(base: str, candidate: str, label: str) -> dict:
        a, b = by_stage[base], by_stage[candidate]
        parity = all(abs(x-y) <= 1e-3 * max(1., abs(x), abs(y)) for x, y in zip(a['losses'], b['losses'], strict=True))
        delta = a['mean_seconds'] - b['mean_seconds']
        reduction = delta / a['mean_seconds']
        short_delta = a['observed_setup_plus_counted_updates_seconds'] - b['observed_setup_plus_counted_updates_seconds']
        return {'strategy': label, 'reference_stage': base, 'candidate_stage': candidate,
                'seconds_saved_per_physical_update': delta,
                'candidate_over_reference': b['mean_seconds'] / a['mean_seconds'],
                'fractional_time_reduction': reduction,
                'observed_setup_plus_counted_updates_seconds_saved': short_delta,
                'observed_short_run_outcome': 'win' if short_delta > 0 else 'loss' if short_delta < 0 else 'tie',
                'short_run_limit': 'Includes proof initialization diagnostics, excludes compile/admission/export; report separately from steady update timing',
                'common_bootstrap_limit': 'First stage may build the common immutable frozen store; later stages reuse its exact bytes. Setup differences are not attributable to E2 or another strategy' if plan['immutable_frozen_bootstrap'] else None,
                'measured_outcome': 'win' if delta > 0 else 'loss' if delta < 0 else 'tie',
                'scalar_loss_trajectory_parity': parity,
                'numerical_scope': 'matched scalar losses at relative-to-max(1,loss) tolerance0.001; finite reached gradients, not full gradient parity/convergence',
                'candidate_go': delta > 0 and parity}
    comparisons = [compare('legacy', 'current', 'total current versus legacy optimization control')]
    for stage, label in (('minus-e1', 'E1 combined arithmetic and norms'), ('minus-e2', 'E2 frozen conditioning and geometry'), ('minus-e3', 'E3 frozen projection packing'), ('minus-reference-owner', 'bounded Reference attention owner')):
        if stage in by_stage:
            comparisons.append(compare(stage, 'current', label + ' conditional benefit'))
    if 'plus-streaming' in by_stage:
        comparisons.append(compare('current', 'plus-streaming', 'E4 streaming candidate conditional benefit'))
    return {'protocol': PROTOCOL, 'plan': plan, 'actual_physical_updates': sum(x['physical_updates'] for x in stages),
            'comparisons': comparisons, 'stages': stages,
            'acceptance_rule': 'any measured positive speedup is a win; slowdown is a loss; numerical suitability separate; no25percent threshold',
            'production_promotion': False, 'claim_limit': plan['comparison_scope'] + '; ' + plan['noise_scope']}


def file_digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hash_.update(block)
    return hash_.hexdigest()


def validate_shared_source_controls(sources: dict[str, str]) -> None:
    """Metadata policy for the real legacy branches; accepts in-memory mutations.

    Counters alone cannot establish an ablation. Keep the actual historic
    broadcast/scaling/composable-return branches bound to the reviewed selectors.
    Runtime receipts independently require their actual consuming witnesses.
    """
    def body(file: str, name: str) -> str:
        source = sources[file]
        start = re.search(r'\bfn\s+' + re.escape(name) + r'\b', source)
        if start is None:
            raise ValueError('missing shared comparison source function: ' + name)
        opening = source.index('{', start.end())
        depth = 1
        end = opening + 1
        while depth and end < len(source):
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        if depth:
            raise ValueError('truncated shared comparison source function')
        return re.sub(r'\s+', '', source[opening+1:end-1])
    required = (
        ('projection.rs', 'apply_factor', 'ifsuper::profile::reference_math()||'),
        ('projection.rs', 'apply_factor', 'returnx.broadcast_matmul(w);'),
        ('projection.rs', 'scaled_lora', 'ifsuper::profile::reference_math(){returnapply_factor(&rank,up)?*scale;}'),
        ('projection.rs', 'prepare_lora', 'if!super::comparison::projection_enabled(){returnOk(None);}'),
        ('projection.rs', 'prepare_adapt', 'if!super::comparison::projection_enabled(){returnOk(None);}'),
        ('norm.rs', 'rms_normalize_f32_impl', 'ifsuper::comparison::legacy_e1()||'),
        ('norm.rs', 'rms_normalize_f32_impl', 'returnnormalize_composable(x,eps,reference);'),
        ('norm.rs', 'rms_norm_diff_impl', 'ifsuper::comparison::legacy_e1()||'),
        ('norm.rs', 'rms_norm_diff_impl', 'returnnormalize_composable(x,eps,reference)?.broadcast_mul(weight)?.to_dtype(x.dtype());'),
        ('norm.rs', 'normalize_composable', 'ifreference{xf.broadcast_div(&rms)}'),
        ('reference_training_attention.rs', 'try_forward', 'if!crate::train::comparison::reference_owner_enabled(){crate::train::profile::cohort_reuse("comparison:reference-owner:composable");returnOk(None);}'),
        ('profile.rs', 'reference_math', '||super::comparison::legacy_e1()'),
        ('profile.rs', 'cohort_claim_probe', 'ifsuper::comparison::active_stage().is_some(){returnfalse;}'),
        ('optim.rs', 'step', 'super::profile::comparison_optimizer_step()?;'),
    )
    for file, name, fragment in required:
        if fragment not in body(file, name):
            raise ValueError('actual shared legacy/control branch missing: ' + file + ':' + name)


def update_accounting(row: dict, case: dict, text: str) -> dict:
    from scripts.ci import model_inference_proofs as proof
    starts = len(re.findall(r'^COHORT_UPDATE_START attempted_update=', text, re.M))
    commits = len(re.findall(r'^COHORT_UPDATE_COMMITTED optimizer_updates=', text, re.M))
    commands = case.get('commands', [])
    no_body = not re.search(r'^(?:TRAINING_COMPARISON_STAGE |COHORT_UPDATE_|MODEL_INFERENCE_BODY_COMPLETED |MODEL_INFERENCE_DEVICE )', text, re.M)
    discovery_only = (len(commands) == 1 and commands[0].get('argv') == proof.command(row, True)
                      and commands[0].get('supervision', {}).get('returncode') == 0)
    refused = (case.get('accepted') is False
               and case.get('cuda_claim_diagnostic', {}).get('requested_uuid') == row.get('required_cuda_device_uuid')
               and case.get('physical_execution') == {'mechanism': 'metadata-only discovery; no model execution'}
               and case.get('error') == 'claimed CUDA UUID is absent or occupied; no model execution'
               and discovery_only and no_body)
    completion = case.get('supervision_failure', {}).get('owned_completion', {})
    supervisor_no_creation = (completion.get('scope') == 'supervisor-no-child-created'
                              and completion.get('creation_attempted') is False
                              and discovery_only and no_body)
    zero = bool((refused or supervisor_no_creation) and starts == commits == 0)
    return {'observed_starts': starts, 'observed_commits': commits,
            'conservative_charge': 0 if zero else spec(row)['updates_per_stage'],
            'zero_source_proven': zero, 'automatic_retry': False,
            'basis': 'affirmative authenticated pre-creation refusal' if zero else 'started/unknown/partial reserves full stage; no automatic retry'}


def verify_stage_deadline(case: dict, child_request: dict, original_deadline: float) -> None:
    from scripts.ci.model_proof_resources import metadata_policy_payload
    if not _finite(original_deadline) or case.get('comparison_original_deadline_monotonic') != original_deadline:
        raise ValueError('stage lost/refreshed the original absolute comparison deadline')
    payload = case.get('comparison_discovery_policy')
    if payload is None:
        if case.get('accepted') is True or case.get('commands'):
            raise ValueError('actual discovery omitted original remaining-deadline policy')
        return  # An incomplete pre-discovery failure cannot become accepted.
    start = case.get('comparison_discovery_started_monotonic')
    if not _finite(start) or not 0 < original_deadline-start <= child_request['policy']['deadlineSeconds']:
        raise ValueError('discovery did not consume the same original remaining budget')
    expected = metadata_policy_payload(child_request['policy'])
    expected['deadlineSeconds'] = original_deadline-start
    if payload != expected:
        raise ValueError('discovery changed resource settings or refreshed deadline')
    commands = case.get('commands', [])
    if commands:
        digest = hashlib.sha256((json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
        if commands[0].get('supervision', {}).get('admission', {}).get('policySha256') != digest:
            raise ValueError('actual metadata supervisor used another deadline policy')
    if case.get('accepted') is True and case.get('physical_execution', {}).get('claim_deadline_monotonic') != original_deadline:
        raise ValueError('body admission refreshed the post-discovery deadline')


def _inventory(root: Path) -> dict:
    from scripts.ci import model_inference_proofs as proof
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if proof.linked(Path(directory) / name):
                raise ValueError('linked comparison artifact directory')
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative in ('receipt.json', 'summary.md'):
                continue
            if proof.linked(path) or not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError('comparison requires unique regular artifact files')
            result[relative] = {'bytes': path.stat().st_size, 'sha256': file_digest(path)}
    return result


def run(rows: tuple[dict, ...], backend: str, value: dict, expected: str, output: Path) -> int:
    from scripts.ci import model_inference_proofs as proof
    # This clock starts before discovery/compilation/admission of the first child.
    # All readiness waits and subsequent children consume this SAME budget.
    started = time.monotonic()
    plan = validate_request(rows, value, backend)
    proof.repository_identity(expected)
    root = proof.owned_output(output, Path(os.environ['RUNNER_TEMP']))
    (root / 'stages').mkdir()
    work = root.parent / (root.name + '-comparison-work')
    proof.unlinked_path(work, Path(os.environ['RUNNER_TEMP']))
    work.mkdir(exist_ok=False)
    receipt = {'schema_version': 1, 'protocol': PROTOCOL, 'source_revision': expected,
               'request': value, 'plan': plan, 'accepted': False, 'stage_records': [],
               'automatic_retry': False, 'production_promotion': False,
               'acceptance_scope': 'controlled-training-timing-supplement',
               'checkpoint_byte_integrity_attested': False, 'verified_named_revision': False,
               'terminal_model_evidence': False}
    receipt['comparison_started_monotonic'] = started
    receipt['comparison_deadline_monotonic'] = started + value['policy']['deadlineSeconds']
    results = []
    inherited_target = os.environ.get('CARGO_TARGET_DIR')
    # One fresh owned compile namespace for the whole model comparison. Later
    # metadata discovery reuses that actual binary; no copied fingerprints and
    # no cross-listener rebuild can replace it between bounded stages.
    os.environ['CARGO_TARGET_DIR'] = str(work / 'cargo-target')
    try:
        for stage in plan['execution_order']:
            elapsed = time.monotonic() - started
            remaining = value['policy']['deadlineSeconds'] - elapsed
            if remaining <= max(value['policy']['pollMillis'], value['policy']['termGraceMillis']) / 1000:
                raise ValueError('original comparison deadline exhausted before next stage; no new child')
            child_value = copy.deepcopy(value)
            child_value.pop('training_comparison')
            child_value['policy']['deadlineSeconds'] = remaining
            selected = stage_row(rows[0], stage)
            destination = root / 'stages' / stage
            record = {'stage': stage, 'artifact': 'stages/' + stage, 'request': child_value,
                      'elapsed_before_stage_seconds': elapsed,
                      'remaining_original_deadline_seconds': remaining}
            receipt['stage_records'].append(record)
            code = proof.run((selected,), backend, child_value, expected, destination,
                             comparison_work=work / stage,
                             comparison_deadline_monotonic=receipt['comparison_deadline_monotonic'])
            record['returncode'] = code
            child_receipt = json.loads((destination / 'receipt.json').read_text(encoding='utf-8'))
            case = child_receipt.get('cases', [{}])[0]
            text = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in
                             (destination / '00000/body.stdout.log', destination / '00000/body.stderr.log') if p.is_file())
            record['update_accounting'] = update_accounting(selected, case, text)
            proof.verify_artifacts(destination, Path(os.environ['RUNNER_TEMP']), child_value, expected, (selected,), backend)
            if 'comparison_executable_sha256' in case:
                record['executable_sha256'] = case['comparison_executable_sha256']
            if code != 0 or child_receipt.get('accepted') is not True:
                raise ValueError('stage failed; comparison stops without retry: ' + stage)
            if results:
                verify_matching_states(results[0], case['training_comparison'])
            results.append(case['training_comparison'])
            if len({x['executable_sha256'] for x in receipt['stage_records']}) != 1:
                raise ValueError('stage compiled native executable identity changed')
        receipt['measurement'] = aggregate(plan, results)
        receipt['accepted'] = True
    except (ValueError, OSError, KeyError, IndexError) as error:
        receipt['error'] = str(error)
    finally:
        if inherited_target is None:
            os.environ.pop('CARGO_TARGET_DIR', None)
        else:
            os.environ['CARGO_TARGET_DIR'] = inherited_target
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['conservative_updates_charged'] = sum(r.get('update_accounting', {}).get('conservative_charge', plan['updates_per_stage']) for r in receipt['stage_records'])
        receipt['artifacts'] = _inventory(root)
        proof.write_json(root / 'receipt.json', receipt, owned=root)
        summary = f"Training comparison `{plan['model']}`; source `{expected}`; accepted={receipt['accepted']}; conservative updates charged {receipt['conservative_updates_charged']}/{plan['max_physical_updates']}. No automatic retry.\n"
        if receipt.get('measurement'):
            for comparison in receipt['measurement']['comparisons']:
                summary += f"\n- {comparison['strategy']}: {comparison['measured_outcome']}, reduction={comparison['fractional_time_reduction']:.6f}, scalar parity={comparison['scalar_loss_trajectory_parity']}."
        summary += '\n\n' + plan['comparison_scope'] + '; ' + plan['noise_scope'] + '.\n'
        (root / 'summary.md').write_text(summary, encoding='utf-8')
    return 0 if receipt['accepted'] else 1


def verify_artifacts(output: Path, temporary: Path, value: dict, expected: str,
                     rows: tuple[dict, ...], backend: str) -> dict:
    from scripts.ci import model_inference_proofs as proof
    plan = validate_request(rows, value, backend)
    proof.unlinked_path(output, temporary)
    for name in ('receipt.json', 'summary.md'):
        path = output / name
        if proof.linked(path) or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('missing/linked/oversized comparison metadata')
    receipt = json.loads((output / 'receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('protocol') != PROTOCOL or receipt.get('source_revision') != expected
            or receipt.get('request') != value or receipt.get('plan') != plan
            or type(receipt.get('accepted')) is not bool or receipt.get('automatic_retry') is not False
            or receipt.get('production_promotion') is not False
            or receipt.get('acceptance_scope') != 'controlled-training-timing-supplement'
            or any(receipt.get(k) is not False for k in ('checkpoint_byte_integrity_attested', 'verified_named_revision', 'terminal_model_evidence'))):
        raise ValueError('comparison source/request/plan binding mismatch')
    records = receipt.get('stage_records')
    original_start = receipt.get('comparison_started_monotonic')
    original_deadline = receipt.get('comparison_deadline_monotonic')
    if (not _finite(original_start) or not _finite(original_deadline)
            or original_deadline != original_start + value['policy']['deadlineSeconds']):
        raise ValueError('comparison original absolute deadline binding mismatch')
    if not isinstance(records, list) or not records or [r.get('stage') for r in records] != plan['execution_order'][:len(records)]:
        raise ValueError('comparison stage order/duplicate/unexpected stage mismatch')
    measurements, charge, binaries = [], 0, []
    for record in records:
        stage = record['stage']
        if record.get('artifact') != 'stages/' + stage:
            raise ValueError('comparison child artifact path mismatch')
        child = record.get('request')
        child_expected = copy.deepcopy(value)
        child_expected.pop('training_comparison')
        elapsed = record.get('elapsed_before_stage_seconds')
        remaining = record.get('remaining_original_deadline_seconds')
        if not _finite(elapsed) or elapsed < 0 or not _finite(remaining, positive=True) or remaining != value['policy']['deadlineSeconds'] - elapsed:
            raise ValueError('comparison refreshed or altered the original deadline')
        child_expected['policy']['deadlineSeconds'] = remaining
        if child != child_expected:
            raise ValueError('comparison stage changed settings/policy beyond remaining deadline')
        selected = stage_row(rows[0], stage)
        destination = output / record['artifact']
        proof.verify_artifacts(destination, temporary, child, expected, (selected,), backend)
        child_receipt = json.loads((destination / 'receipt.json').read_text(encoding='utf-8'))
        case = child_receipt.get('cases', [{}])[0]
        verify_stage_deadline(case, child, original_deadline)
        text = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in
                         (destination / '00000/body.stdout.log', destination / '00000/body.stderr.log') if p.is_file())
        accounting = update_accounting(selected, case, text)
        if record.get('update_accounting') != accounting:
            raise ValueError('comparison failure update charge differs from actual source observations')
        charge += accounting['conservative_charge']
        if child_receipt.get('accepted'):
            binaries.append(case['comparison_executable_sha256'])
            if record.get('executable_sha256') != binaries[-1]:
                raise ValueError('comparison executable source binding mismatch')
            measurements.append(case['training_comparison'])
        elif record is not records[-1] or receipt['accepted']:
            raise ValueError('comparison continued after failed stage or upgraded partial work')
    if charge != receipt.get('conservative_updates_charged') or charge > plan['max_physical_updates'] or len(set(binaries)) > 1:
        raise ValueError('comparison exceeded physical cap or changed native binary')
    if receipt['accepted']:
        measurement = aggregate(plan, measurements)
        if len(records) != len(plan['stages']) or receipt.get('measurement') != measurement:
            raise ValueError('accepted comparison is incomplete or metric decisions altered')
    elif 'measurement' in receipt:
        raise ValueError('failed comparison cannot report complete measurements')
    actual = _inventory(output)
    if actual != receipt.get('artifacts'):
        raise ValueError('comparison artifact hash/file inventory mismatch')
    return {'protocol': PROTOCOL, 'source_revision': expected, 'accepted': receipt['accepted'],
            'files': len(actual) + 2, 'bytes': sum(x['bytes'] for x in actual.values()) + sum((output / n).stat().st_size for n in ('receipt.json', 'summary.md')),
            'conservative_updates_charged': charge, 'automatic_retry': False,
            'production_promotion': False}
