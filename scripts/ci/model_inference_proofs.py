#!/usr/bin/env python3
"""Explicit model-test proofs. Never invoked by PR/push CI; no zero-body acceptance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import contextmanager, nullcontext
import tempfile
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.model_proof_parent_helpers import validate_parent_helpers
from scripts.ci.model_proof_resources import validate_child_visibility
from scripts.ci.model_proof_inputs import assess_inputs, receipt_claims, validate_contract
from scripts.ci import model_proof_multi_cuda as multi_cuda
from scripts.ci.native_darwin_sdk import select_sdk as select_native_darwin_sdk
from scripts.ci.prepare_iris_quant_proof_assets import validate_receipt as validate_iris_asset_receipt

REGISTRY_PATH = Path(__file__).with_name('model-inference-proofs.json')
IGNORE_PREFIX = 'model inference; explicit model-proof workflow'
SENSITIVE_ENVIRONMENT = re.compile(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|AUTHORIZATION|CLIENT_KEY|SIGNING_KEY|CERTIFICATE|PAT(?:_|$)', re.I)
LANES = ('mlx', 'candle-cpu', 'candle-metal', 'candle-cuda')
DEVICES = frozenset(('mlx-cpu', 'mlx-metal', 'candle-cpu', 'candle-metal', 'candle-cuda'))
ROLES = frozenset(('direct', 'cpu_child', 'nested_rank_cpu', 'nested_rank_direct', 'indexed_children'))
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_-]*$')
FQN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$')
RUNNERS = {
    'mlx': 'macos-15',
    'candle-cpu': 'ubuntu-latest',
    'candle-metal': 'macos-15',
    'candle-cuda': ['self-hosted', 'windows', 'cuda', 'real-weights'],
}
POLICY_BACKENDS = {'mlx': 'darwin-mlx', 'candle-metal': 'darwin-mlx',
                   'candle-cpu': 'linux-cpu', 'candle-cuda': 'windows-cuda'}
IRIS_QUANT_PROOFS = frozenset(
    f'{package}/integration/quant_real_weights::{test}'
    for package in ('mlx-gen-iris', 'candle-gen-iris')
    for test in ('real_tiers_shrink_the_resident_weights',
                 'real_quantized_renders_track_the_bf16_render_at_512')
)
IRIS_CUDA_CORRECTNESS_PROOFS = frozenset(
    binding for binding in IRIS_QUANT_PROOFS if binding.startswith('candle-gen-iris/')
)
IRIS_CUDA_CORRECTNESS_PROFILE = 'non-isolated-correctness-low-activity'
SHARED_CUDA_HEADROOM_PROFILE = 'non-isolated-correctness-headroom'


def identity(row: dict) -> str:
    return row['package'] + '/' + row['target'] + '/' + row['fqn']


def device_set(row: dict) -> frozenset[str]:
    if row.get('backend') == 'mlx' and 'expected_devices' not in row:
        return frozenset()  # Device-agnostic stream; observed in-scope set is still mandatory.
    raw = row.get('expected_devices', row['backend'])
    values = raw if isinstance(raw, list) else [raw]
    result = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError('backend/device names must be strings')
        value = value.strip().lower().replace(' ', '-')
        if value in ('cpu', 'metal', 'cuda'):
            if row['package'].startswith('candle-'):
                value = 'candle-' + value
            elif row['package'].startswith('mlx-'):
                value = 'mlx-' + value
            else:
                raise ValueError('non-provider package must declare its engine-qualified backend')
        if value not in DEVICES:
            raise ValueError('unknown actual-device declaration: ' + value)
        result.add(value)
    if not result:
        raise ValueError('empty actual-device declaration')
    return frozenset(result)


def lane(row: dict) -> str:
    if (row.get('configuration_kind') in ('feature-only', 'source-controls') and row.get('default_configuration') is None
            and 'selected_configuration' not in row):
        raise ValueError('explicit configuration required: source default is not a runnable complete NN proof')
    if 'execution_lane' in row:
        recipe = row.get('selected_configuration')
        if (row.get('configuration_kind') not in ('feature-only', 'source-controls') or not isinstance(recipe, dict)
                or recipe.get('execution_lane') != row['execution_lane']):
            raise ValueError('execution lane is authority only from an exact selected source recipe')
        return configuration_route(row, recipe)
    devices = device_set(row)
    if row.get('backend') == 'mlx':
        return 'mlx'
    if any(value.startswith('mlx-') for value in devices):
        if 'candle-cuda' in devices:
            raise ValueError('one proof process cannot span MLX/macOS and CUDA/Windows')
        return 'mlx'
    if 'candle-cuda' in devices:
        return 'candle-cuda'
    if 'candle-metal' in devices:
        return 'candle-metal'
    return 'candle-cpu'


def features(row: dict) -> list[str]:
    value = row.get('cargo_features', [])
    if not isinstance(value, list) or any(not isinstance(v, str) or not re.fullmatch(
            r'[A-Za-z0-9_][A-Za-z0-9_/-]*', v) for v in value):
        raise ValueError('cargo_features must be an explicit fixed feature-name list, never flags')
    return value


SD35_CONFIGURATION_CONTROLS = frozenset({'SD35_PARITY_CUDA', 'SD35_PARITY_SKIP_ENCODERS', 'SD35_PARITY_SKIP_DIT'})
SD35_CONFIGURATION_SOURCE = 'crates/media/candle-gen/candle-gen-sd3/tests/component_parity.rs'
RESERVED_CONFIGURATION_CONTROL = re.compile(
    r'^(?:PATH|HOME|USERPROFILE|PUBLIC|PYTHON.*|RUST.*|CARGO.*|LD_.*|DYLD_.*|GITHUB_.*|RUNNER_.*|'
    r'CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|MODEL_INFERENCE_.*|MODEL_PROOF_.*|SCENEWORKS_METAL_EXECUTION_LOCK_FD)$'
    r'|(?:^|_)(?:CHILD|PERMIT|ALLOW|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTHORIZATION|PRIVATE_KEY|ACCESS_KEY|SIGNING_KEY|CERTIFICATE)(?:_|$)'
    r'|(?:_PATH|_DIR|_ROOT|_FILE|_WEIGHTS)$', re.I)


def validate_source_control_names(row: dict, controls: list[str]) -> None:
    # A required source selector (e.g. FACE_DEVICE.expect) is not thereby an external asset.
    assets = set(row.get('asset_environment_names_in_source', []))
    assets.update(row.get('external_asset_environment', []))
    for name in controls:
        if (not re.fullmatch(r'[A-Z][A-Z0-9_]*', name) or RESERVED_CONFIGURATION_CONTROL.search(name)
                or SENSITIVE_ENVIRONMENT.search(name) or name in assets):
            raise ValueError('source control cannot inject process/child/asset/credential/device-authority environment: ' + name)


def sd35_source_devices(row: dict, environment: dict) -> frozenset[str]:
    """Source branch facts: encoders stay CPU; only selected, non-skipped DiT may use CUDA."""
    function = row['fqn'].split('::')[-1]
    cuda = environment.get('SD35_PARITY_CUDA') == '1'
    if function == 'sd35_bigg_pad_token_ab':
        if set(environment) != {'SD35_PARITY_CUDA'}:
            raise ValueError('SD35 A/B source reads only its CUDA control')
        return frozenset({'candle-cpu', 'candle-cuda'} if cuda else {'candle-cpu'})
    if function not in {'sd35_large_component_parity_vs_diffusers', 'sd35_medium_component_parity_vs_diffusers',
                        'sd35_large_turbo_component_parity_vs_diffusers'} or set(environment) != SD35_CONFIGURATION_CONTROLS:
        raise ValueError('fixed configuration is not an exact supported SD35 source/control binding')
    encoders = environment['SD35_PARITY_SKIP_ENCODERS'] != '1'
    dit = environment['SD35_PARITY_SKIP_DIT'] != '1'
    devices = {'candle-cpu'} if encoders or (dit and not cuda) else set()
    if dit and cuda:
        devices.add('candle-cuda')
    return frozenset(devices)


def configuration_manifest_features(row: dict, root: Path) -> dict:
    """Validate selected-package features, including an explicitly bound shared-source consumer."""
    selected = row.get('configuration_manifest')
    if selected is not None:
        if (not isinstance(selected, str) or Path(selected).is_absolute() or '..' in Path(selected).parts
                or Path(selected).name != 'Cargo.toml'):
            raise ValueError('configuration manifest must be an exact repository-relative Cargo manifest')
        manifest = root / selected
        if not manifest.is_file() or not manifest.resolve().is_relative_to(root.resolve()):
            raise ValueError('configuration manifest is missing or escapes source checkout')
        package = tomllib.loads(manifest.read_text(encoding='utf-8'))
        if package.get('package', {}).get('name') != row['package']:
            raise ValueError('configuration manifest is not the declared selected Cargo package')
        return package.get('features', {})
    directory = (root / row['source']).parent
    while directory.is_relative_to(root):
        manifest = directory / 'Cargo.toml'
        if manifest.is_file():
            if not manifest.resolve().is_relative_to(root.resolve()):
                raise ValueError('configuration feature manifest escapes source checkout')
            package = tomllib.loads(manifest.read_text(encoding='utf-8'))
            if package.get('package', {}).get('name') == row['package']:
                return package.get('features', {})
        if directory == root:
            break
        directory = directory.parent
    raise ValueError('configuration has no actual matching source-package feature manifest')


def configuration_route(row: dict, recipe: dict) -> str:
    devices = frozenset(recipe['expected_devices'])
    if {'candle-cuda', 'candle-metal'} <= devices:
        raise ValueError('one configuration cannot span incompatible CUDA/Metal execution hosts')
    actual_backend = ('candle-cuda' if 'candle-cuda' in devices else
                      'candle-metal' if 'candle-metal' in devices else 'candle-cpu')
    if recipe['backend'] != actual_backend:
        raise ValueError('configuration backend differs from its actual NN device set')
    route = recipe.get('execution_lane', actual_backend)
    declared = features(recipe)
    cuda = bool({'cuda', row['package'] + '/cuda'}.intersection(declared))
    metal = bool({'metal', row['package'] + '/metal'}.intersection(declared))
    if (route not in ('candle-cpu', 'candle-cuda', 'candle-metal')
            or route == 'candle-cpu' and (cuda or metal)
            or route == 'candle-cuda' and (not cuda or metal or 'candle-metal' in devices)
            or route == 'candle-metal' and (not metal or cuda or 'candle-cuda' in devices)):
        raise ValueError('configuration execution lane/features cannot compile or execute its source scope')
    return route


def validate_configurations(row: dict, *, root: Path | None = None) -> None:
    fields = {'configurations', 'default_configuration', 'configuration_controls'}
    if not (fields | {'configuration_kind'}).intersection(row):
        return
    kind = row.get('configuration_kind', 'source-flags')
    feature_only, source_controls = kind == 'feature-only', kind == 'source-controls'
    if kind not in ('source-flags', 'feature-only', 'source-controls') or not fields <= set(row):
        raise ValueError('configuration must declare its fixed kind, controls, recipes and source default')
    sd35 = row['package'] == 'candle-gen-sd3' and row['source'] == SD35_CONFIGURATION_SOURCE
    if sd35 and kind != 'source-flags':
        raise ValueError('SD35 retains its exact existing source-flag contract')
    if feature_only:
        if not row['package'].startswith('candle-') or row['configuration_controls'] != []:
            raise ValueError('feature-only source configurations cannot set any environment controls')
    elif source_controls:
        if not row['package'].startswith('candle-'):
            raise ValueError('fixed source controls require a source-declared Candle case')
    elif not sd35:
        raise ValueError('source-flag configurations require the declared SD35 source case')
    controls = row['configuration_controls']
    if (not isinstance(controls, list) or (not feature_only and not controls)
            or any(not isinstance(name, str) for name in controls)
            or len(set(controls)) != len(controls)):
        raise ValueError('configuration controls must be exact unique source-declared names')
    if source_controls:
        validate_source_control_names(row, controls)
    elif not feature_only and any(name not in SD35_CONFIGURATION_CONTROLS for name in controls):
        raise ValueError('source-flag controls must retain the exact declared SD35 names')
    recipes = row['configurations']
    default_name = row['default_configuration']
    if (not isinstance(recipes, dict) or not recipes or
            (default_name is None and not (feature_only or source_controls)) or
            (default_name is not None and (not isinstance(default_name, str) or default_name not in recipes))):
        raise ValueError('configured case requires an explicit source default or fixed source-bound null default')
    source_features = configuration_manifest_features(row, root) if root is not None else None
    for name, recipe in recipes.items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError('invalid fixed configuration name')
        required = {'source_identity', 'environment', 'backend', 'expected_devices', 'scope', 'required_work', 'cargo_features'}
        optional = {'backend_compatibility', 'source_guard'}
        if feature_only or source_controls:
            optional |= {'execution_lane', 'required_cuda_physical_index', 'cuda_activity_profile'}
        if not isinstance(recipe, dict) or not required <= set(recipe) or not set(recipe) <= required | optional:
            raise ValueError('configuration must bind source, flags, devices and required work')
        if any(not isinstance(recipe[field], str) or not recipe[field] for field in ('backend_compatibility', 'source_guard') if field in recipe):
            raise ValueError('source compatibility/guard metadata must be explicit strings')
        if any(not isinstance(recipe[field], str) or not recipe[field] for field in ('source_identity', 'scope')):
            raise ValueError('configuration source and scope must be explicit strings')
        if (feature_only or source_controls) and not recipe['source_identity'].startswith(row['source'] + '::'):
            raise ValueError('fixed recipe must retain its exact source identity')
        environment = recipe['environment']
        if not isinstance(environment, dict) or set(environment) != set(controls):
            raise ValueError('every declared configuration control requires an exact fixed setting or UNSET')
        if source_controls:
            if any(value is not None and (not isinstance(value, str) or len(value) > 4096 or (value and not value.isprintable()))
                   for value in environment.values()):
                raise ValueError('source control values must be fixed printable literals or explicit UNSET')
        elif any(value is not None and value != '1' for value in environment.values()):
            raise ValueError('every SD35 configuration control requires a fixed literal 1 or explicit UNSET')
        if environment.get('SD35_PARITY_SKIP_ENCODERS') == environment.get('SD35_PARITY_SKIP_DIT') == '1':
            raise ValueError('both SD35 work scopes cannot be skipped')
        work = recipe['required_work']
        if (not isinstance(work, list) or not work
                or any(not isinstance(value, str) or not value or len(value) > 4096 or not value.isprintable() for value in work)
                or len(set(work)) != len(work)):
            raise ValueError('configuration requires exact nonempty printable source work scopes')
        devices = recipe['expected_devices']
        if (not isinstance(devices, list) or not devices
                or any(not isinstance(device, str) or device not in ('candle-cpu', 'candle-cuda', 'candle-metal') for device in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError('configuration devices must be explicit actual Candle NN device sets')
        declared_features = features(recipe)
        if not {'model-inference-proof', row['package'] + '/model-inference-proof'}.intersection(declared_features):
            raise ValueError('configuration must explicitly enable its declared proof hooks')
        if source_features is not None:
            for feature in declared_features:
                if feature.removeprefix(row['package'] + '/') not in source_features:
                    raise ValueError('configuration feature is absent from actual source manifest: ' + feature)
        route = configuration_route(row, recipe)
        expected_activity_profile = (IRIS_CUDA_CORRECTNESS_PROFILE
                                     if all(key in row for key in ('package', 'target', 'fqn'))
                                     and identity(row) in IRIS_CUDA_CORRECTNESS_PROOFS
                                     and name in ('source_selected_cuda', 'source_selected_cuda_gpu0') else None)
        shared_headroom = recipe.get('cuda_activity_profile') == SHARED_CUDA_HEADROOM_PROFILE
        if shared_headroom and (expected_activity_profile is not None or route != 'candle-cuda'
                                or recipe.get('required_cuda_physical_index') != 1
                                or type(recipe.get('required_cuda_physical_index')) is not int
                                or not isinstance(row.get('required_cuda_device_uuid'), str)
                                or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',
                                                    row['required_cuda_device_uuid'])):
            raise ValueError('shared CUDA headroom profile requires an explicit GPU1 UUID/index source binding')
        if not shared_headroom and (('cuda_activity_profile' in recipe) != (expected_activity_profile is not None)
                or recipe.get('cuda_activity_profile') != expected_activity_profile
                or (expected_activity_profile is not None and route != 'candle-cuda')):
            raise ValueError('CUDA low-activity profile is limited to the exact Iris correctness recipes')
        physical_index = recipe.get('required_cuda_physical_index')
        if physical_index is not None and (type(physical_index) is not int or physical_index < 0
                                           or route != 'candle-cuda'):
            raise ValueError('fixed physical CUDA index requires a CUDA source recipe')
        if kind == 'source-flags':
            if frozenset(devices) != sd35_source_devices(row, environment):
                raise ValueError('configuration devices differ from actual source-selected SD35 work')
            if 'candle-cuda' in devices and environment.get('SD35_PARITY_CUDA') != '1':
                raise ValueError('CUDA configuration requires explicit source flag and compile feature')
    if default_name is None:
        if row['backend'] is not None or row.get('expected_devices') != [] or features(row) != []:
            raise ValueError('no-NN source default cannot invent a backend/device/build configuration')
        return
    default = recipes[default_name]
    if any(value is not None for value in default['environment'].values()):
        raise ValueError('source-default configuration must explicitly UNSET its source-read flags')
    if (row['backend'] != default['backend'] or device_set(row) != frozenset(default['expected_devices'])
            or features(row) != features(default)):
        raise ValueError('base case backend/devices must equal its explicit source-default recipe')


def configured_case(row: dict, name: str | None = None) -> dict:
    validate_configurations(row)
    if 'configurations' not in row:
        return row  # Preserve all legacy rows and their defaults byte-for-byte.
    selected = row['default_configuration'] if name is None else name
    if selected is None:
        raise ValueError('explicit configuration required: source default is not a runnable complete NN proof')
    if selected not in row['configurations']:
        raise ValueError('configuration is not declared for selected source case: ' + str(selected))
    recipe = row['configurations'][selected]
    result = {**row, 'backend': recipe['backend'], 'expected_devices': recipe['expected_devices'],
              'cargo_features': recipe['cargo_features'], 'selected_configuration': {'name': selected, **recipe}}
    if 'required_cuda_physical_index' in recipe:
        result['required_cuda_physical_index'] = recipe['required_cuda_physical_index']
    else:
        result.pop('required_cuda_physical_index', None)
    if 'execution_lane' in recipe:
        result['execution_lane'] = recipe['execution_lane']
    return result


def apply_configuration(row: dict, env: dict[str, str]) -> None:
    recipe = row.get('selected_configuration')
    if 'configurations' not in row:
        if recipe is not None:
            raise ValueError('legacy case cannot invent configuration authority')
        return
    if (not isinstance(recipe, dict) or not isinstance(recipe.get('name'), str)
            or recipe['name'] not in row['configurations']):
        raise ValueError('configured source case must resolve a declared recipe before execution')
    declared = row['configurations'][recipe['name']]
    if (recipe != {'name': recipe['name'], **declared} or row['backend'] != declared['backend']
            or row.get('expected_devices') != declared['expected_devices'] or features(row) != features(declared)
            or row.get('execution_lane') != declared.get('execution_lane')
            or row.get('required_cuda_physical_index') != declared.get('required_cuda_physical_index')):
        raise ValueError('effective configuration differs from the exact source declaration')
    base = {key: value for key, value in row.items() if key not in ('selected_configuration', 'execution_lane')}
    default_recipe = row['configurations'].get(row['default_configuration'], {})
    if 'required_cuda_physical_index' not in default_recipe:
        base.pop('required_cuda_physical_index', None)
    else:
        base['required_cuda_physical_index'] = default_recipe['required_cuda_physical_index']
    if row['default_configuration'] is None:
        base.update(backend=None, expected_devices=[], cargo_features=[])
    else:
        default = row['configurations'][row['default_configuration']]
        base.update(backend=default['backend'], expected_devices=default['expected_devices'], cargo_features=default['cargo_features'])
    validate_configurations(base)
    for name, value in recipe['environment'].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value


def load_registry(path: Path = REGISTRY_PATH, *, root: Path = ROOT) -> tuple[dict, ...]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('model-proof registry must be a nonempty array')
    seen = set()
    for row in rows:
        required = {'package', 'target', 'target_kind', 'cargo_target_metadata', 'fqn', 'source', 'execution_role', 'backend',
                    'fixture_identity', 'configuration_source', 'output_contract'}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError('incomplete model-proof row')
        if 'selected_configuration' in row or 'execution_lane' in row:
            raise ValueError('registry declares recipes, never runtime-selected configuration/lane authority')
        if any(not isinstance(row[field], str) for field in ('package', 'target', 'fqn', 'source')):
            raise ValueError('fixed Cargo/source identities must be strings')
        if not NAME.fullmatch(row['package']) or not NAME.fullmatch(row['target']):
            raise ValueError('invalid fixed Cargo package/target')
        if not FQN.fullmatch(row['fqn']):
            raise ValueError('FQN must be the ACTUAL libtest name, not a package/path guess')
        if not isinstance(row['execution_role'], str) or row['execution_role'] not in ROLES:
            raise ValueError('unverified execution role')
        if row['execution_role'].startswith('nested_rank_') and 'expected_successful_summaries' not in row:
            raise ValueError('nested rank proof requires exact source-declared successful summary cardinality')
        if type(row.get('no_default_features', False)) is not bool:
            raise ValueError('default-feature selection must be an explicit boolean')
        if 'expected_successful_summaries' in row and (type(row['expected_successful_summaries']) is not int or row['expected_successful_summaries'] <= 0):
            raise ValueError('successful-summary cardinality must be a positive source-declared integer')
        if row['execution_role'] == 'indexed_children':
            ids = row.get('expected_child_ids')
            if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                    or any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', value) for value in ids)):
                raise ValueError('indexed children require the exact source-declared unique child IDs')
        kind = row.get('target_kind', 'lib' if row['target'] == 'lib' else 'test')
        if kind not in ('lib', 'test', 'example', 'bin'):
            raise ValueError('unsupported fixed Cargo target kind')
        target_metadata = row.get('cargo_target_metadata')
        if target_metadata is None:
            raise ValueError('exact Cargo target metadata is required')
        if target_metadata is not None:
            if (not isinstance(target_metadata, dict) or target_metadata.get('name') != row['target']
                    or any(not isinstance(target_metadata.get(field), list) or not target_metadata[field]
                           or any(not isinstance(value, str) or not NAME.fullmatch(value) for value in target_metadata[field])
                           for field in ('kind', 'crate_types'))):
                raise ValueError('registered Cargo target metadata must retain actual names/kinds/crate-types')
            library_kinds = {'lib', 'rlib', 'staticlib', 'cdylib', 'dylib', 'proc-macro'}
            if not (bool(library_kinds.intersection(target_metadata['kind'])) if kind == 'lib'
                    else kind in target_metadata['kind']):
                raise ValueError('Cargo target selector differs from actual target metadata')
        source = root / row['source']
        if Path(row['source']).is_absolute() or '..' in Path(row['source']).parts or not source.is_file():
            raise ValueError('source must name an existing repository file')
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError('source escapes repository')
        for field in ('fixture_identity', 'configuration_source', 'output_contract'):
            if not row[field]:
                raise ValueError('missing proof identity/configuration/output contract')
        features(row)
        for field in ('required_environment', 'forbidden_root_environment', 'optional_environment', 'external_asset_environment'):
            values = row.get(field, [])
            if not isinstance(values, list) or any(not isinstance(name, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', name) for name in values):
                raise ValueError(field + ' must be explicit source-declared environment-name lists')
        required_uuid = row.get('required_cuda_device_uuid')
        if required_uuid is not None and (not isinstance(required_uuid, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', required_uuid)):
            raise ValueError('fixed physical CUDA UUID must be an actual UUID')
        required_index = row.get('required_cuda_physical_index')
        if required_index is not None and (type(required_index) is not int or required_index < 0):
            raise ValueError('fixed physical CUDA index must be a nonnegative integer')
        if (required_uuid is not None or required_index is not None) and lane(row) != 'candle-cuda':
            raise ValueError('physical CUDA restrictions require the CUDA execution route')
        validate_configurations(row, root=root)
        validate_child_visibility(row)
        validate_contract(row)
        validate_generated_adapter_artifacts(row)
        multi_cuda.validate_source(row)
        default_lane = ('configuration-required' if 'configurations' in row and row['default_configuration'] is None
                        else lane(configured_case(row)))
        key = (identity(row), default_lane, tuple(features(row)), row.get('deployment_target', '15.0'))
        if key in seen:
            raise ValueError('duplicate exact model-proof binding')
        seen.add(key)
    validate_parent_helpers(rows)
    bindings = [identity(row) for row in rows]
    if any('configurations' in row and bindings.count(identity(row)) != 1 for row in rows):
        raise ValueError('configured source case must have exactly one canonical binding')
    return tuple(rows)


def select(rows: tuple[dict, ...], selection: str, backend: str, configuration: str | None = None) -> tuple[dict, ...]:
    validate_parent_helpers(rows)
    if backend not in LANES:
        raise ValueError('unknown fixed proof backend')
    choices = {'all', *(row['package'] for row in rows), *(identity(row) for row in rows),
               *(row['package'] + '/' + row['fqn'] for row in rows)}
    if selection not in choices:
        raise ValueError('selection must be all, a registered package, or an exact registered binding')
    candidates = tuple(row for row in rows if selection in (
        'all', row['package'], identity(row), row['package'] + '/' + row['fqn']))
    if candidates and all(row.get('parent_only_helper') for row in candidates):
        raise ValueError('parent-only helper is not a standalone proof; select its registered parent cases')
    candidates = tuple(row for row in candidates if not row.get('parent_only_helper'))
    if configuration is not None and not any('configurations' in row for row in candidates):
        raise ValueError('selected cases declare no named configurations')
    effective = tuple(configured_case(row, configuration) for row in candidates)
    if configuration is not None and any('selected_configuration' in row and lane(row) != backend for row in effective):
        raise ValueError('selected configuration backend differs from requested execution lane')
    selected = tuple(row for row in effective if lane(row) == backend)
    if not selected:
        raise ValueError('selection has no cases for this backend; no empty proof')
    return selected


def binding(raw: str) -> tuple[str, dict]:
    """Existing default-branch dispatch bridge: no new inputs or shell-expanded JSON."""
    envelope = json.loads(raw)
    if not isinstance(envelope, dict) or set(envelope) != {'backend', 'request'}:
        raise ValueError('bridge binding requires exactly backend and request')
    backend = envelope['backend']
    if not isinstance(backend, str) or backend not in LANES:
        raise ValueError('unknown fixed bridge backend')
    return backend, request(json.dumps(envelope['request']), backend)


def request(raw: str, backend: str) -> dict:
    if backend not in LANES:
        raise ValueError('unknown fixed proof backend')
    value = json.loads(raw)
    if (not isinstance(value, dict) or not {'selection', 'policy'} <= set(value)
            or not set(value) <= {'selection', 'policy', 'runner', 'configuration'} | multi_cuda.FIELDS):
        raise ValueError('request requires selection and explicit resource policy; only runner/configuration may be optional')
    if 'configuration' in value and (not isinstance(value['configuration'], str) or not NAME.fullmatch(value['configuration'])):
        raise ValueError('configuration must select one fixed declared name')
    if value.get('runner', 'hosted') not in ('hosted', 'nax'):
        raise ValueError('runner must be hosted or the existing provisioned nax route')
    if value.get('runner') == 'nax' and backend not in ('mlx', 'candle-metal'):
        raise ValueError('nax is only the existing macOS route')
    if not isinstance(value['selection'], str) or not value['selection']:
        raise ValueError('selection must be explicitly provided')
    policy = value['policy']
    if not isinstance(policy, dict) or policy.get('backend') != POLICY_BACKENDS[backend]:
        raise ValueError('resource policy backend must match the fixed workflow lane')
    multi_cuda.validate_request(value, backend)
    return value


def command(row: dict, discover: bool) -> list[str]:
    kind = row.get('target_kind', 'lib' if row['target'] == 'lib' else 'test')
    args = ['cargo', 'test', '--locked']
    manifest = row.get('manifest_path')
    if manifest is None and row['package'] == 'ios-smoke':
        manifest = 'ios-host/smoke/Cargo.toml'
    if manifest is not None:
        if Path(manifest).is_absolute() or '..' in Path(manifest).parts:
            raise ValueError('manifest escapes repository')
        args += ['--manifest-path', manifest]
    else:
        args += ['-p', row['package']]
    if row.get('no_default_features', False):
        args.append('--no-default-features')
    if features(row):
        args += ['--features', ','.join(features(row))]
    if discover:
        args += ['--message-format=json']
    args += ['--lib'] if kind == 'lib' else ['--' + kind, row['target']]
    return args + [row['fqn'], '--', '--exact', '--ignored',
                   *(['--list'] if discover else ['--nocapture', '--test-threads=1'])]


def verify_discovery(row: dict, text: str) -> dict:
    if re.findall(r'^(.+): test$', text, re.M) != [row['fqn']]:
        raise ValueError('expected exactly the registered ignored test; no zero/multiple discovery')
    if not re.search(r'^1 test, 0 benchmarks$', text, re.M):
        raise ValueError('discovery must report exactly one test')
    expected = row.get('cargo_target_metadata')
    if expected is None:
        return {}  # Pure protocol mocks only; canonical production rows carry target metadata.
    artifacts = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if (isinstance(value, dict) and value.get('reason') == 'compiler-artifact'
                and value.get('profile', {}).get('test') and value.get('executable')):
            artifacts.append(value)
    if len(artifacts) != 1:
        raise ValueError('manual discovery requires one exact Cargo test executable identity')
    artifact = artifacts[0]
    actual = artifact['target']
    if (actual.get('name') != row['target'] or actual.get('name') != expected.get('name')
            or actual.get('kind') != expected.get('kind') or actual.get('crate_types') != expected.get('crate_types')):
        raise ValueError('compiled Cargo target identity differs from registered source metadata')
    return {'target': actual, 'package_id': artifact['package_id'], 'features': artifact.get('features', []),
            'executable': artifact['executable'], 'model_bodies_executed': 0}


def verify_indexed_execution(row: dict, text: str) -> dict:
    """Every declared child must complete its NN body and observe its own actual device scope."""
    ids = row.get('expected_child_ids')
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError('indexed proof has no exact source-declared child topology')
    fqn = re.escape(row['fqn'])
    if re.search(r'^test result: (?!ok\.)', text, re.M):
        raise ValueError('an indexed child/root failed')
    summaries = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text, re.M)
    if len(summaries) != len(ids) + 1 or any(value != ('1', '0', '0') for value in summaries):
        raise ValueError('indexed children require exactly N child + one root successful summaries')
    bodies = re.findall(r'^MODEL_INFERENCE_BODY_COMPLETED ' + fqn + r'#child=([^\r\n]+)$', text, re.M)
    if sorted(bodies) != sorted(ids):
        raise ValueError('indexed body IDs are missing/duplicate/unexpected')
    if text.splitlines().count('MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn']):
        raise ValueError('an unindexed BODY cannot replace indexed actual child bodies')
    parent = 'MODEL_INFERENCE_PARENT_COMPLETED ' + row['fqn']
    if text.splitlines().count(parent) != 1:
        raise ValueError('indexed proof requires exactly one waiting-root completion')
    by_child = {value: set() for value in ids}
    events = re.findall(r'^MODEL_INFERENCE_DEVICE ' + fqn + r'#child=([^ \t\r\n]+) ([a-z-]+)(?:[ \t][^\r\n]*)?$', text, re.M)
    for child, device in events:
        if child not in by_child or device not in DEVICES:
            raise ValueError('unexpected indexed device identity')
        by_child[child].add(device)
    expected = device_set(row)
    for child, devices in by_child.items():
        child_expected = frozenset(row.get('expected_child_devices', {}).get(child, expected))
        if (not devices or (child_expected and devices != child_expected)
                or (not child_expected and not all(value.startswith('mlx-') for value in devices))):
            raise ValueError('indexed actual-device scope missing or mismatched for ' + child)
    actual_union = set().union(*by_child.values())
    if expected and actual_union != expected:
        raise ValueError('indexed device union differs from the full case contract')
    return {'successful_libtest_summaries': len(summaries), 'body_completed': len(ids),
            'parent_completed': 1, 'actual_devices': sorted(set().union(*by_child.values())),
            'actual_child_devices': {child: sorted(devices) for child, devices in by_child.items()}}


def verify_execution(row: dict, text: str) -> dict:
    if row.get('parent_only_helper'):
        raise ValueError('parent-only helper cannot claim standalone runtime proof')
    if re.search(r'^MODEL_INFERENCE_PREREQUISITE_MISSING ' + re.escape(row['fqn']) + r'(?:#child=[A-Za-z0-9_.:-]+)?(?:[ \t]|$)', text, re.M):
        raise ValueError('required model work was skipped; no partial/zero-body proof')
    if row['execution_role'] == 'indexed_children':
        return verify_indexed_execution(row, text)
    if re.search(r'^test result: (?!ok\.)', text, re.M):
        raise ValueError('a root, CPU child or rank failed')
    results = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text, re.M)
    role = row['execution_role']
    minimum = 2 if role in ('cpu_child', 'nested_rank_cpu') else 1
    expected_count = row.get('expected_successful_summaries', minimum)
    if role in ('direct', 'cpu_child') and expected_count != minimum:
        raise ValueError('direct/CPU-child roles cannot silently become multi-child topologies')
    if type(expected_count) is not int or expected_count < minimum or len(results) != expected_count:
        raise ValueError('wrong completed execution-role count')
    if any(result != ('1', '0', '0') for result in results):
        raise ValueError('zero, ignored or failed body cannot be evidence')
    body = row.get('body_completion_marker', 'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'])
    if text.splitlines().count(body) != 1:
        raise ValueError('missing/duplicate full model-body completion')
    parent = row.get('parent_completion_marker')
    if role.startswith('nested_rank_'):
        parent = parent or 'MODEL_INFERENCE_PARENT_COMPLETED ' + row['fqn']
    if parent is not None and text.splitlines().count(parent) != 1:
        raise ValueError('missing/duplicate waiting-root completion')
    observed = frozenset(re.findall(r'^MODEL_INFERENCE_DEVICE ' + re.escape(row['fqn'])
                                    + r' ([a-z-]+)(?:[ \t][^\r\n]*)?$', text, re.M))
    expected = device_set(row)
    if (not observed or not observed <= DEVICES
            or (expected and observed != expected)
            or (not expected and not all(value.startswith('mlx-') for value in observed))):
        raise ValueError('actual in-scope device SET differs from the case contract: ' + repr(sorted(observed)))
    return {'successful_libtest_summaries': len(results), 'body_completed': 1,
            'parent_completed': 1 if parent is not None else None, 'actual_devices': sorted(observed)}


def repository_identity(expected: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{40}', expected):
        raise ValueError('expected source must be an exact lowercase Git revision')
    def git(*args: str) -> str:
        return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                              text=True, encoding='utf-8').stdout.strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != ROOT.resolve() or git('rev-parse', 'HEAD') != expected:
        raise ValueError('proof source is not the exact checked-out repository revision')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('proof source has tracked modifications')
    return {'revision': expected, 'tree': git('rev-parse', 'HEAD^{tree}')}


def require_host(backend: str, env: dict[str, str], route: str = 'hosted') -> None:
    if env.get('GITHUB_EVENT_NAME') != 'workflow_dispatch':
        raise ValueError('model proofs are explicit workflow_dispatch only')
    expected = 'Windows' if backend == 'candle-cuda' else 'Linux' if backend == 'candle-cpu' else 'macOS'
    if env.get('RUNNER_OS') != expected:
        raise ValueError('wrong proof host platform')
    if expected == 'macOS':
        if route == 'nax' and env.get('RUNNER_NAME') != 'nax-macos-2':
            raise ValueError('provisioned model route must be nax-macos-2, never the shared dev Mac')
        if route == 'hosted' and env.get('MODEL_PROOF_RUNNER_ENVIRONMENT') != 'github-hosted':
            raise ValueError('fixture route requires the isolated hosted Mac environment')


def linked(path: Path) -> bool:
    """Windows directory junctions/reparse points are links too, not safe owned ancestors."""
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        return True
    try:
        attributes = getattr(path.lstat(), 'st_file_attributes', 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def unlinked_path(path: Path, boundary: Path) -> None:
    """Reject linked ancestors below the owned boundary, not merely a linked final leaf."""
    boundary = boundary.absolute()
    path = path.absolute()
    if linked(boundary):
        raise ValueError('owned boundary must not be linked')
    if not path.is_relative_to(boundary):
        raise ValueError('path escapes owned boundary')
    for candidate in (path, *path.parents):
        if candidate == boundary:
            break
        if linked(candidate):
            raise ValueError('linked evidence ancestor is forbidden')
        if candidate.exists() and candidate != path and not candidate.is_dir():
            raise ValueError('non-directory evidence ancestor')


def owned_output(path: Path, temporary: Path) -> Path:
    unlinked_path(path, temporary)
    resolved = path.resolve()
    if linked(path) or not resolved.is_relative_to(temporary.resolve()) or resolved == temporary.resolve():
        raise ValueError('proof output must be a fresh owned directory below RUNNER_TEMP')
    if resolved.exists():
        raise ValueError('proof output directory already exists')
    resolved.mkdir(parents=True)
    return resolved


def validate_generated_adapter_spec(spec: dict) -> None:
    """Source-owned export shape contract, never caller/environment permission."""
    fields = {'kind', 'rank', 'dtype', 'per_target_alpha', 'max_bytes', 'targets'}
    if isinstance(spec, dict) and 'metadata_format' in spec:
        # Native FLUX3 export_adapter has its own established PEFT metadata.
        # This is an explicit dialect, not a global set of interchangeable aliases.
        fields |= {'metadata_format', 'alpha'}
        if (spec.get('metadata_format') != 'flux3' or spec.get('per_target_alpha') is not False
                or type(spec.get('alpha')) not in (int, float)
                or not math.isfinite(spec['alpha']) or spec['alpha'] <= 0):
            raise ValueError('native FLUX3 adapter needs its explicit format and positive expected alpha')
    if (not isinstance(spec, dict) or set(spec) != fields or spec['kind'] != 'lora'
            or type(spec['rank']) is not int or not 1 <= spec['rank'] <= 1024
            or spec['dtype'] not in ('F32', 'F16', 'BF16')
            or type(spec['per_target_alpha']) is not bool
            or type(spec['max_bytes']) is not int or not 1 <= spec['max_bytes'] <= 2**31
            or not isinstance(spec['targets'], dict) or not spec['targets']):
        raise ValueError('generated adapter needs an explicit bounded LoRA export contract')
    for name, shape in spec['targets'].items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.]*', name)
                or '..' in name or name.endswith('.') or not isinstance(shape, list) or len(shape) != 2
                or any(type(n) is not int or not spec['rank'] <= n <= 2**20 for n in shape)):
            raise ValueError('generated adapter targets need exact names and [out,in] dimensions')


def validate_generated_adapter_artifacts(row: dict) -> dict:
    """Only explicitly listed generated outputs can carry an adapter exception."""
    specs = row.get('generated_adapter_artifacts', {})
    if not isinstance(specs, dict):
        raise ValueError('generated adapter artifacts must be an exact output-path map')
    for relative, spec in specs.items():
        if (not isinstance(relative, str) or '\\' in relative or ':' in relative
                or Path(relative).is_absolute() or '..' in Path(relative).parts
                or Path(relative).suffix != '.safetensors'
                or relative not in row.get('output_artifacts', [])):
            raise ValueError('generated adapter permission must name a confined declared output artifact')
        validate_generated_adapter_spec(spec)
    return specs


def validate_generated_lora_bytes(raw: bytes, spec: dict) -> None:
    """Validate exact factors and packed payload ranges, without executing tensors.

    This validates an exported adapter's format, not its training history. Fresh
    owned generated paths and the selected source body's update/save/reload proof
    establish that lifecycle; input paths, links and generic checkpoints are not
    admitted by this header validator.
    """
    validate_generated_adapter_spec(spec)
    if len(raw) < 8 or len(raw) > spec['max_bytes']:
        raise ValueError('generated adapter is truncated or exceeds its exact file bound')
    size = int.from_bytes(raw[:8], 'little')
    if not 2 <= size <= 1024 * 1024 or size + 8 > len(raw):
        raise ValueError('generated adapter safetensors header length is invalid')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate generated adapter header key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('nonfinite generated adapter JSON constant: ' + value)
    try:
        header = json.loads(raw[8:8+size].decode('utf-8'), object_pairs_hook=unique,
                            parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid generated adapter safetensors header') from error
    if not isinstance(header, dict):
        raise ValueError('generated adapter safetensors header must be an object')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError('generated adapter metadata must contain only string values')
    if spec.get('metadata_format') == 'flux3':
        # candle-gen-flux3/src/training/checkpoint.rs::export_adapter. Base identity
        # strings are exporter metadata, not attestation of checkpoint bytes/revision.
        required = {'base_model', 'base_revision', 'network_type', 'rank', 'lora_alpha',
                    'scaling_convention', 'training_identity', 'lora_adapter_metadata'}
        if (not required <= set(metadata) or set(metadata) - required - {'fixture'}
                or metadata['network_type'] != 'lora' or metadata['rank'] != str(spec['rank'])
                or metadata['scaling_convention'] != 'alpha/rank'
                or any(not metadata[key] for key in required)):
            raise ValueError('native FLUX3 adapter metadata differs from its exporter contract')
        try:
            alpha = float(metadata['lora_alpha'])
            peft = json.loads(metadata['lora_adapter_metadata'], object_pairs_hook=unique,
                              parse_constant=reject_constant)
        except (ValueError, TypeError) as error:
            raise ValueError('native FLUX3 adapter has invalid alpha/PEFT metadata') from error
        expected_peft = {'peft_type': 'LORA', 'r': spec['rank'], 'lora_alpha': spec['alpha'],
                         'use_rslora': False, 'use_dora': False, 'rank_pattern': {}, 'alpha_pattern': {}}
        if (not math.isfinite(alpha) or alpha != spec['alpha'] or not isinstance(peft, dict)
                or peft != expected_peft or type(peft.get('r')) is not int
                or type(peft.get('lora_alpha')) not in (int, float)
                or peft.get('use_rslora') is not False or peft.get('use_dora') is not False):
            raise ValueError('native FLUX3 adapter alpha/rank/PEFT metadata conflicts with source contract')
    else:
        if metadata.get('networkType') != 'lora' or metadata.get('rank') != str(spec['rank']):
            raise ValueError('generated adapter metadata differs from its LoRA rank contract')
        try:
            if not math.isfinite(float(metadata['alpha'])):
                raise ValueError('nonfinite generated adapter alpha')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('generated adapter needs finite alpha metadata') from error
    expected = {}
    for name, (out_features, in_features) in spec['targets'].items():
        expected[name + '.lora_A.weight'] = (spec['dtype'], [spec['rank'], in_features])
        expected[name + '.lora_B.weight'] = (spec['dtype'], [out_features, spec['rank']])
        if spec['per_target_alpha']:
            expected[name + '.alpha'] = ('F32', [1])
    if set(header) != set(expected):
        raise ValueError('generated adapter has missing factors or unapproved/model tensor keys')
    spans = []
    for name, (dtype, shape) in expected.items():
        entry = header[name]
        if (not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}
                or entry['dtype'] != dtype or entry['shape'] != shape
                or any(type(n) is not int for n in entry['shape'])):
            raise ValueError('generated adapter factor shape/dtype differs from source contract')
        offsets = entry['data_offsets']
        length = math.prod(shape) * (4 if dtype == 'F32' else 2)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] - offsets[0] != length):
            raise ValueError('generated adapter factor payload range is invalid')
        spans.append(tuple(offsets))
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise ValueError('generated adapter payload has gaps or overlapping factors')
        end = stop
    if end != len(raw) - 8 - size:
        raise ValueError('generated adapter contains truncated or trailing payload')


def publish_file(source: Path, destination: Path, owned: Path, cap: int, *, log: bool = False, destination_root: Path | None = None, adapter: dict | None = None) -> dict:
    unlinked_path(source, owned)
    unlinked_path(destination, destination_root if destination_root is not None else owned)
    info = source.lstat()
    if linked(source) or not source.is_file() or info.st_nlink != 1:
        raise ValueError('evidence must be a regular non-linked owned file')
    if not source.resolve().is_relative_to(owned.resolve()) or info.st_size > cap:
        raise ValueError('evidence path/size violates the owned-output contract')
    if adapter is not None and (log or source.suffix != '.safetensors'):
        raise ValueError('generated adapter exception is only for its original safetensors format')
    if not log and adapter is None and source.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.svg', '.wav', '.flac', '.json', '.txt', '.csv', '.npy'}:
        raise ValueError('checkpoint/model payload is not an uploadable output')
    if not log and any(re.search(r'checkpoint|huggingface|models--|optimizer|credentials', part, re.I) for part in source.parts):
        raise ValueError('model/cache/credential path is not evidence')
    with source.open('rb') as handle:
        raw = handle.read(cap + 1)
    if adapter is not None:
        validate_generated_lora_bytes(raw, adapter)
    if log:
        text = raw.decode('utf-8', errors='replace')
        text = redact(text, dict(os.environ))
        raw = text.encode('utf-8')
    if len(raw) > cap:
        raise ValueError('redacted evidence exceeds its explicit byte budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_nlink != 1:
        raise ValueError('linked evidence destination is forbidden')
    destination.write_bytes(raw)
    return {'path': destination.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path: Path, value: object, *, owned: Path | None = None) -> None:
    unlinked_path(path, owned if owned is not None else path.parent)
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError('proof JSON destination must be a unique regular file')
    fd, staging = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.collecting', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)  # Only this call's newly-created owned staging file.



def redact(text: str, env: dict[str, str]) -> str:
    for name, value in env.items():
        if value and SENSITIVE_ENVIRONMENT.search(name):
            text = text.replace(value, '[REDACTED]')
    return text


def configuration_bindings(row: dict, env: dict[str, str]) -> dict:
    """Record registered/source-declared controls, never all environment values or credentials."""
    prerequisites = row.get('prerequisites', {})
    names = set()
    for values in (row.get('required_environment', []), row.get('optional_environment', []),
                   row.get('asset_environment_names_in_source', []),
                   row.get('route_metadata', {}).get('env_gates', []),
                   prerequisites.get('required_environment', []), prerequisites.get('optional_environment', [])):
        names.update(name for name in values if isinstance(name, str) and re.fullmatch(r'[A-Z][A-Z0-9_]*', name))
    if row.get('selected_configuration'):
        names.update(row['configuration_controls'])
    if env.get('CUDA_VISIBLE_DEVICES'):
        names.update(('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'))
    return {name: {'present': name in env,
                   'value': '[REDACTED]' if SENSITIVE_ENVIRONMENT.search(name) and env.get(name)
                   else redact(env[name], env) if name in env else None} for name in sorted(names)}


def configure_native_darwin_build(env, rustc, *, root, system):
    """Avoid Apple's malformed proc-macro LINKEDIT output; never configure a cross target.

    Cargo's encoded flags preserve paths containing spaces and take precedence over
    plain/target Rust flags. The proof controller builds only its native host target.
    Probe SDK/linker compatibility before discovery; native MLX verification stays independent.
    """
    if system != "Darwin":
        return {"mechanism": "not applicable"}

    def inspect(arguments):
        return subprocess.run([*rustc, *arguments], cwd=root, env=env, check=True,
                              capture_output=True, text=True, encoding="utf-8",
                              timeout=30).stdout.strip()

    version = inspect(["-vV"])
    match = re.search(r"^host: (\S+)$", version, re.M)
    if not match or not match.group(1).endswith("-apple-darwin"):
        raise ValueError("native Mac build requires the selected Darwin Rust host")
    host = match.group(1)
    if env.get("CARGO_BUILD_TARGET", host) != host:
        raise ValueError("native Mac proof cannot configure a cross target")
    sysroot = Path(inspect(["--print", "sysroot"]))
    linker = sysroot / "lib/rustlib" / host / "bin/rust-lld"
    if not sysroot.is_absolute() or not linker.is_file():
        raise ValueError("selected Rust toolchain lacks its native Darwin rust-lld")
    target_flags = "CARGO_TARGET_" + host.upper().replace("-", "_") + "_RUSTFLAGS"
    if "CARGO_ENCODED_RUSTFLAGS" in env:
        flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if env["CARGO_ENCODED_RUSTFLAGS"] else []
    else:
        # Cargo splits plain RUSTFLAGS on whitespace, rather than shell syntax.
        flags = env.get("RUSTFLAGS", env.get(target_flags, "")).split()
    flags += ["-C", "linker-flavor=ld64.lld", "-C", "linker=" + str(linker)]
    selected = env.copy()
    selected["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    for profile in ("DEV", "TEST"):
        selected[f"CARGO_PROFILE_{profile}_STRIP"] = "none"
        selected[f"CARGO_PROFILE_{profile}_BUILD_OVERRIDE_STRIP"] = "none"
    sdk, sdk_preflight = select_native_darwin_sdk(rustc, selected, root=root)
    selected["SDKROOT"] = sdk
    env.update(selected)
    return {"mechanism": "selected-toolchain native Darwin rust-lld",
            "host": host, "sysroot": str(sysroot), "linker": str(linker),
            "linker_sha256": hashlib.sha256(linker.read_bytes()).hexdigest(),
            "sdk": sdk, "rustflags": flags, "strip": "none", "sdk_preflight": sdk_preflight}


def verified_toolchain(env: dict[str, str], *, root: Path = ROOT) -> dict:
    """Bind compiler commands to this checkout's declaration, not a host PATH report."""
    declaration = root / 'rust-toolchain.toml'
    raw = declaration.read_bytes()
    channel = tomllib.loads(raw.decode('utf-8'))['toolchain']['channel']
    if not isinstance(channel, str) or not channel:
        raise ValueError('repository toolchain channel must be explicit')
    override = env.get('RUSTUP_TOOLCHAIN')
    if override and override != channel:
        raise ValueError('RUSTUP_TOOLCHAIN differs from repository toolchain declaration')
    if os.name == 'nt':
        return verified_windows_toolchain(env, root, raw, channel, override)
    cargo_home = Path(env.get('CARGO_HOME') or str(Path.home() / '.cargo'))
    if not cargo_home.is_absolute():
        raise ValueError('Cargo home must be absolute for verified Rustup selection')
    bin_dir = cargo_home / 'bin'
    suffix = '.exe' if os.name == 'nt' else ''
    proxies = {tool: bin_dir / (tool + suffix) for tool in ('cargo', 'rustc', 'rustup')}
    if not all(path.is_file() and os.access(path, os.X_OK) for path in proxies.values()):
        raise ValueError('required Rustup cargo/rustc/rustup proxies are missing')
    env['PATH'] = str(bin_dir) + os.pathsep + env.get('PATH', '')
    for tool, path in proxies.items():
        selected = shutil.which(tool, path=env['PATH'])
        if selected is None or Path(selected).resolve() != path.resolve():
            raise ValueError('compiler PATH does not select verified Rustup proxy: ' + tool)
    env['RUSTUP_TOOLCHAIN'] = channel

    def inspect(argv: list[str]) -> str:
        return subprocess.run(argv, cwd=root, env=env, check=True, capture_output=True,
                              text=True, timeout=30, encoding="utf-8").stdout.strip()

    binaries = {tool: inspect([str(proxies['rustup']), 'which', '--toolchain', channel, tool])
                for tool in ('cargo', 'rustc')}
    for name in ('RUSTC', 'CARGO_BUILD_RUSTC'):
        if env.get(name) and Path(env[name]).resolve() not in (proxies['rustc'].resolve(), Path(binaries['rustc']).resolve()):
            raise ValueError(name + ' differs from verified repository Rustup compiler')
    # RUSTC takes precedence over Cargo configuration, including a host build.rustc setting.
    env['RUSTC'] = str(proxies['rustc'])
    native_linker = configure_native_darwin_build(
        env, [str(proxies['rustc'])], root=root, system=platform.system())
    return {'declaration': 'rust-toolchain.toml', 'declaration_sha256': hashlib.sha256(raw).hexdigest(),
            'channel': channel, 'inherited_rustup_toolchain': override,
            'proxies': {name: str(path) for name, path in proxies.items()}, 'binaries': binaries,
            'cargo_version': inspect([str(proxies['cargo']), '--version', '--verbose']),
            'rustc_version': inspect([str(proxies['rustc']), '-vV']),
            'native_linker': native_linker}


def verified_windows_toolchain(env: dict[str, str], root: Path, declaration: bytes,
                               channel: str, override: str | None) -> dict:
    """Use the real pinned toolchain; Windows listener CARGO_HOME is a bin-less cache."""
    rustup_home = Path(env.get('RUSTUP_HOME') or str(Path(env['USERPROFILE']) / '.rustup'))
    expected = rustup_home / 'toolchains' / f'{channel}-x86_64-pc-windows-msvc' / 'bin'
    selected = Path(env.get('MODEL_PROOF_RUST_BIN', ''))
    if not selected.is_absolute() or selected.resolve() != expected.resolve():
        raise ValueError('selected Windows Rust bin differs from the pinned repository toolchain')
    binaries = {tool: expected / f'{tool}.exe' for tool in ('cargo', 'rustc')}
    if not all(path.is_file() and os.access(path, os.X_OK) for path in binaries.values()):
        raise ValueError('real pinned Windows cargo/rustc binaries are missing')
    env['PATH'] = str(expected) + os.pathsep + env.get('PATH', '')
    for tool, path in binaries.items():
        selected_tool = shutil.which(tool, path=env['PATH'])
        if selected_tool is None or Path(selected_tool).resolve() != path.resolve():
            raise ValueError('compiler PATH does not select the real pinned Windows toolchain: ' + tool)
    for name in ('RUSTC', 'CARGO_BUILD_RUSTC'):
        if env.get(name) and Path(env[name]).resolve() != binaries['rustc'].resolve():
            raise ValueError(name + ' differs from verified repository toolchain compiler')
    env['RUSTUP_TOOLCHAIN'] = channel
    env['RUSTC'] = str(binaries['rustc'])

    def inspect(tool: str, *args: str) -> str:
        return subprocess.run([str(binaries[tool]), *args], cwd=root, env=env, check=True,
                              capture_output=True, text=True, timeout=30, encoding='utf-8').stdout.strip()

    cargo_version = inspect('cargo', '--version', '--verbose')
    rustc_version = inspect('rustc', '-vV')
    if not re.search(rf'(?m)^cargo {re.escape(channel)}\b', cargo_version) or not re.search(
            rf'(?m)^release: {re.escape(channel)}$', rustc_version):
        raise ValueError('real Windows toolchain version differs from repository pin')
    return {'declaration': 'rust-toolchain.toml', 'declaration_sha256': hashlib.sha256(declaration).hexdigest(),
            'channel': channel, 'inherited_rustup_toolchain': override,
            'proxies': {}, 'binaries': {name: str(path) for name, path in binaries.items()},
            'cargo_version': cargo_version, 'rustc_version': rustc_version}


def run(rows: tuple[dict, ...], backend: str, value: dict, expected: str, output: Path) -> int:
    if any(row.get('parent_only_helper') for row in rows):
        raise ValueError('parent-only helper must execute only through registered parent cases')
    if any(lane(row) != backend for row in rows):
        raise ValueError('execution route differs from selected source configuration lane')
    require_host(backend, dict(os.environ), value.get('runner', 'hosted'))
    source_identity = repository_identity(expected)
    artifact = owned_output(output, Path(os.environ['RUNNER_TEMP']))
    work = artifact.parent / (artifact.name + '-work')
    if work.exists() or work.is_symlink():
        raise ValueError('proof working directory must be fresh')
    work.mkdir()
    sys.path.insert(0, str(ROOT))
    from scripts.media_campaign_supervisor import load_policy, run_guarded, SupervisionError, SystemProbe
    from scripts.ci.model_proof_resources import CudaClaimRejected, CudaReleaseRejected, metadata_policy_payload, execution_claim
    policy_path = work / 'policy.json'
    write_json(policy_path, value['policy'])
    policy = load_policy(policy_path)
    approved_uuids = multi_cuda.bind_rows(rows, value, backend)
    additional_policies = []
    for role, raw in enumerate(value.get('additional_cuda_policies', []), 1):
        extra_path = work / f'cuda-policy-role-{role}.json'
        write_json(extra_path, raw)
        additional_policies.append(load_policy(extra_path))
    cuda_policies = (policy, *additional_policies)
    metadata_path = work / 'metadata-policy.json'
    write_json(metadata_path, metadata_policy_payload(value['policy']))
    discovery_policy = load_policy(metadata_path)
    receipt = {'schema_version': 1, 'source': source_identity, 'host': {
        'platform': platform.platform(), 'machine': platform.machine(),
        'runner_name': os.environ.get('RUNNER_NAME'), 'runner_os': os.environ.get('RUNNER_OS')},
        'lane': backend, 'resource_policy': json.loads(policy.canonical_bytes),
        'policy_sha256': policy.sha256, 'selected': len(rows), 'selected_bindings': [identity(row) for row in rows],
        'request_selection': value['selection'], 'completed': 0,
        'accepted': False, 'cases': []}
    receipt.update(receipt_claims([], False))
    if approved_uuids:
        receipt['approved_cuda_uuid_roles'] = list(approved_uuids)
        receipt['additional_resource_policies'] = [json.loads(item.canonical_bytes) for item in additional_policies]
    if any('selected_configuration' in row for row in rows):
        receipt['request_configuration'] = value.get('configuration')
    code = 1
    try:
        for index, row in enumerate(rows):
            case_work = work / f'{index:05d}'
            case_artifact = artifact / f'{index:05d}'
            case_work.mkdir()
            case_artifact.mkdir()
            generated = case_work / 'generated'
            generated.mkdir()
            env = dict(os.environ, MODEL_INFERENCE_CASE_FQN=row['fqn'], MODEL_PROOF_OUTPUT_DIR=str(generated))
            apply_configuration(row, env)
            if identity(row) in IRIS_QUANT_PROOFS and row['fqn'].endswith('real_quantized_renders_track_the_bf16_render_at_512'):
                env['IRIS_OUT'] = str(generated)
            if backend == 'candle-cuda':
                env['CUDA_VISIBLE_DEVICES'] = ','.join(approved_uuids) if approved_uuids else policy.cuda_device_uuid
                env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                env['RUSTC_WRAPPER'] = ''
            result = {'binding': identity(row), 'declared': row, 'actual_devices': [],
                      'fixture_identity_basis': 'Exact source plus actual successful input receipts; source descriptors alone never attest external inputs or checkpoint integrity',
                      'source_file_sha256': hashlib.sha256((ROOT / row['source']).read_bytes()).hexdigest(),
                      'configuration_environment': configuration_bindings(row, env),
                      'cargo_configuration': {'features': features(row), 'default_features': not row.get('no_default_features', False),
                                              'profile': 'debug', 'target_selector': row.get('target_kind', 'lib' if row['target'] == 'lib' else 'test'),
                                              'registered_cargo_target_metadata': row.get('cargo_target_metadata')},
                      'accepted': False, 'body_proof_accepted': False, 'resource_verification_pending': True,
                      'commands': [], 'outputs': []}
            result.update(receipt_claims([], False))
            receipt['cases'].append(result)
            started = time.monotonic()
            try:
                validate_contract(row, required=True)
                forbidden = [name for name in row.get('forbidden_root_environment', []) if env.get(name)]
                if forbidden:
                    raise ValueError('reserved child-role environment present at root: ' + ','.join(forbidden))
                missing = [name for name in row.get('required_environment', []) if not env.get(name)]
                if missing:
                    raise ValueError('missing declared model/fixture prerequisites: ' + ','.join(missing))
                if identity(row) in IRIS_QUANT_PROOFS:
                    path = env.get('MODEL_PROOF_ASSET_RECEIPT')
                    if not path:
                        raise ValueError('missing authenticated Iris asset preflight receipt')
                    result['asset_preflight'] = validate_iris_asset_receipt(
                        json.loads(Path(path).read_text(encoding='utf-8')), backend, env)
                if backend == 'mlx':
                    env['MACOSX_DEPLOYMENT_TARGET'] = mlx_build_deployment(row)
                result['toolchain'] = verified_toolchain(env, root=ROOT)
                result['build_preparation'] = prepare_build(row, env)
                for discover in (True, False):
                    phase = 'discovery' if discover else 'body'
                    stdout = case_work / (phase + '.stdout.log')
                    stderr = case_work / (phase + '.stderr.log')
                    argv = command(row, discover)
                    if discover:
                        claim = nullcontext({'mechanism': 'metadata-only discovery; no model execution'})
                    elif approved_uuids:
                        claim = multi_cuda.execution_claim(row, cuda_policies, value, case_work, env)
                    else:
                        claim = execution_claim(backend, policy, case_work, env, constraints=row)
                    supervision = {}
                    if approved_uuids and not discover:
                        probes = tuple(SystemProbe(item) for item in cuda_policies)
                        supervision = {'probe': probes[0], 'additional_cuda_policies': tuple(additional_policies),
                                       'additional_cuda_probes': probes[1:]}
                    with claim as physical:
                        result['physical_execution'] = physical
                        guarded = run_guarded(argv, cwd=ROOT, env=env, policy=discovery_policy if discover else policy,
                                              stdout_path=stdout, stderr_path=stderr,
                                              event_path=case_work / (phase + '.events.jsonl'),
                                              require_gpu_sample=False,
                                              pass_fds=((int(env['SCENEWORKS_METAL_EXECUTION_LOCK_FD']),)
                                                        if 'SCENEWORKS_METAL_EXECUTION_LOCK_FD' in env else ()), **supervision)
                        result['commands'].append({'argv': argv, 'supervision': asdict(guarded), 'owned_tree_empty': True})
                        if approved_uuids and not discover:
                            result['cuda_devices'] = list(guarded.cuda_devices)
                        if not discover:
                            body_text = stdout.read_text(encoding='utf-8', errors='replace') + '\n' + stderr.read_text(encoding='utf-8', errors='replace')
                            result.update(verify_execution(row, body_text))
                            result.update(assess_inputs(row, redact(body_text, env)))
                            result['body_proof_accepted'] = True
                            # This is a staged body proof, NEVER preliminary whole-run acceptance.
                            write_json(case_artifact / 'result.json', result, owned=artifact)
                    if not discover:
                        if physical.get('post_execution_verified') is not True:
                            raise ValueError('outer physical controller did not seal resource completion')
                        result['resource_verification_pending'] = False
                    result['gpu_peak_observation'] = gpu_peak_observation(guarded)
                    text = stdout.read_text(encoding='utf-8', errors='replace') + '\n' + stderr.read_text(encoding='utf-8', errors='replace')
                    if guarded.returncode:
                        raise ValueError('proof child returned nonzero')
                    if discover:
                        result['compiled_target'] = verify_discovery(row, text)
                    else:
                        result.update(verify_execution(row, text))
                    for path, cap in ((stdout, policy.stdout_cap_bytes), (stderr, policy.stderr_cap_bytes),
                                      (case_work / (phase + '.events.jsonl'), policy.event_cap_bytes)):
                        # The supervisor bounds observer events if one is written; it does not
                        # create that optional file. Never synthesize an event or infer one from
                        # a successful child. Existing non-file paths still fail in publish_file.
                        if path.name.endswith('.events.jsonl') and not (path.exists() or path.is_symlink()):
                            continue
                        published = publish_file(path, case_artifact / path.name, case_work, cap, log=True, destination_root=artifact)
                        published['path'] = (case_artifact / path.name).relative_to(artifact).as_posix()
                        result['outputs'].append(published)
                adapter_specs = validate_generated_adapter_artifacts(row)
                for relative in row.get('output_artifacts', []):
                    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
                        raise ValueError('known output must be confined to the owned generated directory')
                    destination = case_artifact / 'generated' / relative
                    adapter = adapter_specs.get(relative)
                    cap = adapter['max_bytes'] if adapter is not None else policy.stdout_cap_bytes
                    published = publish_file(generated / relative, destination, generated, cap, destination_root=artifact, adapter=adapter)
                    published['path'] = destination.relative_to(artifact).as_posix()
                    result['outputs'].append(published)
                repository_identity(expected)
                if result['input_grade'] == 'no-input-executed':
                    raise ValueError('external-absent: no input executed; not successful case execution')
                if not result['body_proof_accepted'] or result['resource_verification_pending']:
                    raise ValueError('model body/resource proof is incomplete')
                if approved_uuids:
                    multi_cuda.validate_case_evidence(row, value, result)
                result['accepted'] = True
                receipt['completed'] += 1
            except (ValueError, OSError, SupervisionError) as error:
                result['error'] = redact(str(error), env)
                if isinstance(error, CudaClaimRejected):
                    result['cuda_claim_diagnostic'] = error.diagnostic
                if isinstance(error, CudaReleaseRejected):
                    result['cuda_release_diagnostic'] = error.diagnostic
                if isinstance(error, SupervisionError):
                    result['supervision_failure'] = {'reason': error.reason, 'admission': error.admission,
                                                     'pid': error.pid, 'watchdog_child_footprint': error.watchdog_child_footprint}
                    if approved_uuids:
                        result['cuda_devices'] = list(getattr(error, 'cuda_devices', ()))
                # Publish only bounded/redacted regular logs from this owned case, even on failure.
                known_logs = [(case_work / (phase + suffix), cap)
                              for phase in ('discovery', 'body')
                              for suffix, cap in (('.stdout.log', policy.stdout_cap_bytes),
                                                  ('.stderr.log', policy.stderr_cap_bytes),
                                                  ('.events.jsonl', policy.event_cap_bytes))]
                for path, cap in known_logs:
                    if not path.exists():
                        continue
                    try:
                        published = publish_file(path, case_artifact / path.name, case_work, cap, log=True, destination_root=artifact)
                        published['path'] = (case_artifact / path.name).relative_to(artifact).as_posix()
                        result['outputs'] = [item for item in result['outputs'] if item['path'] != published['path']]
                        result['outputs'].append(published)
                    except (ValueError, OSError) as invalid:
                        result.setdefault('rejected_outputs', []).append(str(invalid))
                break
            finally:
                result['duration_seconds'] = round(time.monotonic() - started, 6)
                write_json(case_artifact / 'result.json', result, owned=artifact)
                write_json(artifact / 'receipt.json', receipt, owned=artifact)
        receipt['accepted'] = bool(rows) and receipt['completed'] == len(rows)
        receipt.update(receipt_claims(receipt['cases'], receipt['accepted']))
        code = 0 if receipt['accepted'] else 1
        return code
    finally:
        receipt.update(receipt_claims(receipt['cases'], receipt['accepted']))
        write_json(artifact / 'receipt.json', receipt, owned=artifact)
        summary = ['## Explicit model inference proofs', '',
                   f"Source `{expected}`; lane `{backend}`; completed {receipt['completed']}/{len(rows)}; accepted={receipt['accepted']}.", '',
                   '| Binding | Actual devices | Accepted | Seconds |', '|---|---|---|---|']
        summary += [f"| `{case['binding']}` | {','.join(case.get('actual_devices', []))} | {case['accepted']} | {case.get('duration_seconds', 0)} |" for case in receipt['cases']]
        configured = [case for case in receipt['cases'] if case['declared'].get('selected_configuration')]
        if configured:
            summary += ['', '### Fixed source configuration scopes', '']
            for case in configured:
                recipe = case['declared']['selected_configuration']
                summary += [f"- `{case['binding']}` / `{recipe['name']}`: **{recipe['scope']}**.",
                            '  Required source work: ' + ', '.join(recipe['required_work']) + '.',
                            '  Source configuration: `' + recipe['source_identity'] + '`.']
                if 'execution_lane' in recipe:
                    summary += ['  Compile/resource execution lane: `' + recipe['execution_lane'] + '`; actual NN device contract: '
                                + ', '.join(recipe['expected_devices']) + '. The lane is not NN device provenance.']
        summary += ['', '### Acceptance scope and actual input provenance', '',
                    'Scope: `' + receipt['acceptance_scope'] + '`; input grade: `' + receipt['input_grade'] + '`.',
                    'Checkpoint-byte integrity attested: ' + str(receipt['checkpoint_byte_integrity_attested'])
                    + '; verified named revision: False; standalone terminal model evidence: False.']
        summary += ['- ' + limit for limit in receipt['input_claim_limits']]
        for case in receipt['cases']:
            summary += ['- `' + case['binding'] + '`: scope `' + case['acceptance_scope']
                        + '`, input grade `' + case['input_grade'] + '`.']
        text = '\n'.join(summary) + '\n'
        unlinked_path(artifact / 'summary.md', artifact)
        if (artifact / 'summary.md').exists() and (artifact / 'summary.md').stat().st_nlink != 1:
            raise ValueError('linked proof summary destination is forbidden')
        (artifact / 'summary.md').write_text(text, encoding='utf-8')
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as stream:
                stream.write(text)


def gpu_peak_observation(guarded: object) -> str:
    """Keep a global card upper bound distinct from attributed child GPU use."""
    upper_bound_sampled = any(sample.get('phase') == 'cuda-global-used-upper-bound-sample'
                              for sample in guarded.samples)
    if hasattr(guarded, 'cuda_devices'):
        upper_bound_sampled |= any(device.get('globalUsedUpperBoundSampleCount', 0) > 0
                                   for device in guarded.cuda_devices)
    if guarded.peak_gpu_bytes is None:
        return ('global-used-upper-bound-sampled/not-child-attributed'
                if upper_bound_sampled else 'unknown/not-attributed-before-exit')
    return 'attributed-and-global-upper-bound-sampled' if upper_bound_sampled else 'sampled'


def mlx_build_deployment(row: dict) -> str:
    deployment = row.get('deployment_target', '15.0')
    if deployment not in ('15.0', '26.2'):
        raise ValueError('unregistered MLX deployment configuration')
    return deployment


def prepare_build(row: dict, env: dict[str, str]) -> dict:
    """Use the existing published-native helper; never download or copy a model checkpoint."""
    if not any(device.startswith('mlx-') for device in device_set(row)) and row.get('backend') != 'mlx':
        return {'prebuilt': 'not applicable'}
    deployment = mlx_build_deployment(row)
    env['MACOSX_DEPLOYMENT_TARGET'] = deployment
    env.pop('PMETAL_MLX_PREBUILT_DIR', None)
    env.pop('PMETAL_METALLIB_PATH', None)
    result = subprocess.run([str(ROOT / 'scripts/fetch-prebuilt-mlx.sh'), '--deployment-target', deployment,
                             '--build-type', 'Debug'], cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding='utf-8')
    if result.returncode not in (0, 1):
        raise ValueError('native prebuilt integrity/preparation failed: ' + result.stderr[-2000:])
    if result.returncode == 1:
        return {'prebuilt': 'source fallback', 'deployment_target': deployment, 'diagnostic': redact(result.stderr[-2000:], env)}
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if set(values) != {'PMETAL_MLX_PREBUILT_DIR', 'PMETAL_METALLIB_PATH'}:
        raise ValueError('native helper did not provide both verified prebuilt paths')
    env.update(values)
    return {'prebuilt': 'verified existing archive', 'deployment_target': deployment,
            'metallib_path': values['PMETAL_METALLIB_PATH']}


def verify_artifacts(output: Path, temporary: Path, value: dict, expected: str,
                     approved_rows: tuple[dict, ...], backend: str | None = None) -> dict:
    """Upload only exact approved source cases/known outputs, after the outer resource seal.

    Stream budgets are per command: discovery and body each get those explicit limits.
    Each case/result metadata copy gets at most their sum; no unbounded aggregate upload.
    """
    if not approved_rows:
        raise ValueError('an approved nonempty selection is required for evidence validation')
    unlinked_path(output, temporary)
    root = output.resolve()
    if linked(output) or not root.is_relative_to(temporary.resolve()) or root == temporary.resolve():
        raise ValueError('evidence root escapes owned workflow output')
    receipt_path = root / 'receipt.json'
    if linked(receipt_path) or not receipt_path.is_file() or receipt_path.stat().st_nlink != 1:
        raise ValueError('missing unique regular proof receipt')
    stream_caps = {'.stdout.log': value['policy']['stdoutCapBytes'],
                   '.stderr.log': value['policy']['stderrCapBytes'],
                   '.events.jsonl': value['policy']['eventCapBytes']}
    metadata_cap = sum(stream_caps.values())
    file_caps = {'receipt.json': metadata_cap * (len(approved_rows) + 1), 'summary.md': metadata_cap}
    case_allowed = []
    adapter_contracts = {}
    for index, row in enumerate(approved_rows):
        prefix = f'{index:05d}/'
        file_caps[prefix + 'result.json'] = metadata_cap
        allowed = set()
        for phase in ('discovery', 'body'):
            for suffix, cap in stream_caps.items():
                relative = prefix + phase + suffix
                file_caps[relative] = cap
                allowed.add(relative)
        adapter_specs = validate_generated_adapter_artifacts(row)
        for generated in row.get('output_artifacts', []):
            if not isinstance(generated, str) or Path(generated).is_absolute() or '..' in Path(generated).parts:
                raise ValueError('approved generated output escapes its owned directory')
            relative = prefix + 'generated/' + generated
            adapter = adapter_specs.get(generated)
            file_caps[relative] = adapter['max_bytes'] if adapter is not None else value['policy']['stdoutCapBytes']
            if adapter is not None:
                adapter_contracts[relative] = adapter
            allowed.add(relative)
        case_allowed.append(allowed)
    if receipt_path.stat().st_size > file_caps['receipt.json']:
        raise ValueError('receipt exceeds its derived explicit metadata budget')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('source', {}).get('revision') != expected:
        raise ValueError('evidence belongs to another source revision')
    approved_bindings = [identity(row) for row in approved_rows]
    approved_uuids = ()
    if multi_cuda.FIELDS.intersection(value) or any(all(row.get(key) == item for key, item in multi_cuda.CASE.items()) for row in approved_rows):
        approved_uuids = multi_cuda.bind_rows(approved_rows, value, 'candle-cuda')
        if (receipt.get('approved_cuda_uuid_roles') != list(approved_uuids)
                or receipt.get('additional_resource_policies') != value['additional_cuda_policies']):
            raise ValueError('evidence differs from explicit ordered multi-GPU authorization and per-UUID policies')
    if (type(receipt.get('selected')) is not int or receipt.get('selected_bindings') != approved_bindings or receipt.get('selected') != len(approved_rows)
            or receipt.get('request_selection') != value['selection'] or receipt.get('resource_policy') != value['policy']):
        raise ValueError('evidence differs from its approved source selection/resource policy')
    if any('selected_configuration' in row for row in approved_rows) and receipt.get('request_configuration') != value.get('configuration'):
        raise ValueError('evidence differs from its approved fixed configuration selection')
    cases = receipt.get('cases')
    if (type(receipt.get('accepted')) is not bool or not isinstance(cases, list)
            or len(cases) > len(approved_rows) or type(receipt.get('completed')) is not int):
        raise ValueError('malformed proof completion state')
    if receipt.get('accepted') and (receipt['completed'] != len(approved_rows) or len(cases) != len(approved_rows)):
        raise ValueError('accepted receipt does not contain every selected completed case')
    expected_files = {'receipt.json', 'summary.md'}
    hashes = {}
    for index, case in enumerate(cases):
        if case.get('declared') != approved_rows[index] or case.get('binding') != approved_bindings[index]:
            raise ValueError('case evidence does not match the exact approved registered binding')
        recipe = approved_rows[index].get('selected_configuration')
        if recipe is not None:
            recorded = case.get('configuration_environment', {})
            for name, setting in recipe['environment'].items():
                if recorded.get(name) != {'present': setting is not None, 'value': setting}:
                    raise ValueError('effective source control differs from approved fixed configuration: ' + name)
        if type(case.get('accepted')) is not bool:
            raise ValueError('case acceptance must be boolean')
        if case['accepted'] and (case.get('body_proof_accepted') is not True or case.get('resource_verification_pending') is not False
                                 or case.get('physical_execution', {}).get('post_execution_verified') is not True):
            raise ValueError('pre-seal body proof cannot be uploaded as accepted')
        result_name = f'{index:05d}/result.json'
        expected_files.add(result_name)
        case_path = root / result_name
        unlinked_path(case_path, root)
        if (not case_path.is_file() or case_path.stat().st_nlink != 1 or case_path.stat().st_size > file_caps[result_name]
                or json.loads(case_path.read_text(encoding='utf-8')) != case):
            raise ValueError('case result is missing or differs from the final receipt')
        for item in case.get('outputs', []):
            relative = item['path']
            if relative not in case_allowed[index] or relative in hashes:
                raise ValueError('unknown/duplicate output, not an approved source-defined artifact')
            expected_files.add(relative)
            hashes[relative] = item
        required_outputs = {f'{index:05d}/generated/{name}' for name in approved_rows[index].get('output_artifacts', [])}
        if case['accepted'] and not required_outputs <= {item['path'] for item in case.get('outputs', [])}:
            raise ValueError('accepted case omits a required generated output artifact')
    if receipt['completed'] != sum(case['accepted'] for case in cases):
        raise ValueError('completed count differs from sealed accepted cases')
    found = set()
    total = 0
    aggregate_cap = sum(file_caps.values())
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if linked(Path(directory) / name):
                raise ValueError('symlink directory is not uploadable evidence')
        for name in files:
            path = Path(directory) / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if linked(path) or not path.is_file() or info.st_nlink != 1 or relative not in expected_files:
                raise ValueError('unknown/linked/nonregular file in evidence tree')
            if info.st_size > file_caps[relative]:
                raise ValueError('evidence file exceeds its explicit/derived byte budget')
            total += info.st_size
            if total > aggregate_cap:
                raise ValueError('evidence tree exceeds its bounded aggregate byte budget')
            if relative in hashes:
                recorded = hashes[relative]
                with path.open('rb') as handle:
                    raw = handle.read(file_caps[relative] + 1)
                if info.st_size != recorded['bytes'] or len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != recorded['sha256']:
                    raise ValueError('published evidence bytes differ from their recorded hash')
                if relative in adapter_contracts:
                    validate_generated_lora_bytes(raw, adapter_contracts[relative])
            found.add(relative)
    if found != expected_files:
        raise ValueError('missing evidence files; no partial upload')
    # Only read approved, bounded, unlinked, hash-checked body logs after the complete tree check.
    # A resource-completion seal or edited JSON flag can never upgrade weak input provenance.
    for index, case in enumerate(cases):
        if not case['accepted']:
            continue
        if identity(approved_rows[index]) in IRIS_QUANT_PROOFS:
            validate_iris_asset_receipt(case.get('asset_preflight'), backend)
        logs = [f'{index:05d}/body.stdout.log', f'{index:05d}/body.stderr.log']
        if any(name not in hashes for name in logs):
            raise ValueError('accepted input claims require their exact bounded body logs')
        body = '\n'.join((root / name).read_text(encoding='utf-8', errors='replace') for name in logs)
        execution = verify_execution(approved_rows[index], body)
        if any(case.get(key) != value for key, value in execution.items()):
            raise ValueError('accepted execution differs from its actual body logs')
        if approved_uuids:
            multi_cuda.validate_case_evidence(approved_rows[index], value, case)
        assessment = assess_inputs(approved_rows[index], body)
        if assessment['input_grade'] == 'no-input-executed':
            raise ValueError('external-absent: no input executed; cannot accept case evidence')
        if any(case.get(key) != value for key, value in assessment.items()):
            raise ValueError('accepted input scope/grade/completeness differs from actual successful reader receipts')
    claims = receipt_claims(cases, receipt['accepted'])
    if any(receipt.get(key) != value for key, value in claims.items()):
        raise ValueError('aggregate physical completion cannot upgrade input acceptance scope or grade')
    return {'files': len(found), 'bytes': total, 'aggregate_cap_bytes': aggregate_cap, 'source_revision': expected,
            'accepted_model_run': receipt['accepted'], 'accepted_case_execution': receipt['accepted'], **claims}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=LANES)
    requests = parser.add_mutually_exclusive_group(required=True)
    requests.add_argument('--request', help='JSON with exact selection and explicit supervisor policy')
    requests.add_argument('--request-env', help='fixed environment variable carrying JSON, never shell interpolation')
    requests.add_argument('--binding-env', help='existing workflow binding envelope with fixed backend and request')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--verify-artifacts', action='store_true')
    parser.add_argument('--registry', type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    try:
        for name in (args.request_env, args.binding_env):
            if name is not None and not re.fullmatch(r'[A-Z][A-Z0-9_]*', name):
                raise ValueError('invalid request environment variable name')
        if args.binding_env is not None:
            if args.backend is not None:
                raise ValueError('bridge backend comes exclusively from the validated envelope')
            args.backend, value = binding(os.environ.get(args.binding_env, ''))
        else:
            if args.backend is None:
                raise ValueError('explicit fixed backend is required')
            raw = args.request if args.request_env is None else os.environ.get(args.request_env, '')
            value = request(raw, args.backend)
        rows = select(load_registry(args.registry), value['selection'], args.backend, value.get('configuration'))
        multi_cuda.bind_rows(rows, value, args.backend)
        if args.verify_artifacts:
            if args.output is None:
                raise ValueError('owned evidence output is required')
            print(json.dumps(verify_artifacts(args.output, Path(os.environ['RUNNER_TEMP']), value, args.expected_sha, rows, args.backend)))
            return 0
        if args.plan:
            for row in rows:
                validate_contract(row, required=True)
            repository_identity(args.expected_sha)
            sys.path.insert(0, str(ROOT))
            from scripts.media_campaign_supervisor import load_policy
            with tempfile.TemporaryDirectory(prefix='model-proof-policy-') as directory:
                path = Path(directory) / 'policy.json'
                write_json(path, value['policy'])
                load_policy(path)
                for role, secondary in enumerate(value.get('additional_cuda_policies', []), 1):
                    extra_path = Path(directory) / f'cuda-policy-role-{role}.json'
                    write_json(extra_path, secondary)
                    load_policy(extra_path)
            runner = ['self-hosted', 'macOS', 'ARM64', 'rw-llm'] if value.get('runner') == 'nax' else RUNNERS[args.backend]
            iris_route = ((args.backend == 'mlx' and value.get('runner') == 'nax')
                          or args.backend == 'candle-cuda')
            asset_setup = ('iris-quant' if iris_route and len(rows) == 1
                           and identity(rows[0]) in IRIS_QUANT_PROOFS else 'none')
            if (args.backend == 'candle-cuda' and len(rows) == 1
                    and rows[0]['package'] == 'candle-gen-krea'
                    and rows[0]['fqn'] == 'trainer_e2e::krea_bounded_acceleration_profile'):
                asset_setup = 'krea-training'
            plan = {'runner': runner, 'asset_setup': asset_setup, 'backend': args.backend, 'request': value, 'selected': len(rows),
                    'bindings': [identity(row) for row in rows]}
            configured = {identity(row): row['selected_configuration'] for row in rows if 'selected_configuration' in row}
            if configured:
                plan['configurations'] = configured
            print(json.dumps(plan, indent=2))
            return 0
        if args.output is None:
            raise ValueError('an owned workflow output directory is required')
        return run(rows, args.backend, value, args.expected_sha, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('model proof rejected: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
