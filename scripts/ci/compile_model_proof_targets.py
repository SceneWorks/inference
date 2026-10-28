#!/usr/bin/env python3
"""Compile/lint selected proof hooks and exact source recipe feature mechanisms.

Never execute tests/models. Keep the original all-targets base pass (including
CUDA without FlashAttention), then deduplicate auxiliary graphs, not cases.
The compile host is distinct from the actual NN device: a CPU-body recipe may
require a CUDA build host. Cargo declarations validate features, not permission
to enable every feature. Only selected registry rows supply auxiliary graphs.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.model_inference_proofs import (  # noqa: E402
    configuration_manifest_features,
    configuration_route,
    features as declared_features,
    lane as proof_lane,
    validate_configurations,
)

FEATURE = 'model-inference-proof'
REGISTRY = Path(__file__).with_name('model-inference-proofs.json')
HOST_LANES = ('cpu', 'metal', 'cuda')


def _bare(feature: str, package: str) -> str:
    return feature.removeprefix(package + '/')


def _options(command: list[str]) -> tuple[int, list[str], set[str]]:
    if command[:2] not in (['cargo', 'check'], ['cargo', 'clippy']):
        raise ValueError('only cargo check/clippy proof-hook compilation is permitted')
    boundary = command.index('--') if '--' in command else len(command)
    options = command[2:boundary]
    if '--all-targets' not in options:
        raise ValueError('proof-hook coverage must compile all applicable test targets')
    patterns, caller = [], set()
    index = 0
    while index < len(options):
        item = options[index]
        if item in ('-p', '--package', '--features', '-F'):
            index += 1
            if index == len(options) or options[index].startswith('-'):
                raise ValueError('missing package selector or feature value')
            if item in ('-p', '--package'):
                patterns.append(options[index])
            else:
                caller.update(options[index].replace(',', ' ').split())
        elif item.startswith('--package='):
            patterns.append(item.split('=', 1)[1])
        elif item.startswith('-p') and len(item) > 2:
            patterns.append(item[2:])
        elif item.startswith('--features='):
            caller.update(item.split('=', 1)[1].replace(',', ' ').split())
        elif item.startswith('-F') and len(item) > 2:
            caller.update(item[2:].replace(',', ' ').split())
        index += 1
    if not patterns or {'--workspace', '--all', '--all-features'}.intersection(options):
        raise ValueError('proof hooks require explicit existing package selection, never all-features')
    return boundary, patterns, caller


def _selected(patterns: list[str], metadata: dict) -> dict:
    workspace = set(metadata['workspace_members'])
    packages = {p['name']: p for p in metadata['packages'] if p['id'] in workspace}
    selected = {}
    for pattern in patterns:
        matches = {name: p for name, p in packages.items() if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError('package selector has no workspace matches: ' + pattern)
        selected.update(matches)
    return selected


def proof_command(command: list[str], metadata: dict) -> list[str]:
    """The original base pass, independent of registry population/runnable defaults."""
    boundary, patterns, _ = _options(command)
    selected = _selected(patterns, metadata)
    hooks = sorted(name + '/' + FEATURE for name, p in selected.items() if FEATURE in p['features'])
    if not hooks:
        raise ValueError('selected source graph declares no model-inference-proof hooks')
    return command[:boundary] + ['--features', ','.join(hooks)] + command[boundary:]


def _feature_markers(package: str, feature: str, packages: dict, seen=None) -> set[str]:
    """Follow actual Cargo feature forwarding; empty definitions prove no CPU affinity."""
    seen = set() if seen is None else seen
    key = (package, feature)
    if key in seen:
        return set()
    seen.add(key)
    if '/' in feature:
        owner, feature = feature.split('/', 1)
        return _feature_markers(owner.rstrip('?'), feature, packages, seen)
    markers = {feature} if feature in ('cuda', 'metal') else set()
    for forwarded in packages.get(package, {}).get('features', {}).get(feature, []):
        if not forwarded.startswith('dep:'):
            markers.update(_feature_markers(package, forwarded, packages, seen))
    return markers


def _manifest(row: dict, package: dict, root: Path) -> set[str]:
    """Reuse runner path/name semantics; additionally bind to actual Cargo ownership."""
    source = row.get('source')
    if (not isinstance(source, str) or Path(source).is_absolute() or '..' in Path(source).parts
            or not (root / source).is_file()
            or not (root / source).resolve().is_relative_to(root.resolve())):
        raise ValueError('selected proof source is missing or escapes source checkout')
    declared = configuration_manifest_features(row, root)
    if row.get('configuration_manifest') is not None:
        manifest = root / row['configuration_manifest']
    else:
        directory = (root / source).parent
        # The imported validator already establishes a matching source owner.
        while True:
            manifest = directory / 'Cargo.toml'
            if (manifest.is_file() and
                    tomllib.loads(manifest.read_text(encoding="utf-8")).get('package', {}).get('name') == row['package']):
                break
            if directory == root:
                raise ValueError('selected proof has no source Cargo manifest')
            directory = directory.parent
    actual = Path(package['manifest_path'])
    if actual.resolve() != manifest.resolve() or not actual.resolve().is_relative_to(root.resolve()):
        raise ValueError('configuration/source manifest differs from actual Cargo metadata ownership: '
                         + row['package'])
    if declared != package['features']:
        raise ValueError('source feature declarations differ from actual Cargo metadata: ' + row['package'])
    return set(declared)


def _row_graphs(row: dict, root: Path, package: dict) -> list[tuple[str, frozenset[str]]]:
    """Root AND every fixed configuration; a null default supplies no root graph."""
    if 'execution_lane' in row or 'selected_configuration' in row:
        raise ValueError('execution_lane is recipe-only, not root-row authority')
    declared = _manifest(row, package, root)
    validate_configurations(row, root=root)
    graphs = []
    base = frozenset(declared_features(row))
    if base:
        graphs.append((proof_lane(row), base))
    for recipe in (row.get('configurations') or {}).values():
        graphs.append((configuration_route(row, recipe), frozenset(declared_features(recipe))))
    for _, graph in graphs:
        for feature in graph:
            if _bare(feature, row['package']) not in declared:
                raise ValueError('proof graph feature is absent from actual selected source manifest: '
                                 + row['package'] + ' -> ' + feature)
    return graphs


def _mechanisms(rows: tuple[dict, ...], selected: dict, metadata: dict, *,
                host_lane: str, root: Path, no_defaults: bool) -> list[list[str]]:
    packages = {p['name']: p for p in metadata['packages']}
    table = {}
    for row in rows:
        # Separate workspaces (e.g. ios-smoke) are not this invocation's authority.
        if not isinstance(row, dict) or not isinstance(row.get('package'), str):
            raise ValueError('registry rows must identify a Cargo package')
        package = row['package']
        if package not in selected:
            continue
        for recipe_lane, graph in _row_graphs(row, root, selected[package]):
            markers = set().union(*(_feature_markers(package, f, packages) for f in graph))
            expected_host = 'metal' if recipe_lane == 'mlx' else recipe_lane.removeprefix('candle-')
            if markers - {expected_host}:
                raise ValueError('source recipe features contradict its compile host: '
                                 + package + ' -> ' + ','.join(sorted(graph)))
            if expected_host != host_lane:
                continue
            if row.get('no_default_features', False) and not no_defaults:
                raise ValueError('selected recipe requires --no-default-features: ' + package)
            # Lane authority is the exact recipe, never guessed from feature names.
            mechanism = tuple(sorted(_bare(f, package) for f in graph))
            table.setdefault(mechanism, set()).update(package + '/' + f for f in mechanism)
    return [sorted(table[key]) for key in sorted(table)]


def proof_commands(command: list[str], metadata: dict, rows: tuple[dict, ...], *,
                   host_lane: str, root: Path = ROOT) -> list[list[str]]:
    boundary, patterns, caller = _options(command)
    selected = _selected(patterns, metadata)
    if host_lane not in HOST_LANES:
        raise ValueError('unknown compile-host lane: ' + str(host_lane))
    packages = {p['name']: p for p in metadata['packages']}
    markers = set()
    for package in selected:
        for feature in caller:
            markers.update(_feature_markers(package, feature, packages))
        if '--no-default-features' not in command[2:boundary]:
            markers.update(_feature_markers(package, 'default', packages))
    if markers - {host_lane}:
        raise ValueError('caller/default feature lane differs from declared compile-host lane')
    base = proof_command(command, metadata)
    base_features = set(base[boundary + 1].split(','))
    for feature in base_features:
        markers.update(_feature_markers('', feature, packages))
    if markers - {host_lane}:
        raise ValueError('base proof hooks contradict declared compile-host lane')
    caller_graph = set()
    for feature in caller:
        if '/' in feature:
            caller_graph.add(feature)
        else:
            caller_graph.update(p + '/' + feature for p, data in selected.items()
                                if feature in data['features'])
    commands = [base]
    seen = {frozenset(base_features | caller_graph)}
    for graph in _mechanisms(rows, selected, metadata, host_lane=host_lane, root=root,
                             no_defaults='--no-default-features' in command[2:boundary]):
        merged = base_features | set(graph)
        signature = frozenset(merged | caller_graph)
        if signature in seen:
            continue
        seen.add(signature)
        commands.append(command[:boundary] + ['--features', ','.join(sorted(merged))] + command[boundary:])
    return commands


def _registry(path: Path) -> tuple[dict, ...]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('model-proof registry must be a nonempty array')
    if any(not isinstance(row, dict) or not isinstance(row.get('package'), str) for row in rows):
        raise ValueError('registry rows must identify a Cargo package')
    return tuple(rows)


def _metadata_command(command: list[str], root: Path) -> list[str]:
    """Honor an existing caller's separate workspace; never guess a consumer root."""
    boundary, _, _ = _options(command)
    options = command[2:boundary]
    result = ['cargo', 'metadata', '--format-version=1', '--no-deps', '--locked']
    for index, option in enumerate(options):
        if option == '--manifest-path':
            if index + 1 == len(options):
                raise ValueError('missing Cargo manifest path')
            value = options[index + 1]
        elif option.startswith('--manifest-path='):
            value = option.split('=', 1)[1]
        else:
            continue
        manifest = root / value
        if (manifest.name != 'Cargo.toml' or '..' in Path(value).parts or not manifest.is_file()
                or not manifest.resolve().is_relative_to(root.resolve())):
            raise ValueError('caller manifest must be an actual Cargo.toml inside source checkout')
        result += ['--manifest-path', value]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', default=str(REGISTRY), help='canonical model-proof registry')
    parser.add_argument('--lane', choices=HOST_LANES, help='compile host, not actual NN device')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        _, _, caller = _options(command)
        rows = _registry(Path(args.registry))
        metadata_command = _metadata_command(command, ROOT)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    host_lane = args.lane
    if host_lane is None:
        markers = {f.split('/')[-1] for f in caller}.intersection({'cuda', 'metal'})
        if len(markers) > 1:
            parser.error('one proof graph cannot mix CUDA and Metal compile lanes')
        host_lane = next(iter(markers), 'cpu')
    metadata = subprocess.run(metadata_command, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    try:
        commands = proof_commands(command, json.loads(metadata.stdout), rows,
                                  host_lane=host_lane, root=ROOT)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result = 0
    for enriched in commands:
        print('Compile-only proof hooks: ' + shlex.join(enriched), flush=True)
        returned = subprocess.run(enriched, cwd=ROOT)
        if returned.returncode != 0 and result == 0:
            result = returned.returncode if returned.returncode > 1 else 1
    return result


if __name__ == '__main__':
    raise SystemExit(main())
