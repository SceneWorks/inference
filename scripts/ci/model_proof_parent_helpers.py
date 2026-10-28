"""Structural parent-only Iris helper bindings; no runtime protocol or training-success claim."""
from __future__ import annotations

IRIS_HELPER = {
    'package': 'candle-gen-iris', 'target': 'integration', 'target_kind': 'test',
    'fqn': 'train_distributed::rank_entry',
    'source': 'crates/media/candle-gen/candle-gen-iris/tests/train_distributed.rs',
}


def identity(row: dict) -> str:
    return row['package'] + '/' + row['target'] + '/' + row['fqn']


def validate_parent_helpers(rows: list[dict] | tuple[dict, ...]) -> dict[str, tuple[str, ...]]:
    """Every helper is ignored/source-checked separately and exercised only through real parents."""
    result = {}
    for helper in rows:
        if 'parent_only_helper' not in helper and 'parent_bindings' not in helper:
            continue
        if helper.get('parent_only_helper') is not True or any(helper.get(key) != value for key, value in IRIS_HELPER.items()):
            raise ValueError('parent-only relation is limited to the exact source-declared Candle Iris rank helper')
        if helper.get('cargo_target_metadata') != {'name': 'integration', 'kind': ['test'], 'crate_types': ['bin']}:
            raise ValueError('parent-only helper must retain the exact actual Cargo test target metadata')
        if helper.get('source_function', 'rank_entry') != 'rank_entry':
            raise ValueError('parent-only helper cannot alias another source function')
        if helper.get('execution_role') != 'direct' or any(key in helper for key in (
                'body_completion_marker', 'parent_completion_marker', 'expected_successful_summaries', 'configurations')):
            raise ValueError('parent-only helper cannot declare independent runtime completion authority')
        bindings = helper.get('parent_bindings')
        if (not isinstance(bindings, list) or not bindings or any(not isinstance(value, str) for value in bindings)
                or len(set(bindings)) != len(bindings)):
            raise ValueError('parent-only helper requires exact unique parent bindings')
        if sum(identity(row) == identity(helper) for row in rows) != 1:
            raise ValueError('parent-only helper requires one canonical source binding')
        parents = [row for row in rows if all(row.get(key) == helper[key] for key in ('package', 'target', 'target_kind', 'source'))
                   and row.get('execution_role') in ('nested_rank_direct', 'nested_rank_cpu')]
        if set(bindings) != {identity(parent) for parent in parents} or len(bindings) != len(parents):
            raise ValueError('helper links must cover every exact same-target nested-rank parent, without missing/extra cases')
        for parent in parents:
            if (parent.get('parent_only_helper') or parent.get('cargo_target_metadata') != helper.get('cargo_target_metadata')
                    or type(parent.get('expected_successful_summaries')) is not int
                    or parent['expected_successful_summaries'] < (2 if parent['execution_role'] == 'nested_rank_cpu' else 1)):
                raise ValueError('helper parent must retain actual Cargo target and source-declared libtest completion count')
        result[identity(helper)] = tuple(bindings)
    return result
