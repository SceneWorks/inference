#!/usr/bin/env python3
"""Check one real-weights gate log against its registered model-inference proof contract.

A real-weights wrapper that runs a registered case itself (rather than through the
`model-inference-proofs` profile) sets `MODEL_INFERENCE_CASE_FQN`, builds the case's registered
features, and passes its combined stdout/stderr here. The check is the proof runner's own
`verify_execution`: exactly one successful libtest summary, exactly one BODY completion and an
actual in-scope device set that matches the case (or configuration) declaration.

`--backend` selects the one configuration a multi-configuration case declares for that backend.
`--no-learned-work` is for a registered gate whose body runs no learned tensor (an index or
tokenizer check): it keeps the summary, BODY and missing-prerequisite checks and drops only the
device set, which such a body has nothing to observe on. Nothing is synthesized either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proofs  # noqa: E402


def registered_case(rows: tuple[dict, ...], package: str, fqn: str, configuration: str | None,
                    backend: str | None = None) -> dict:
    matches = [row for row in rows if row['package'] == package and row['fqn'] == fqn]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one registered case {package}/{fqn}, found {len(matches)}')
    row = matches[0]
    if backend is not None and configuration is None and 'configurations' in row:
        names = [name for name, recipe in row['configurations'].items() if recipe['backend'] == backend]
        if len(names) != 1:
            raise ValueError(f'expected exactly one {backend} configuration, found {names}')
        configuration = names[0]
    case = proofs.configured_case(row, configuration)
    if backend is not None and case.get('backend') != backend:
        raise ValueError(f'registered backend {case.get("backend")!r} is not {backend!r}')
    return case


def verify_without_learned_work(row: dict, text: str) -> dict:
    """`verify_execution` minus the device set, for a direct body that runs no learned tensor."""
    if row['execution_role'] != 'direct':
        raise ValueError('only a direct case can be checked without learned work')
    fqn = re.escape(row['fqn'])
    if re.search(r'^MODEL_INFERENCE_PREREQUISITE_MISSING ' + fqn + r'(?:[ \t]|$)', text, re.M):
        raise ValueError('required work was skipped; no partial/zero-body proof')
    if re.search(r'^test result: (?!ok\.)', text, re.M):
        raise ValueError('the test failed')
    results = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text, re.M)
    if results != [('1', '0', '0')]:
        raise ValueError('expected exactly one successful summary of one test: ' + repr(results))
    if text.splitlines().count('MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn']) != 1:
        raise ValueError('missing/duplicate full body completion')
    return {'successful_libtest_summaries': 1, 'body_completed': 1, 'actual_devices': None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--fqn', required=True)
    parser.add_argument('--configuration')
    parser.add_argument('--backend')
    parser.add_argument('--no-learned-work', action='store_true')
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=proofs.REGISTRY_PATH)
    args = parser.parse_args()
    try:
        row = registered_case(proofs.load_registry(args.registry), args.package, args.fqn,
                              args.configuration, args.backend)
        text = args.log.read_text(encoding='utf-8', errors='replace')
        verify = verify_without_learned_work if args.no_learned_work else proofs.verify_execution
        print(json.dumps({'binding': proofs.identity(row), **verify(row, text)}))
        return 0
    except (ValueError, OSError) as error:
        print(f'model-inference evidence rejected for {args.package}/{args.fqn}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
