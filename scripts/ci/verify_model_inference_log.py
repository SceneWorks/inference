#!/usr/bin/env python3
"""Check one real-weights gate log against its registered model-inference proof contract.

A real-weights wrapper that runs a registered case itself (rather than through the
`model-inference-proofs` profile) sets `MODEL_INFERENCE_CASE_FQN`, builds the case's registered
features, and passes its combined stdout/stderr here. The check is the proof runner's own
`verify_execution`: exactly one successful libtest summary, exactly one BODY completion and an
actual in-scope device set that matches the case (or configuration) declaration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proofs  # noqa: E402


def registered_case(rows: tuple[dict, ...], package: str, fqn: str, configuration: str | None) -> dict:
    matches = [row for row in rows if row['package'] == package and row['fqn'] == fqn]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one registered case {package}/{fqn}, found {len(matches)}')
    return proofs.configured_case(matches[0], configuration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--fqn', required=True)
    parser.add_argument('--configuration')
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=proofs.REGISTRY_PATH)
    args = parser.parse_args()
    try:
        row = registered_case(proofs.load_registry(args.registry), args.package, args.fqn, args.configuration)
        text = args.log.read_text(encoding='utf-8', errors='replace')
        print(json.dumps({'binding': proofs.identity(row), **proofs.verify_execution(row, text)}))
        return 0
    except (ValueError, OSError) as error:
        print(f'model-inference evidence rejected for {args.package}/{args.fqn}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
