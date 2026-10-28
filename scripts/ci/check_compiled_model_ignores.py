#!/usr/bin/env python3
"""Before ordinary test execution, prove compiled registered model cases are actually ignored.

Compile using the original Cargo configuration, list libtest metadata only, then run the
unchanged ordinary command. No model body runs during the check; cfg-absent is never proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'scripts/ci/model-inference-proofs.json'
LIBRARY_KINDS = {'lib', 'rlib', 'staticlib', 'cdylib', 'dylib', 'proc-macro'}


def cargo_boundary(command: list[str]) -> int:
    if command[:1] == ['cargo']:
        boundary = 0
    elif (len(command) >= 5 and command[1].replace('\\', '/').endswith('scripts/ci/run_ci_collection.py')
          and command[2:4] == ['--', 'cargo']):
        boundary = 3
    else:
        raise ValueError('compiled-ignore check accepts only an existing Cargo test invocation')
    index = boundary + 1
    while index < len(command):
        argument = command[index]
        if argument == 'test':
            return boundary
        if argument in ('--config', '--color', '-Z'):
            index += 2
        elif argument.startswith(('+', '--config=', '--color=')) or argument in ('--offline', '--locked', '--frozen', '--verbose', '-v', '-vv', '--quiet', '-q'):
            index += 1
        else:
            break
    raise ValueError('compiled-ignore check requires the Cargo test subcommand')


def listed(text: str) -> set[str]:
    return set(re.findall(r'^(.+): test$', text, re.M))


def verify_lists(rows: list[dict], all_text: str, ignored_text: str) -> dict:
    all_cases, ignored = listed(all_text), listed(ignored_text)
    if not ignored <= all_cases:
        raise ValueError('ignored discovery is not a subset of all compiled tests')
    registered = {row['fqn'] for row in rows}
    visible = registered & all_cases
    escaped = visible - ignored
    if escaped:
        raise ValueError('registered model tests are compiled but NOT ignored: ' + ', '.join(sorted(escaped)))
    return {'listed_registered_ignored': len(visible), 'cfg_absent_not_runtime_proof': len(registered - all_cases),
            'model_bodies_executed': 0}


def compile_artifacts(command: list[str]) -> tuple[int, list[dict]]:
    artifacts = []
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    for line in child.stdout:
        try:
            message = json.loads(line)
        except ValueError:
            print(line, end='', flush=True)
            continue
        if message.get('reason') == 'compiler-artifact' and message.get('profile', {}).get('test') and message.get('executable'):
            artifacts.append(message)
        elif message.get('reason') == 'compiler-message':
            rendered = message.get('message', {}).get('rendered')
            if rendered:
                print(rendered, end='', file=sys.stderr, flush=True)
    return child.wait(), artifacts


def metadata(cargo: list[str]) -> dict:
    args = ['cargo', 'metadata', '--no-deps', '--locked', '--offline', '--format-version', '1']
    if '--manifest-path' in cargo:
        at = cargo.index('--manifest-path')
        args += cargo[at:at + 2]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise ValueError('Cargo target metadata failed: ' + result.stderr[-3000:])
    return json.loads(result.stdout)


def check_artifacts(artifacts: list[dict], rows: list[dict], target_metadata: dict) -> list[dict]:
    package_names = {package['id']: package['name'] for package in target_metadata['packages']}
    reports = []
    for artifact in artifacts:
        package = package_names.get(artifact['package_id'])
        if package is None:
            raise ValueError('compiled target has no exact Cargo package identity')
        target = artifact['target']
        library = bool(set(target['kind']) & LIBRARY_KINDS)
        bound = [row for row in rows if row['package'] == package and row['target'] == target['name']
                 and ((row['target_kind'] == 'lib') if library else row['target_kind'] in target['kind'])]
        if not bound:
            continue  # Never execute unknown/custom harnesses with a guessed --list protocol.
        executable = Path(artifact['executable'])
        if executable.is_symlink() or not executable.is_file():
            raise ValueError('compiled model test executable must be a regular Cargo artifact')
        texts = []
        for flags in (['--list'], ['--list', '--ignored']):
            result = subprocess.run([str(executable), *flags], cwd=ROOT, capture_output=True,
                                    text=True, encoding='utf-8', timeout=60)
            if result.returncode:
                raise ValueError('metadata-only libtest listing failed for ' + package + '/' + target['name'])
            texts.append(result.stdout)
        report = verify_lists(bound, *texts)
        reports.append(dict(report, package=package, target=target['name'],
                            cargo_target_kind=target['kind'], features=artifact.get('features', [])))
    return reports


def run(command: list[str], *, list_only: bool = False) -> int:
    boundary = cargo_boundary(command)
    cargo = command[boundary:]
    if list_only and '--doc' in cargo:
        raise ValueError('doctest listing is governed separately; list-only never executes doc bodies')
    if '--no-run' in cargo or '--doc' in cargo:
        # Cross-target compilation and compile/no_run docs are not runnable libtest targets.
        # Runnable Rustdoc is separately covered by check_model_inference_policy.py.
        return subprocess.call(command, cwd=ROOT)
    if any(flag in cargo for flag in ('--ignored', '--include-ignored')):
        raise ValueError('ordinary CI cannot override model-test ignores')
    separator = command.index('--', boundary) if '--' in command[boundary:] else len(command)
    compilation = command[:separator] + ['--no-run', '--message-format=json']
    code, artifacts = compile_artifacts(compilation)
    if code:
        return code
    rows = json.loads(REGISTRY.read_text(encoding='utf-8'))
    reports = check_artifacts(artifacts, rows, metadata(cargo))
    print('COMPILED_MODEL_IGNORE_POLICY ' + json.dumps({'targets': reports, 'model_bodies_executed': 0}, sort_keys=True), flush=True)
    return 0 if list_only else subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-only', action='store_true', help='compile/list only; never execute ordinary tests either')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return run(command, list_only=args.list_only)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('Compiled model-ignore policy rejected: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
