#!/usr/bin/env python3
"""CI-only structural policy: model tests compile normally but execute only as explicit proofs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ci.model_proof_parent_helpers import identity as helper_identity, validate_parent_helpers

CANONICAL = 'model inference; explicit model-proof workflow'
RAW_STRING = re.compile(r'(?:b|c)?r(#{0,255})"')
CHARACTER = re.compile(r"'(?:\\.|[^'\\\n])'")
MODIFIER = re.compile(r'(?:pub(?:\([^()]*\))?|async|unsafe)\s*$')


def rust_mask(text: str) -> str:
    """Mask comments/strings without changing offsets; do not mistake quoted Rust for code."""
    chars = list(text)
    index = 0
    while index < len(text):
        start = index
        if text.startswith('//', index):
            index = text.find('\n', index)
            if index < 0:
                index = len(text)
        elif text.startswith('/*', index):
            index += 2
            depth = 1
            while index < len(text) and depth:
                if text.startswith('/*', index):
                    depth += 1
                    index += 2
                elif text.startswith('*/', index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
        else:
            raw = RAW_STRING.match(text, index)
            if raw:
                close = '"' + raw[1]
                end = text.find(close, index + len(raw[0]))
                index = len(text) if end < 0 else end + len(close)
            elif text[index] == '"':
                index += 1
                while index < len(text):
                    if text[index] == '\\':
                        index += 2
                    elif text[index] == '"':
                        index += 1
                        break
                    else:
                        index += 1
            elif text[index] == "'" and CHARACTER.match(text, index):
                index += len(CHARACTER.match(text, index)[0])
            else:
                index += 1
                continue
        for position in range(start, min(index, len(chars))):
            if chars[position] != '\n':
                chars[position] = ' '
    return ''.join(chars)


def matching_brace(mask: str, opening: int) -> int:
    depth = 1
    position = opening + 1
    while position < len(mask) and depth:
        depth += (mask[position] == '{') - (mask[position] == '}')
        position += 1
    if depth:
        raise ValueError('unbalanced Rust function body')
    return position


def attributes_before(text: str, mask: str, position: int) -> list[str]:
    attributes = []
    while position > 0:
        while position > 0 and mask[position - 1].isspace():
            position -= 1
        offset = max(0, position - 160)
        modifier = MODIFIER.search(mask[offset:position])
        if modifier:
            position = offset + modifier.start()
            continue
        if position == 0 or mask[position - 1] != ']':
            break
        end = position
        depth = 1
        position -= 1
        while position > 0 and depth:
            position -= 1
            depth += (mask[position] == ']') - (mask[position] == '[')
        if depth or position == 0 or mask[position - 1] != '#':
            break
        position -= 1
        attributes.append(text[position:end])
    return attributes


def functions(text: str, *, source: str | None = None) -> list[dict]:
    mask = rust_mask(text)
    inline_modules = []
    for module in re.finditer(r'\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{', mask):
        opening = mask.find('{', module.start())
        inline_modules.append((opening, matching_brace(mask, opening), module[1]))
    result = []
    for function in re.finditer(r'\bfn\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\(', mask):
        opening = mask.find('{', function.end())
        if opening < 0:
            continue
        end = matching_brace(mask, opening)
        attrs = attributes_before(text, mask, function.start())
        if not any(re.search(r'#\s*\[\s*(?:test|tokio::test|rstest|test_case)\b', attr) for attr in attrs):
            continue
        ignores = [attr for attr in attrs if re.match(r'#\s*\[\s*ignore\b', attr)]
        modules = tuple(name for start, finish, name in inline_modules if start < function.start() < finish)
        result.append({'name': function[1], 'start': function.start(), 'end': end,
                       'line': text.count('\n', 0, function.start()) + 1,
                       'inline_modules': modules, 'attributes': attrs, 'ignored': bool(ignores),
                       'canonical': any(CANONICAL in attr for attr in ignores),
                       'body': text[opening:end]})
    if source is not None:
        result.extend(generated_test_bindings(text, mask, inline_modules, source))
    return result


def generated_test_bindings(text: str, mask: str, modules: list, source: str) -> list[dict]:
    """Expand only the two audited source forms, with exact invocation/ignore/BODY relationships."""
    specs = {
        'crates/media/candle-gen/candle-gen-minimax-h3/tests/turbo_lora.rs': (
            'per_alpha_spelling', 'turbo_lora', '($($name:ident,$chain:ident=>$spelling:expr;)+)=>{',
            'assert_twin_equivalence($spelling);'),
        'crates/media/candle-gen/candle-gen-sd3/tests/trainer_cuda_smoke.rs': (
            'smoke', 'trainer_cuda_smoke', '($name:ident,$id:literal,$env:literal,$kind:expr)=>{',
            'run($id,$env,$kind);'),
    }
    if source not in specs:
        return []
    macro, namespace, matcher, call = specs[source]
    definitions = list(re.finditer(r'\bmacro_rules!\s*' + macro + r'\s*\{', mask))
    if len(definitions) != 1:
        raise ValueError(source + ': exact generated-test macro definition is missing/ambiguous')
    opening = mask.find('{', definitions[0].start())
    end = matching_brace(mask, opening)
    if not re.sub(r'\s+', '', mask[opening + 1:end - 1]).startswith(matcher):
        raise ValueError(source + ': unsupported generated-test macro definition')
    templates = {}
    for function in re.finditer(r'\bfn\s+\$(name|chain)\s*\(\s*\)\s*\{', mask[opening:end]):
        start = opening + function.start()
        brace = mask.find('{', start)
        finish = matching_brace(mask, brace)
        attrs = attributes_before(text, mask, start)
        if function[1] in templates or not any(re.search(r'#\s*\[\s*test\s*\]', attr) for attr in attrs):
            raise ValueError(source + ': generated test template is missing/ambiguous')
        templates[function[1]] = (start, brace, finish, attrs)
    if set(templates) != ({'name', 'chain'} if macro == 'per_alpha_spelling' else {'name'}):
        raise ValueError(source + ': generated test template names differ from exact source form')
    definition_shape = mask[opening + 1:end - 1]
    for template, (start, _, finish, _) in sorted(templates.items(), key=lambda item: item[1][0], reverse=True):
        definition_shape = (definition_shape[:start - opening - 1] + '__' + template.upper() + '__'
                            + definition_shape[finish - opening - 1:])
    expected_shape = (matcher + 'constGENERATED_SPELLINGS:&[AlphaSpelling]=&[$($spelling),+];'
                      + '$(#[test]#[ignore=]__NAME__#[test]__CHAIN__)+};' if macro == 'per_alpha_spelling'
                      else matcher + '#[test]#[ignore=]__NAME__};')
    if re.sub(r'\s+', '', definition_shape) != expected_shape:
        raise ValueError(source + ': generated macro definition/repetition/attributes changed')
    nn_start, nn_brace, nn_end, nn_attrs = templates['name']
    nn_body = text[nn_brace:nn_end]
    nn_mask = mask[nn_brace:nn_end]
    select = re.search(r'match\s+stringify!\(\$name\)\s*\{', nn_mask)
    if select is None:
        raise ValueError(source + ': generated BODY dispatch is missing')
    prefix = re.sub(r'\s+', '', nn_body[1:select.start()])
    if prefix != call + '#[cfg(feature="model-inference-proof")]':
        raise ValueError(source + ': generated NN call/late proof-feature dispatch changed')
    dispatch_open = nn_mask.find('{', select.start())
    dispatch_end = matching_brace(nn_mask, dispatch_open)
    if re.sub(r'\s+', '', nn_body[dispatch_end:]) != ';}':
        raise ValueError(source + ': generated completion is not the exact late source dispatch')
    dispatch = nn_body[dispatch_open + 1:dispatch_end - 1]
    arm_pattern = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*=>\s*println!\(\s*"\\nMODEL_INFERENCE_BODY_COMPLETED ([A-Za-z_][A-Za-z0-9_:]*)"\s*\)\s*,')
    arms = {}
    remainder = list(dispatch)
    for arm in arm_pattern.finditer(dispatch):
        if (arm[1] in arms or arm[2] != namespace + '::' + arm[1]
                or re.sub(r'\s+', '', rust_mask(arm[0])) != '=>println!(),'):
            raise ValueError(source + ': generated BODY arm is duplicate/wrong-name/not actual code')
        arms[arm[1]] = arm[2]
        remainder[arm.start():arm.end()] = ' ' * (arm.end() - arm.start())
    residual = ''.join(remainder)
    if (re.sub(r'\s+', '', rust_mask(residual)) != '_=>panic!(),'
            or not re.search(r'panic!\(\s*"undeclared generated model-proof case"\s*\)', residual)):
        raise ValueError(source + ': generated BODY dispatch contains unsupported/unchecked arms')
    generated = []
    for invocation in re.finditer(r'\b' + macro + r'\s*!\s*([({])', mask):
        delimiter = invocation[1]
        if delimiter != ('{' if macro == 'per_alpha_spelling' else '('):
            raise ValueError(source + ': unsupported generated-test invocation delimiter')
        begin = invocation.end() - 1
        if delimiter == '{':
            finish = matching_brace(mask, begin)
        else:
            depth, finish = 1, begin + 1
            while finish < len(mask) and depth:
                depth += (mask[finish] == '(') - (mask[finish] == ')')
                finish += 1
            if depth:
                raise ValueError(source + ': unbalanced generated-test invocation')
        arguments = text[begin + 1:finish - 1]
        if macro == 'per_alpha_spelling':
            pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*=>\s*(AlphaSpelling::[A-Za-z_][A-Za-z0-9_]*)\s*;')
            matches = list(pattern.finditer(arguments))
            rest = list(arguments)
            for item in matches:
                rest[item.start():item.end()] = ' ' * (item.end() - item.start())
                generated.extend(((item[1], 'name', {'$spelling': item[3]}, invocation.start(), finish),
                                  (item[2], 'chain', {'$spelling': item[3]}, invocation.start(), finish)))
            if not matches or rust_mask(''.join(rest)).strip():
                raise ValueError(source + ': unsupported alpha-spelling invocation arguments')
        else:
            item = re.fullmatch(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*("[^"\\]*")\s*,\s*("[^"\\]*")\s*,\s*(NetworkType::[A-Za-z_][A-Za-z0-9_]*)\s*,?\s*', arguments)
            if item is None:
                raise ValueError(source + ': unsupported trainer-smoke invocation arguments')
            generated.append((item[1], 'name', {'$id': item[2], '$env': item[3], '$kind': item[4]}, invocation.start(), finish))
    names = [name for name, template, _, _, _ in generated if template == 'name']
    if not names or len({item[0] for item in generated}) != len(generated) or set(names) != set(arms):
        raise ValueError(source + ': invocation names and exact late BODY arms do not agree')
    result = []
    for name, template, substitutions, start, finish in generated:
        _, brace, function_end, attrs = templates[template]
        ignored = any(re.match(r'#\s*\[\s*ignore\b', attr) for attr in attrs)
        if template == 'chain' and (ignored or re.sub(r'\s+', '', text[brace:function_end]) != '{assert_alpha_resolves_before_the_split($spelling);}'):
            raise ValueError(source + ': ordinary alpha-metadata generated arm changed')
        body = text[brace:function_end]
        for old, new in {'$name': name, '$chain': name, **substitutions}.items():
            body = body.replace(old, new)
        result.append({'name': name, 'start': start, 'end': finish, 'line': text.count('\n', 0, start) + 1,
                       'inline_modules': tuple(module for begin, stop, module in modules if begin < start < stop),
                       'attributes': attrs, 'ignored': ignored, 'canonical': any(CANONICAL in attr for attr in attrs),
                       'body': body, 'generated_by': macro, 'generated_fqn': arms.get(name)})
    return result


def parent_helper_source_errors(text: str, candidates: list[dict]) -> list[str]:
    """The sole source-bound Iris helper keeps its real child guard and same-executable spawn."""
    errors = []
    mask = rust_mask(text)
    constant = re.search(r'\bconst\s+CHILD_ENV\s*:\s*&str\s*=', mask)
    if constant is None or re.match(r'"IRIS_DIST_CHILD"\s*;', text[constant.end():].lstrip()) is None:
        errors.append('parent-only helper must retain exact IRIS_DIST_CHILD source selector')
    for candidate in candidates:
        body = rust_mask(candidate['body'])
        if (not re.search(r'let\s+Ok\s*\(raw\)\s*=\s*std::env::var\(CHILD_ENV\)\s*else\s*\{\s*return\s*;\s*\}', body)
                or 'run_job(&j)' not in re.sub(r'\s+', '', body)):
            errors.append('parent-only helper must retain its child-only guard and actual run_job path')
        if not any('the rank-process entry of train_distributed (driven by the scenarios)' in attr for attr in candidate['attributes']):
            errors.append('parent-only helper must preserve its original ignore reason')
    spawn = re.search(r'\bfn\s+spawn\s*\([^)]*\)[^{]*\{', mask)
    if spawn is None:
        errors.append('parent-only helper has no exact source spawn seam')
        return errors
    opening = mask.find('{', spawn.start())
    finish = matching_brace(mask, opening)
    compact = re.sub(r'\s+', '', mask[opening:finish])
    source = text[opening:finish]
    args = re.search(r'\.args\(\s*\[(.*?)\]\s*\)', source, re.S)
    if ('Command::new(std::env::current_exe().unwrap())' not in compact or '.env(CHILD_ENV,j.to_string())' not in compact
            or args is None or re.findall(r'"([^"\\]*)"', args[1]) != [
                '--exact', 'train_distributed::rank_entry', '--ignored', '--nocapture', '--test-threads', '1']):
        errors.append('parent-only helper must retain exact same-executable/ignored child invocation')
    return errors


def registry_errors(root: Path, rows: list[dict], legacy: list[dict] = ()) -> list[str]:
    errors = []
    try:
        helpers = validate_parent_helpers(rows)
    except ValueError as error:
        errors.append(str(error))
        helpers = {}
    by_source = {}
    legacy_bindings = {(r['package'], r['target'], r['fqn'], r['source']) for r in legacy}
    for row in rows:
        by_source.setdefault(row['source'], []).append(row)
    cache = {}
    for relative, bound in by_source.items():
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(relative + ': missing/escaping source')
            continue
        text = path.read_text(encoding='utf-8')
        try:
            parsed = functions(text, source=relative)
        except ValueError as error:
            errors.append(str(error))
            parsed = functions(text)
        cache[relative] = parsed
        for row in bound:
            name = row.get('source_function', row['fqn'].split('::')[-1])
            parts = row['fqn'].split('::')
            candidates = [item for item in parsed if item['name'] == name
                          and (not item['inline_modules'] or tuple(parts[-len(item['inline_modules']) - 1:-1]) == item['inline_modules'])]
            if not candidates:
                errors.append(row['fqn'] + ': registered source test function is missing')
                continue
            old = (row['package'], row.get('legacy_registry_target', row['target']), row['fqn'], relative) in legacy_bindings
            # Existing ignore reasons are prerequisites, not an execution leak. Preserve them.
            # Registration + actual ignore + late completion/device protocols establish routing.
            if not all(item['ignored'] for item in candidates):
                errors.append(row['fqn'] + ': registered model test must be unconditionally ignored')
            marker = row.get('body_completion_marker', 'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'])
            if any(item.get('generated_by') and (item.get('generated_fqn') != row['fqn']
                       or marker != 'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn']) for item in candidates):
                errors.append(row['fqn'] + ': generated test requires its exact source invocation/BODY binding')
            if helper_identity(row) in helpers:
                errors.extend(parent_helper_source_errors(text, candidates))
                if marker in text:
                    errors.append(row['fqn'] + ': parent-only helper cannot fabricate a standalone BODY')
                continue
            if marker.startswith('IOS_CAPTURE_BODY_COMPLETED ') and not old:
                errors.append(row['fqn'] + ': legacy protocol is limited to its unchanged registered source binding')
            if marker not in text:
                errors.append(row['fqn'] + ': actual body completion marker is missing')
    # Coverage binds tagged source functions, not a frozen population/count. New cases classify
    # and register; removing an entry cannot silently leave a newly manual test unreachable.
    sources = set(root.glob('crates/**/*.rs')) | set(root.glob('ios-host/**/*.rs'))
    for path in sorted(sources):
        relative = path.relative_to(root).as_posix()
        parsed = cache.get(relative)
        if parsed is None:
            text = path.read_text(encoding='utf-8')
            try:
                parsed = functions(text, source=relative)
            except ValueError as error:
                errors.append(str(error))
                parsed = functions(text)
        for marker in set(re.findall(r'MODEL_INFERENCE_BODY_COMPLETED ([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)', path.read_text(encoding='utf-8'))):
            if not any(row['fqn'] == marker for row in by_source.get(relative, [])):
                errors.append(f'{relative}: model body marker has no executable registry binding: {marker}')
        for item in parsed:
            if item['canonical'] and not any(row.get('source_function', row['fqn'].split('::')[-1]) == item['name']
                                             and (not item['inline_modules'] or tuple(row['fqn'].split('::')[-len(item['inline_modules']) - 1:-1]) == item['inline_modules'])
                                             for row in by_source.get(relative, [])):
                errors.append(f'{relative}:{item["line"]}: tagged model test is not registered: {item["name"]}')
    return errors


def model_doc_errors(text: str, source: str) -> list[str]:
    """Check runnable Rustdoc, including implicit fence end at the end of a doc fragment.

    This structural seam detector supplements the source callgraph audit; it does not claim
    a Rust parser can infer arbitrary user wrappers or distinguish every tokenizer alias.
    Raw functional kernels remain permitted. no_run/compile_fail compile without execution.
    """
    errors = []
    learned = re.compile(r'\.(?:forward(?:_[A-Za-z0-9_]+)?|generate(?:_[A-Za-z0-9_]+)?|train|backward)\s*\(')
    fragments = []
    current = []
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'\s*//[/!] ?(.*)', line)
        if match:
            if not current:
                start = number
            current.append(match[1])
        elif current:
            fragments.append((start, current))
            current = []
    if current:
        fragments.append((start, current))
    # Block-form Rustdoc is uncommon but also executable; don't silently omit it.
    for match in re.finditer(r'/\*[*!](.*?)\*/', text, re.S):
        lines = [re.sub(r'^\s*\* ?', '', line) for line in match[1].splitlines()]
        fragments.append((text.count('\n', 0, match.start()) + 1, lines))
    for start, lines in fragments:
        block = None
        for offset, line in enumerate([*lines, '```']):
            fence = re.match(r'\s*(`{3,}|~{3,})(.*)', line)
            if fence:
                if block is None:
                    block = (offset, fence[2].strip(), [])
                else:
                    opening, tag, body = block
                    block = None
                    pieces = {piece.strip() for piece in tag.split(',')}
                    language = tag.split(',')[0].strip()
                    rust = language in ('', 'rust', 'should_panic') or language.startswith('edition')
                    if rust and not pieces.intersection({'no_run', 'ignore', 'compile_fail'}):
                        if learned.search(rust_mask('\n'.join(body))):
                            errors.append(f'{source}:{start + opening}: runnable Rustdoc executes a learned-model seam; use an explicit model proof, or no_run for compile-only documentation')
            elif block is not None:
                block[2].append(line)
            elif (re.match(r'\s{4,}(?:#\s*)?(?:let\b|return\b|assert[A-Za-z_]*!|[A-Za-z_][A-Za-z0-9_]*\s*\.|\.)', line)
                  and learned.search(rust_mask(line))):
                errors.append(f'{source}:{start + offset}: indented runnable Rustdoc executes a learned-model seam')
    return errors


# The workflow analysis is intentionally a conservative source interpreter, not a shell runner.
# Unknown executable edges are policy failures on automatic paths. Do not add workflow-name
# exemptions: extend the resolver with a tested command semantic instead.
UNKNOWN = '__MODEL_POLICY_UNRESOLVED__'


def condition_values(expression: str, context: dict) -> set:
    """Possible values of a GitHub condition; unknown atoms may be true OR false.

    In particular, a dispatch substring is not a gate (`dispatch || always()` is automatic).
    Unsupported syntax loses precision, never safety. No workflow text is evaluated as code.
    """
    import ast
    expression = str(expression or 'true').strip()
    if expression.startswith('${{') and expression.endswith('}}'):
        expression = expression[3:-2]
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', ' not ', expression).strip()
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return {True, False}

    def values(node):
        if isinstance(node, ast.Constant):
            return {node.value}
        if isinstance(node, (ast.Name, ast.Attribute)):
            key = ast.unparse(node)
            return set(context.get(key, {'true': {True}, 'false': {False}, 'null': {None}}.get(key, {UNKNOWN})))
        if isinstance(node, ast.BoolOp):
            result = values(node.values[0])
            for child in node.values[1:]:
                other = values(child)
                result = {item for value in result for item in ({True, False} if value == UNKNOWN else {value})}
                other = {item for value in other for item in ({True, False} if value == UNKNOWN else {value})}
                result = {bool(a) and bool(b) if isinstance(node.op, ast.And) else bool(a) or bool(b)
                          for a in result for b in other}
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return {result for value in values(node.operand) for result in ({True, False} if value == UNKNOWN else {not bool(value)})}
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                result = {outcome for a in values(node.left) for b in values(node.comparators[0])
                          for outcome in ({True, False} if UNKNOWN in (a, b) else {a == b})}
                return {not x for x in result} if isinstance(node.ops[0], ast.NotEq) else result
        return {UNKNOWN}
    return values(tree.body)


def can_run(expression: str, context: dict) -> bool:
    return any(bool(value) for value in condition_values(expression, context))


def automatic_jobs(workflow: dict, event: str, extra_context: dict | None = None, failed: frozenset = frozenset()) -> dict[str, bool]:
    """Resolve needs as well as job guards, without trusting skipped dependencies.

    A normal dependent job has GitHub's implicit success(). A status-function override can
    run after skipped prerequisites, so in that case only an explicit guard is proof.
    """
    jobs = workflow.get('jobs') or {}
    possible = {}
    active = set()

    def visit(name):
        if name in possible:
            return possible[name]
        if name in active or name not in jobs:
            return True
        if name in failed:
            possible[name] = False
            return False
        active.add(name)
        job = jobs[name]
        needs = job.get('needs') or []
        needs = [needs] if isinstance(needs, str) else needs
        parents = {parent: visit(parent) for parent in needs}
        context = {'github.event_name': {event}, **(extra_context or {})}
        context.update({f'needs.{parent}.result': {'failure'} if parent in failed else {'success', 'failure', 'cancelled', 'skipped'} if value else {'skipped'}
                        for parent, value in parents.items()})
        expression = str(job.get('if', 'true'))
        result = can_run(expression, context)
        if not re.search(r'\b(?:always|failure|cancelled|success)\s*\(', expression):
            result = result and all(parents.values())
        possible[name] = result
        active.remove(name)
        return result
    for name in jobs:
        visit(name)
    return possible


class RunGraph:
    """Resolve repository script/driver edges, without importing or executing the scripts.

    `model`, `proof`, and `unknown` findings carry their complete source-edge chain. Python
    subprocess argv literals and simple assignments are resolved; dynamic eval/exec/import,
    shell command heads and unavailable repository-local source fail closed. Cargo
    build/lint/metadata and ordinary ignored-model test invocations are not model execution.
    """
    SYSTEM = set('echo printf exit return true false set export unset local readonly shift test '
                 'cd pwd mkdir rm cp mv ln chmod touch cat head tail tee grep egrep fgrep sed awk '
                 'cut sort uniq wc tr find du df date sleep read basename dirname realpath '
                 'shasum sha256sum md5sum stat strings readlink mktemp uname hostname id ps pgrep '
                 'sysctl sw_vers vm_stat pmset lsof kill wait disown command type hash '
                 'git rustup rustc cmake ninja make xcodebuild xcode-select xcrun '
                 'curl wget tar unzip zip zstd gzip gunzip install file security codesign '
                 'uv pip pip3 sudo timeout nice nproc free nvidia-smi jq ls :'.split())
    PYTHON = re.compile(r'^(?:python(?:\d+(?:\.\d+)*)?(?:\.exe)?|py)$', re.I)
    SHELL = {'bash', 'sh', 'zsh', 'pwsh', 'powershell', 'cmd', 'call', '.', 'source', '&'}

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.active = set()
        self.findings = []
        self.variables = {}
        self.generated = {}
        self.test_artifacts = {}
        self.env_exports = {}
        self.imported_metadata = set()

    def finding(self, kind, chain, detail):
        self.findings.append((kind, ' -> '.join([*chain, detail])))

    def command(self, words, chain=(), base=None):
        if not words:
            return
        words = list(words)
        while words and (re.match(r'^\w+=', words[0]) or words[0] in ('env', 'exec', 'nohup')):
            words.pop(0)
        while words and words[0] == '--':
            words.pop(0)
        if not words:
            return
        head = words[0].strip('"\'')
        args = words[1:]
        normalized = head.replace('\\', '/')
        leaf = normalized.rsplit('/', 1)[-1]
        if ('$' in head or '%' in head or UNKNOWN in head) and any(argument.replace('\\', '/').rsplit('/', 1)[-1] == 'model_inference_proofs.py' for argument in args) and not set(args).intersection({'--plan', '--verify-artifacts'}):
            self.finding('proof', chain, 'possible model-proof execution through an unresolved interpreter')
            self.finding('model', chain, 'possible model-proof execution through an unresolved interpreter')
        listing = '--list' in args or '--help' in args or args == ['--version']
        if not listing and '--no-run' not in args and any(flag in args for flag in ('--ignored', '--include-ignored', '--run-ignored')):
            self.finding('model', chain, 'invocation overrides model-test ignores')
        if not listing and ('--product-command' in args or '--parent-command' in args):
            self.finding('model', chain, 'native model parent/product driver')
        if not listing and any(option in args for option in ('--snapshot', '--llama-snapshot', '--qwen-snapshot')) and any(mode in args for mode in ('parent', 'noise-floor-parent')):
            self.finding('model', chain, 'snapshot-bound native model parent')
        if leaf in ('cargo', 'rustc') and '--test' in args and '-o' in args:
            at = args.index('-o')
            if at + 1 < len(args):
                self.test_artifacts[args[at + 1]] = ' -> '.join(chain)
        if head in self.test_artifacts and (listing or not any(flag in args for flag in ('--ignored', '--include-ignored', '--run-ignored'))):
            self.finding('ordinary', chain, 'ordinary libtest artifact compiled at ' + self.test_artifacts[head])
            return
        if leaf == 'cargo':
            if '--no-run' in args or listing:
                self.finding('ordinary', chain, 'Cargo compilation/listing: ' + ' '.join(args))
                return
            if any(flag in args for flag in ('--ignored', '--include-ignored', '--run-ignored')):
                self.finding('model', chain, 'Cargo overrides model-test ignores')
            elif 'run' in args:
                self.finding('model', chain, 'Cargo native driver execution')
            elif not any(sub in args for sub in ('test', 'build', 'check', 'clippy', 'fmt', 'metadata', 'tree', 'doc', 'clean', 'install', 'deny', 'rustc', '--version')):
                self.finding('unknown', chain, 'unresolved Cargo subcommand: ' + ' '.join(args))
            elif any('$' in arg or UNKNOWN in arg for arg in args) and 'test' in args:
                self.finding('unknown', chain, 'dynamic Cargo test arguments')
            else:
                self.finding('ordinary', chain, 'Cargo metadata/build/lint/ordinary ignored-model test: ' + ' '.join(args))
            return
        if leaf in ('sudo', 'nice', 'timeout'):
            while args and args[0].startswith('-'):
                option = args.pop(0)
                if option in ('-u', '-g', '-n', '-k', '--kill-after', '--signal') and args: args.pop(0)
            if leaf == 'timeout' and args: args.pop(0)
            self.command(args or [UNKNOWN], (*chain, leaf + ' command'), base)
            return
        if leaf == 'uv' and args[:1] in (['run'], ['tool']):
            self.finding('unknown', chain, 'unresolved uv execution route')
            return
        if self.PYTHON.fullmatch(leaf):
            while args and args[0] in ('-I', '-u', '-B', '-E', '-s'):
                args.pop(0)
            if not args:
                self.finding('unknown', chain, 'Python code from stdin')
                return
            if args[0] in ('--version', '-V'): return
            if args[0] == '-c':
                if len(args) < 2:
                    self.finding('unknown', chain, 'missing Python code')
                else:
                    self.python(args[1], (*chain, 'python -c'), args[2:])
                return
            if args[0] == '-m':
                if len(args) < 2:
                    self.finding('unknown', chain, 'missing Python module')
                elif args[1] not in ('unittest', 'pip', 'venv', 'compileall'):
                    self.script(args[1].replace('.', '/') + '.py', args[2:], chain, base)
                return
            return self.script(args[0], args[1:], chain, base)
        if leaf.lower() in ('powershell', 'powershell.exe', 'pwsh', 'pwsh.exe'):
            # PowerShell also accepts a positional script after its host options. Normalize
            # that spelling to -File, retaining every argument for source/switch analysis.
            if not any(str(word).lower() in ('-file', '-command', '-c') for word in args):
                index = 0
                while index < len(args) and str(args[index]).lower() in ('-nologo', '-noexit', '-noprofile', '-noninteractive', '-sta', '-mta', '-executionpolicy', '-windowstyle', '-inputformat', '-outputformat'):
                    takes_value = str(args[index]).lower() in ('-executionpolicy', '-windowstyle', '-inputformat', '-outputformat')
                    index += 2 if takes_value else 1
                if index < len(args) and str(args[index]).lower().endswith('.ps1'):
                    args = [*args[:index], '-File', *args[index:]]
            quoted = ["'" + str(word).replace("'", "''") + "'" for word in [head, *args]]
            return self.powershell('& ' + ' '.join(quoted), chain, base)
        if leaf in self.SHELL:
            if not args:
                self.finding('unknown', chain, 'shell code from stdin')
            elif args[0] in ('-c', '-Command', '/c'):
                if len(args) > 1:
                    self.shell(args[1], chain, base)
                else:
                    self.finding('unknown', chain, 'dynamic shell code')
            else:
                while args and args[0].startswith('-'):
                    args.pop(0)
                if args:
                    if leaf in ('call', '&'): self.command(args, chain, base)
                    else: self.script(args[0], args[1:], chain, base)
            return
        if normalized.endswith(('.sh', '.cmd', '.ps1', '.py')):
            self.script(normalized, args, chain, base)
            return
        if leaf == 'trap':
            if args and args[0] not in ('-', ''):
                self.shell(args[0], (*chain, 'trap handler'), base)
            return
        if leaf == 'xargs':
            while args and args[0].startswith('-'):
                option = args.pop(0)
                if option in ('-n', '-P', '-I', '-L', '-s') and args: args.pop(0)
            self.command(args or ['echo'], (*chain, 'xargs command'), base)
            return
        if leaf == 'vcvars64.bat' and 'Microsoft Visual Studio' in normalized:
            self.finding('ordinary', chain, 'literal compiler toolchain environment activation: ' + normalized)
            return  # compiler toolchain environment activation, resolved from literal job env
        if leaf == 'lms' and args[:1] == ['ps']:
            self.finding('ordinary', chain, 'source-proven lms tool lookup followed by process inventory only')
            return  # process inventory, not load/generate
        if leaf == 'cp' and len(args) >= 2 and args[-2] in self.test_artifacts:
            self.test_artifacts[args[-1]] = self.test_artifacts[args[-2]]
        if leaf == 'eval':
            self.finding('unknown', chain, 'dynamic eval execution')
        elif leaf in self.SYSTEM:
            self.finding('ordinary', chain, 'source-resolved tool operation: ' + ' '.join([head, *args]))
            return
        elif normalized.startswith(('target/', './target/')) or '/target/' in normalized:
            self.finding('model', chain, 'native driver: ' + head)
        elif '$' in head or '%' in head or head == UNKNOWN:
            self.finding('unknown', chain, 'dynamic executable: ' + head)
        elif '/' in normalized:
            self.finding('unknown', chain, 'unresolved native driver: ' + head)
        else:
            self.finding('unknown', chain, 'unresolved command: ' + head)

    def script(self, name, args, chain, base=None):
        name = name.replace('\\', '/')
        name = name.replace('${{ github.workspace }}/', '').replace('$GITHUB_WORKSPACE/', '')
        name = name.replace('${GITHUB_WORKSPACE}/', '')
        if Path(name).name == 'model_inference_proofs.py':
            if not set(args).intersection({'--plan', '--verify-artifacts'}):
                self.finding('proof', chain, name)
                self.finding('model', chain, name)
            return
        if name in self.generated:
            self.shell(self.generated[name], (*chain, 'generated ' + name), base, args)
            return
        path = self.root / name
        if not path.is_file() and base is not None:
            path = base / name
        try:
            path = path.resolve()
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            self.finding('unknown', chain, 'script outside source root: ' + name)
            return
        if not path.is_file():
            self.finding('unknown', chain, 'unresolved script: ' + name)
            return
        # The exact reviewed suite runner may execute the literal routine partition in
        # ordinary CI. Its executable argv is built from Cargo compiler-artifact records,
        # which this graph cannot resolve statically; bind the entire source body and CLI
        # mode instead of blessing arbitrary dynamic subprocess calls.
        if relative == 'scripts/ci/test_suites.py':
            reviewed_sha = 'a4b0d80b4c34becb98b056ee9351eec4bbf622c6ca72865e54c4c7ac488aa488'
            command_at = args.index('--') if '--' in args else -1
            command = args[command_at + 1:] if command_at >= 0 else []
            if (hashlib.sha256(path.read_bytes()).hexdigest() != reviewed_sha
                    or args[:command_at].count('routine') != 1
                    or any(mode in args[:command_at] for mode in ('exhaustive', 'check'))
                    or command[:2] != ['cargo', 'test']
                    or any(flag in command for flag in ('--ignored', '--include-ignored', '--run-ignored'))):
                self.finding('unknown', chain, 'unreviewed suite source, mode, or Cargo command')
            else:
                self.finding('ordinary', chain, 'exact-source routine suite over ordinary ignored-model Cargo tests')
            return
        key = (relative, tuple(args))
        if key in self.active:
            self.finding('unknown', chain, 'recursive execution edge: ' + relative)
            return
        self.active.add(key)
        before = len(self.findings)
        chain = (*chain, relative)
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            output = self.python(text, chain, args)
        elif path.suffix.lower() == '.ps1':
            output = self.powershell(text, chain, path.parent, args)
        else:
            output = self.shell(text, chain, path.parent, args, windows=path.suffix.lower() == '.cmd')
        if not any(kind in ('model', 'proof', 'unknown') for kind, _ in self.findings[before:]):
            self.finding('ordinary', chain, 'source-resolved script has no model or unresolved executable edges')
        self.active.remove(key)
        return output

    def python(self, text, chain, args=(), entry="__main__", function_args=()):
        import ast
        try:
            tree = ast.parse(text)
        except SyntaxError:
            self.finding('unknown', chain, 'unparseable Python execution')
            return
        if entry not in (None, '__main__'):
            entry = entry.rsplit('.', 1)[-1]
            def literal(item):
                if isinstance(item, (list, tuple)):
                    return ast.List(elts=[literal(part) for part in item], ctx=ast.Load())
                return ast.Constant(value=item if isinstance(item, (str, int, float, bool, type(None), Path)) else UNKNOWN)
            call = ast.Call(func=ast.Name(id=entry, ctx=ast.Load()), args=[literal(item) for item in function_args], keywords=[])
            ast.fix_missing_locations(call)
            tree.body.append(ast.Expr(value=call))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        remainder = 'argparse.REMAINDER' in text

        def scope(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return node
            return tree

        # CLI identity comes from argparse declarations, including subparser and append options.
        append_options = {}
        subparser_options = {}
        parser_names = set()
        for item in ast.walk(tree):
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == 'parse_args':
                parser_names.update(target.id for target in item.targets if isinstance(target, ast.Name))
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute): continue
            keywords = {part.arg: part.value.value for part in item.keywords if isinstance(part.value, ast.Constant)}
            flags = [part.value for part in item.args if isinstance(part, ast.Constant) and isinstance(part.value, str)]
            if item.func.attr == 'add_argument' and keywords.get('action') == 'append' and flags:
                destination = keywords.get('dest', flags[-1].lstrip('-').replace('-', '_'))
                append_options[destination] = flags
            if item.func.attr == 'add_subparsers' and keywords.get('dest'):
                subparser_options[keywords['dest']] = {part.args[0].value for part in ast.walk(tree)
                    if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr == 'add_parser'
                    and part.args and isinstance(part.args[0], ast.Constant)}

        mutated_parser_attributes = set()
        for item in ast.walk(tree):
            target = item.func.value if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr in ('append', 'extend', 'insert', 'pop', 'remove', 'clear', 'reverse', 'sort') else item if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store) else None
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in parser_names:
                mutated_parser_attributes.add((target.value.id, target.attr))

        artifact_schema = any(isinstance(item, ast.If) and all(piece in ast.unparse(item.test) for piece in ("get('reason') == 'compiler-artifact'", "get('profile', {}).get('test')", "get('executable')")) for item in ast.walk(tree))
        assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(target, ast.Name):
                        assignments.setdefault((scope(node), target.id), []).append(ast.BinOp(left=target, op=ast.Add(), right=node.value) if isinstance(node, ast.AugAssign) else node.value)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                assignments.setdefault((scope(node), node.target.id), []).extend(node.iter.elts if isinstance(node.iter, (ast.Tuple, ast.List)) else [ast.Subscript(value=node.iter, slice=ast.Constant(value=0), ctx=ast.Load())])
        # Mutations are executable dataflow too: looking only at argv's initializer would
        # miss command.append('--ignored') or removal of a compile-only flag.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                name = node.func.value.id
                operation = node.func.attr
                if operation in ('append', 'extend') and node.args:
                    addition = ast.List(elts=[node.args[0]], ctx=ast.Load()) if operation == 'append' else node.args[0]
                    update = ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=ast.Add(), right=addition)
                    assignments.setdefault((scope(node), name), []).append(update)
                elif operation in ('insert', 'pop', 'remove', 'clear', 'reverse', 'sort'):
                    assignments.setdefault((scope(node), name), []).append(ast.Constant(value=UNKNOWN))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        assignments.setdefault((scope(node), target.value.id), []).append(ast.Constant(value=UNKNOWN))
        resolving = set()
        resolving_functions = set()
        bindings = {}

        class Choices:
            def __init__(self, items): self.items = items

        def variants(item):
            return item.items if isinstance(item, Choices) else [item]

        def merge(possibilities):
            flattened = []
            for choice in possibilities:
                for item in variants(choice):
                    if isinstance(item, list) and item[:1] == ['--']: item = item[1:]
                    if item not in flattened: flattened.append(item)
            if not flattened or len(flattened) > 128: return UNKNOWN
            return flattened[0] if len(flattened) == 1 else Choices(flattened)

        def combine(left, right, operation):
            results = []
            for a in variants(left):
                for b in variants(right):
                    try: results.append(operation(a, b))
                    except (TypeError, IndexError, ValueError): results.append(UNKNOWN)
                    if len(results) > 128: return UNKNOWN
            return merge(results)

        def package_pairs(function):
            """Prove stdout package selection is only -p/name pairs, never arbitrary Cargo flags.

            Both direct caller patterns and scope words filtered by stdlib fnmatch to those
            patterns are overapproximated by the complete declared package universe. This is
            source analysis only: CI still receives the actual narrowed selector output.
            """
            returns = [item.value for item in ast.walk(function) if isinstance(item, ast.Return)]
            if not returns: return UNKNOWN
            if any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == function.name for item in ast.walk(tree)): return UNKNOWN
            if any(isinstance(item, (ast.AugAssign, ast.Delete)) or isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Store) for item in ast.walk(function)): return UNKNOWN
            parameters = {item.arg for item in function.args.args}
            assigned = {}
            for item in ast.walk(function):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name): assigned.setdefault(target.id, []).append(item.value)
            def patterns(node):
                if isinstance(node, ast.Name) and node.id in parameters:
                    values = value(node, function)
                    if isinstance(values, list) and values and all(isinstance(part, str) and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_*-]*', part) for part in values):
                        return values
                if not isinstance(node, ast.Name) or len(assigned.get(node.id, [])) != 1: return None
                expression = assigned[node.id][0]
                if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == 'sorted' and len(expression.args) == 1): return None
                generator = expression.args[0]
                if not (isinstance(generator, ast.GeneratorExp) and isinstance(generator.elt, ast.Name) and len(generator.generators) == 1): return None
                outer = generator.generators[0]
                if not (isinstance(outer.target, ast.Name) and outer.target.id == generator.elt.id and len(outer.ifs) == 1): return None
                if not (isinstance(outer.iter, ast.Call) and isinstance(outer.iter.func, ast.Attribute) and outer.iter.func.attr == 'split' and not outer.iter.args and not outer.iter.keywords): return None
                condition = outer.ifs[0]
                if not (isinstance(condition, ast.Call) and isinstance(condition.func, ast.Name) and condition.func.id == 'any' and len(condition.args) == 1): return None
                matches = condition.args[0]
                if not (isinstance(matches, ast.GeneratorExp) and len(matches.generators) == 1 and isinstance(matches.elt, ast.Call)): return None
                inner = matches.generators[0]
                call = matches.elt
                if not (isinstance(inner.target, ast.Name) and not inner.ifs and ast.unparse(call.func) == 'fnmatch.fnmatchcase'
                        and len(call.args) == 2 and ast.unparse(call.args[0]) == outer.target.id and ast.unparse(call.args[1]) == inner.target.id): return None
                # The qualified matcher must be the stdlib import, not a rebound helper.
                if not any(isinstance(item, ast.Import) and any(alias.name == 'fnmatch' and alias.asname in (None, 'fnmatch') for alias in item.names) for item in tree.body): return None
                if any(isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in ('fnmatch', 'any', 'sorted') for item in ast.walk(tree)): return None
                if any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id in ('fnmatch', 'any', 'sorted') for item in ast.walk(tree)): return None
                return patterns(inner.iter)
            domains = []
            for returned in returns:
                if not (isinstance(returned, ast.ListComp) and isinstance(returned.elt, ast.Name) and len(returned.generators) == 2): return UNKNOWN
                outer, inner = returned.generators
                if not (isinstance(outer.target, ast.Name) and isinstance(inner.target, ast.Name)
                        and not outer.ifs and not inner.ifs and returned.elt.id == inner.target.id
                        and isinstance(inner.iter, (ast.Tuple, ast.List)) and len(inner.iter.elts) == 2
                        and isinstance(inner.iter.elts[0], ast.Constant) and inner.iter.elts[0].value == '-p'
                        and isinstance(inner.iter.elts[1], ast.Name) and inner.iter.elts[1].id == outer.target.id): return UNKNOWN
                domain = patterns(outer.iter)
                if domain is None: return UNKNOWN
                domains.extend(domain)
            # Mutating selector data after filtering would invalidate the bounded domain.
            if any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr in ('append', 'extend', 'insert', 'pop', 'remove', 'clear', 'reverse', 'sort') for item in ast.walk(function)): return UNKNOWN
            return [argument for name in dict.fromkeys(domains) for argument in ('-p', name)]

        def value(node, context=None):
            context = context or scope(node)
            if isinstance(node, ast.Constant): return node.value
            if isinstance(node, (ast.List, ast.Tuple)):
                result = []
                for child in node.elts:
                    if isinstance(child, ast.Starred):
                        part = value(child.value, context)
                        result = combine(result, part, lambda a, b: a + (b if isinstance(b, list) else [UNKNOWN]))
                    else:
                        result = combine(result, value(child, context), lambda a, b: a + [b])
                return result
            if isinstance(node, ast.Name):
                if node.id == '__file__': return str(self.root / chain[-1])
                key = (context, node.id)
                if key in resolving: return bindings.get(key, UNKNOWN)
                resolving.add(key)
                entries = assignments.get(key, assignments.get((tree, node.id), []))
                if entries:
                    candidates = []
                    for entry in entries:
                        candidate = value(entry, context)
                        candidates.append(candidate)
                        bindings[key] = candidate
                    result = merge(candidates)
                else:
                    result = UNKNOWN
                    parameters = [item.arg for item in context.args.args] if hasattr(context, 'args') else []
                    if hasattr(context, 'args') and context.args.vararg and context.args.vararg.arg == node.id:
                        callsites = [call for call in ast.walk(tree) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == context.name]
                        result = merge([value(ast.List(elts=call.args[len(parameters):], ctx=ast.Load()), scope(call)) for call in callsites])
                    if node.id in parameters:
                        index = parameters.index(node.id)
                        callsites = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                                     and isinstance(call.func, ast.Name) and call.func.id == context.name and len(call.args) > index]
                        result = merge([value(call.args[index], scope(call)) for call in callsites])
                resolving.remove(key)
                return result
            if isinstance(node, ast.Attribute):
                name = ast.unparse(node)
                if name == 'sys.executable': return 'python3'
                if isinstance(node.value, ast.Name) and (node.value.id, node.attr) in mutated_parser_attributes: return UNKNOWN
                option = '--' + node.attr.replace('_', '-')
                if isinstance(node.value, ast.Name) and node.value.id == 'args' and option in args:
                    at = args.index(option)
                    if at + 1 < len(args):
                        return Path(args[at + 1]) if any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == 'add_argument' and any(isinstance(arg, ast.Constant) and arg.value == option for arg in item.args) and any(keyword.arg == 'type' and isinstance(keyword.value, ast.Name) and keyword.value.id == 'Path' for keyword in item.keywords) for item in ast.walk(tree)) else args[at + 1]
                if remainder and node.attr == 'command' and '--' in args:
                    return list(args[args.index('--'):])
                if isinstance(node.value, ast.Name) and node.value.id in parser_names:
                    if node.attr in append_options:
                        flags = append_options[node.attr]
                        return [args[index + 1] for index, flag in enumerate(args[:-1]) if flag in flags]
                    if node.attr in subparser_options and args and args[0] in subparser_options[node.attr]:
                        return args[0]
                parent = value(node.value, context)
                if isinstance(parent, Path):
                    if node.attr == 'parent': return parent.parent
                    if node.attr == 'parents': return list(parent.parents)
            if isinstance(node, ast.BinOp):
                left, right = value(node.left, context), value(node.right, context)
                if isinstance(node.op, ast.Add):
                    return combine(left, right, lambda a, b: a + b if isinstance(a, type(b)) and isinstance(a, (str, list)) else UNKNOWN)
                if isinstance(node.op, ast.Div):
                    return combine(left, right, lambda a, b: Path(a) / b if isinstance(a, (Path, str)) and isinstance(b, str) else UNKNOWN)
            if isinstance(node, ast.IfExp):
                if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                    left, right = value(node.test.left, context), value(node.test.comparators[0], context)
                    if left != UNKNOWN and right != UNKNOWN and isinstance(node.test.ops[0], (ast.Eq, ast.NotEq)):
                        truth = left == right
                        if isinstance(node.test.ops[0], ast.NotEq): truth = not truth
                        return value(node.body if truth else node.orelse, context)
                return merge([value(node.body, context), value(node.orelse, context)])
            if isinstance(node, ast.Subscript):
                if artifact_schema and isinstance(node.slice, ast.Constant) and node.slice.value == 'executable':
                    return '__MODEL_POLICY_CARGO_HARNESS__'
                source = value(node.value, context)
                if ast.unparse(node.value) == 'sys.argv': source = [chain[-1] if chain else '<inline>', *args]
                if isinstance(source, (list, tuple, str, Choices)):
                    if isinstance(node.slice, ast.Slice):
                        lower = value(node.slice.lower, context) if node.slice.lower else None
                        upper = value(node.slice.upper, context) if node.slice.upper else None
                        # Unknown slices cannot add executing arguments: retaining the full
                        # source overapproximates every possible flag in the forwarded argv.
                        if any(item is not None and not isinstance(item, int) for item in (lower, upper)): return source
                        return merge([item[lower:upper] if isinstance(item, (list, tuple, str)) else UNKNOWN for item in variants(source)])
                    index = value(node.slice, context)
                    return combine(source, index, lambda a, b: a[b])
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ('str', 'list', 'tuple', 'Path') and node.args:
                    item = value(node.args[0], context)
                    if name == 'str': return merge([str(part) for part in variants(item)])
                    if name == 'Path' and item != UNKNOWN: return Path(item)
                    if name in ('list', 'tuple'):
                        return merge([list(part) if isinstance(part, (list, tuple)) else UNKNOWN for part in variants(item)])
                if isinstance(node.func, ast.Attribute) and node.func.attr in ('resolve', 'absolute'):
                    item = value(node.func.value, context)
                    if isinstance(item, Path): return item
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'join' and node.args:
                    separator, parts = value(node.func.value, context), value(node.args[0], context)
                    if isinstance(separator, str) and isinstance(parts, list) and all(isinstance(part, str) for part in parts):
                        return separator.join(parts)
                if isinstance(node.func, ast.Name) and node.func.id in functions:
                    function = functions[node.func.id]
                    if function in resolving_functions: return UNKNOWN
                    resolving_functions.add(function)
                    try:
                        packages = package_pairs(function)
                        if packages != UNKNOWN: return packages
                        # Resolve pure argv constructors by substituting their source parameters.
                        returns = [item.value for item in ast.walk(function) if isinstance(item, ast.Return) and item.value is not None]
                        return merge([value(item, function) for item in returns])
                    finally:
                        resolving_functions.remove(function)
            return UNKNOWN

        # Follow the script entrypoint and local calls. CLI store_true branches are
        # decidable from this invocation: --self-test must not traverse the unused producer.
        boolean_flags = {}
        parser_results = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if any(item.arg == 'action' and isinstance(item.value, ast.Constant) and item.value.value == 'store_true' for item in node.keywords):
                    flags = [item.value for item in node.args if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value.startswith('--')]
                    for flag in flags:
                        boolean_flags[flag[2:].replace('-', '_')] = flag in args
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args':
                parser_results.update(target.id for target in node.targets if isinstance(target, ast.Name))

        def truth(node):
            if isinstance(node, ast.Constant): return bool(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                result = truth(node.operand)
                return None if result is None else not result
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in parser_results:
                return boolean_flags.get(node.attr)
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == '__name__' and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant):
                equal = ('__main__' if entry == '__main__' else '<imported>') == node.comparators[0].value
                return not equal if isinstance(node.ops[0], ast.NotEq) else equal
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.left, ast.Attribute)
                    and isinstance(node.left.value, ast.Name) and node.left.value.id in parser_names):
                left, right = value(node.left), value(node.comparators[0])
                if isinstance(left, (str, int, bool)) and isinstance(right, (str, int, bool)) and UNKNOWN not in (left, right):
                    if isinstance(node.ops[0], ast.Eq): return left == right
                    if isinstance(node.ops[0], ast.NotEq): return left != right
            return None

        function_aliases = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Name) and item.value.id in functions:
                for target in item.targets:
                    if isinstance(target, ast.Name): function_aliases.setdefault(target.id, set()).add(item.value.id)
        reachable = set()
        seen_functions = set()

        def expression(node):
            if node is None: return
            reachable.add(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): return
            if isinstance(node, ast.Call):
                called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                if called in classes:
                    for method in classes[called].body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in ('__init__', '__new__', '__post_init__'):
                            if method not in seen_functions:
                                seen_functions.add(method); statements(method.body)
                for candidate in function_aliases.get(called, {called}):
                    if candidate in functions and candidate not in seen_functions:
                        seen_functions.add(candidate)
                        statements(functions[candidate].body)
            for child in ast.iter_child_nodes(node): expression(child)

        def statements(body):
            for node in body:
                reachable.add(node)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
                if isinstance(node, ast.If):
                    expression(node.test)
                    result = truth(node.test)
                    if result is not None:
                        if statements(node.body if result else node.orelse): return True
                    else:
                        left = statements(node.body); right = statements(node.orelse)
                        if left and right: return True
                elif isinstance(node, (ast.Return, ast.Raise)):
                    expression(node); return True
                elif isinstance(node, ast.Try):
                    statements(node.body)
                    for handler in node.handlers: statements(handler.body)
                    statements(node.orelse); statements(node.finalbody)
                else:
                    expression(node)
            return False
        statements(tree.body)
        for node in reachable:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or '']
                if any(module.split('.')[0] in ('torch', 'transformers', 'diffusers', 'mlx', 'mlx_lm') for module in modules):
                    self.finding('unknown', (*chain, f'line {node.lineno}'), 'learned-framework Python execution requires an explicit model route')

        # Repository-local imports are source edges too. Parse their top-level code as
        # imports (not __main__), and follow actual imported function calls without loading
        # the module. This catches wrapper.py -> helper.run() -> native/model process.
        local_imports = {}
        source_path = next((self.root / part for part in reversed(chain) if (self.root / part).is_file()), self.root / 'inline.py')

        def local_module(module_name, level=0):
            if level:
                directory = source_path.parent
                for _ in range(level - 1): directory = directory.parent
                candidate = directory / module_name.replace('.', '/')
                choices = [candidate.with_suffix('.py'), candidate / '__init__.py']
            else:
                relative = module_name.replace('.', '/')
                choices = [self.root / (relative + '.py'), source_path.parent / (relative + '.py'), self.root / relative / '__init__.py']
            for candidate in choices:
                if candidate.is_file() and candidate.resolve().is_relative_to(self.root): return candidate.resolve()
            return None

        def imported(path, function=None, arguments=()):
            relative = path.relative_to(self.root).as_posix()
            key = ('python-import', relative, function, repr(arguments))
            if function is None:
                if relative in self.imported_metadata: return
                self.imported_metadata.add(relative)
            if key in self.active:
                if function is not None: self.finding('unknown', chain, 'recursive imported Python execution: ' + relative)
                return
            self.active.add(key)
            self.python(path.read_text(encoding='utf-8'), (*chain, relative), entry=function, function_args=arguments)
            self.active.remove(key)

        for item in reachable:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    path = local_module(alias.name)
                    if path:
                        local_imports[alias.asname or alias.name] = (path, None)
                        imported(path)
            elif isinstance(item, ast.ImportFrom):
                path = local_module(item.module or '', item.level)
                for alias in item.names:
                    child = local_module((item.module + '.' if item.module else '') + alias.name, item.level)
                    if child:
                        local_imports[alias.asname or alias.name] = (child, None)
                        imported(child)
                    elif path:
                        local_imports[alias.asname or alias.name] = (path, alias.name)
                        imported(path)
        for item in list(reachable):
            if not isinstance(item, ast.Call): continue
            called = ast.unparse(item.func)
            head, dot, tail = called.partition('.')
            if isinstance(item.func, ast.Attribute) and isinstance(item.func.value, ast.Call) and isinstance(item.func.value.func, ast.Name) and item.func.value.func.id in local_imports:
                path, constructor = local_imports[item.func.value.func.id]
                imported(path, (constructor + '.' if constructor else '') + item.func.attr, [value(argument) for argument in item.args])
            if head in local_imports:
                path, function = local_imports[head]
                if dot and function is not None:
                    module_tree = ast.parse(path.read_text(encoding='utf-8'))
                    literals = [node.value for node in module_tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == function for target in node.targets)]
                    try:
                        data = ast.literal_eval(literals[0]) if len(literals) == 1 else None
                    except (ValueError, TypeError):
                        data = None
                    if isinstance(data, (dict, list, tuple, set, str)) and tail in ('get', 'items', 'keys', 'values', 'count', 'index', 'copy'):
                        continue
                    function += '.' + tail
                else:
                    function = tail if dot else function
                if function:
                    arguments = [value(argument) for argument in item.args]
                    if any(isinstance(argument, Choices) for argument in arguments):
                        self.finding('unknown', (*chain, f'line {getattr(item, "lineno", 1)}'), 'ambiguous imported Python callable arguments')
                    else:
                        imported(path, function, arguments)

        if entry not in (None, '__main__') and entry not in functions and entry not in classes and entry not in local_imports and entry not in function_aliases:
            definitions = assignments.get((tree, entry), [])
            if not definitions or not all(isinstance(item, (ast.Attribute, ast.Name, ast.Lambda)) for item in definitions):
                self.finding('unknown', chain, 'unresolved imported Python callable: ' + entry)
        aliases = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                for alias in item.names: aliases[alias.asname or alias.name] = alias.name
            elif isinstance(item, ast.ImportFrom) and item.module in ('subprocess', 'os'):
                for alias in item.names: aliases[alias.asname or alias.name] = item.module + '.' + alias.name
        for item in ast.walk(tree):
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Attribute):
                target = ast.unparse(item.value)
                first, dot, rest = target.partition('.')
                target = aliases.get(first, first) + dot + rest
                if target.startswith(('subprocess.', 'os.exec', 'os.spawn')):
                    for name in item.targets:
                        if isinstance(name, ast.Name): aliases[name.id] = target
        pending = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or node not in reachable: continue
            if isinstance(node.func, (ast.Call, ast.Subscript)):
                self.finding('unknown', (*chain, f'line {node.lineno}'), 'computed Python callable execution')
            name = ast.unparse(node.func)
            head, separator, tail = name.partition('.')
            name = aliases.get(head, head) + (separator + tail if separator else '')
            if name.startswith(('os.exec', 'os.spawn')):
                self.finding('unknown', (*chain, f'line {node.lineno}'), 'native process replacement requires resolved executable/argv provenance')
            if (name.startswith('subprocess.') and name.rsplit('.', 1)[-1] in ('run', 'call', 'check_call', 'check_output', 'Popen')
                    or name in ('os.system', 'os.popen') or name.startswith(('os.exec', 'os.spawn'))):
                command_node = node.args[0] if node.args else next((item.value for item in node.keywords if item.arg == 'args'), None)
                command = value(command_node) if command_node is not None else UNKNOWN
                where = (*chain, f'line {node.lineno}')
                pending.extend((candidate, where) for candidate in variants(command))
            elif name in ('eval', 'exec', '__import__', 'importlib.import_module'):
                self.finding('unknown', (*chain, f'line {node.lineno}'), 'dynamic Python execution')

        # Artifact construction is a source fact independent of AST traversal order.
        for candidate, where in pending:
            if isinstance(candidate, list) and candidate and candidate[0] in ('cargo', 'rustc') and '--test' in candidate and '-o' in candidate:
                at = candidate.index('-o')
                if at + 1 < len(candidate): self.test_artifacts[str(candidate[at + 1])] = ' -> '.join(where)
        if artifact_schema and any(isinstance(candidate, list) and any(candidate[index:index + 2] == ['cargo', 'test'] for index in range(len(candidate) - 1)) and '--no-run' in candidate and '--message-format=json' in candidate for candidate, _ in pending):
            self.test_artifacts['__MODEL_POLICY_CARGO_HARNESS__'] = ' -> '.join(chain) + ': guarded Cargo compiler-artifact test/executable records from --no-run compilation'
        for candidate, where in pending:
            if isinstance(candidate, list): self.command([str(piece) for piece in candidate], where)
            elif isinstance(candidate, str) and candidate != UNKNOWN: self.shell(candidate, where)
            else: self.finding('unknown', where, 'dynamic Python process argv')
        outputs = [value(item.args[0]) for item in reachable if isinstance(item, ast.Call)
                   and isinstance(item.func, ast.Name) and item.func.id == 'print' and len(item.args) == 1]
        if (len(outputs) == 1 and isinstance(outputs[0], str) and UNKNOWN not in outputs[0]
                and not any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == 'print' for item in ast.walk(tree))):
            return outputs[0]

    def powershell(self, text, chain=(), base=None, args=()):
        from scripts.ci.model_policy_powershell import analyze
        return analyze(self, text, chain, base, args)

    def shell(self, text, chain=(), base=None, args=(), variables=None, windows=False):
        """Tokenize complete shell text (not individual YAML lines).

        Quoted multiline Python/awk and case patterns are data. Substitutions and sourced
        functions are execution edges. Unsupported executable expansions remain unknown.
        """
        import shlex
        variables = dict(self.variables if variables is None else variables)
        variables.update({str(index): str(argument) for index, argument in enumerate(args, 1)})
        variables['@'] = ' '.join(shlex.quote(str(arg)) for arg in args) if args else UNKNOWN
        variables['#'] = str(len(args))
        if base is not None:
            variables.setdefault('SCRIPT_DIR', str(base))
        functions = getattr(self, 'functions', {})
        self.functions = functions

        pending_substitutions = {}
        arrays = {}
        command_outputs = []

        def expand(word):
            for symbol, code in pending_substitutions.items():
                if symbol in word:
                    output = self.shell(code, (*chain, "command substitution"), base, args, variables)
                    lookup = re.fullmatch(r"\s*command\s+-v\s+([\w.-]+)\s*", code)
                    managed = re.fullmatch(r'\s*uv python find [0-9.]+ --managed-python(?: --no-project)? --no-config\s*', code)
                    result = lookup[1] if lookup else 'python3' if managed else output if isinstance(output, str) else UNKNOWN
                    # Build stdout is a compiler-artifact JSON record, filtered to exactly one
                    # test executable by a metadata-only Python reader before being copied.
                    if all(piece in code for piece in ('cargo test ', '--no-run', '--message-format=json', 'compiler-artifact', 'm.get("profile", {}).get("test")', 'm.get("executable")', 'print(found[-1] if len(found) == 1 else "")')):
                        result = '__MODEL_POLICY_CARGO_HARNESS__'
                        self.test_artifacts[result] = ' -> '.join(chain) + ': Cargo --no-run compiler-artifact test executable selection'
                    # A tool-lookup function may return either command -v TOOL or one
                    # literal fallback path with the SAME executable basename. This is
                    # source provenance, not a helper/function-name exemption.
                    if code.strip() in functions:
                        body, _ = functions[code.strip()]
                        pattern = r'\s*if command -v (?P<tool>[\w.-]+) >/dev/null 2>&1; then command -v (?P=tool)\s*\n\s*elif \[ -x "(?P<path>[^"\n]+)" \]; then echo "(?P=path)"\s*\n\s*fi\s*'
                        producer = re.fullmatch(pattern, body)
                        if producer and producer['path'].rsplit('/', 1)[-1] == producer['tool']:
                            result = producer['tool']
                    word = word.replace(symbol, result)
            def substitution(match):
                key = match[1] or match[2]
                if ':' in key:
                    key, default = key.split(':', 1)
                    default = default[1:] if default.startswith('-') else UNKNOWN
                    return variables.get(key, default)
                return variables.get(key, '$' + key)
            word = re.sub(r'%([A-Za-z_]\w*)%', lambda match: variables.get(match[1], match[0]), word) if windows else word
            return re.sub(r'\$\{([^}]+)\}|\$(\w+|[@#])', substitution, word)

        # Here-docs are parsed as source only for the interpreter that consumes them.
        text = re.sub(r'(?m)^\s*#.*$', '', text)
        lines = text.replace('\\\n', ' ').splitlines(keepends=True)
        filtered = []
        index = 0
        while index < len(lines):
            line = lines[index]
            heredoc = re.search(r'(?<!<)<<-?\s*[\'"]?([A-Za-z_][A-Za-z_0-9]*)', line)
            if heredoc:
                index += 1
                body = []
                while index < len(lines) and lines[index].strip() != heredoc[1]:
                    body.append(lines[index]); index += 1
                if re.search(r'\bpython[\d.]*\b', line):
                    self.python(''.join(body), (*chain, 'Python heredoc'))
                    line = ''
                else:
                    destination = re.search(r'\bcat\s*>\s*[\"\']?([^\"\'\s]+)', line)
                    if destination:
                        self.generated[destination[1]] = ''.join(body)
                    line = line[:heredoc.start()] + '\n'
            filtered.append(line)
            index += 1
        text = ''.join(filtered)
        # shlex consumes the newline along with an inline comment. That merges the next
        # executable into the previous command's argv (e.g. echo ... # comment\npython ...),
        # hiding its source edge. Mask comments first, retaining command boundaries and quoted,
        # escaped, or word-internal hashes; then disable shlex's newline-eating comment handler.
        masked = []
        index = 0
        quote = None
        while index < len(text):
            char = text[index]
            if char == '\\' and quote != "'" and index + 1 < len(text):
                masked.append(text[index:index + 2]); index += 2; continue
            if char in ('"', "'"):
                if quote == char: quote = None
                elif quote is None: quote = char
            if char == '#' and quote is None and (index == 0 or text[index - 1] in ' \t\r\n;&|()<>'):
                while index < len(text) and text[index] != '\n':
                    masked.append(' '); index += 1
                continue
            masked.append(char); index += 1
        text = ''.join(masked)
        original_text = text
        if windows:
            # Managed CPython discovery is a tool-identity proof, not a variable-name rule.
            pattern = r'''for /f "delims=" %%([A-Za-z]) in \('uv python find [0-9.]+ --managed-python --no-project --no-config'\) do set "([A-Za-z_]\w*)=%%\1"'''
            text = re.sub(pattern, lambda match: match[2] + '=python3', text)
        text = re.sub(r'\$\{\{.*?\}\}', UNKNOWN, text, flags=re.S)
        # Static dirname-of-current-source is a local source edge, not a dynamic filename.
        if base is not None:
            text = re.sub(r'\$\(dirname\s+(?:--\s+)?[\'"]?(?:\$0|\$\{BASH_SOURCE\[0\]\})[\'"]?\)', str(base), text)
            text = re.sub(r'\$\(cd\s+[\'"]?' + re.escape(str(base)) + r'[\'"]?\s*&&\s*pwd\)', str(base), text)
        # Extract balanced command substitutions, respecting single quotes and escaped ticks.
        def substitutions(source):
            output = []
            index = 0
            quote = None
            while index < len(source):
                char = source[index]
                if char == '\\' and quote != "'" and index + 1 < len(source):
                    output.append(source[index:index + 2]); index += 2; continue
                if char in ('"', "'"):
                    if quote == char:
                        quote = None
                    elif quote is None:
                        quote = char
                    output.append(char); index += 1; continue
                start = index
                if quote != "'" and source.startswith('$(', index) and not source.startswith('$((', index):
                    depth = 1; index += 2; opening = index; inner_quote = None
                    while index < len(source) and depth:
                        current = source[index]
                        if current == '\\':
                            index += 2; continue
                        if current in ('"', "'"):
                            if inner_quote == current: inner_quote = None
                            elif inner_quote is None: inner_quote = current
                        elif inner_quote != "'":
                            if current == '(': depth += 1
                            elif current == ')': depth -= 1
                        index += 1
                    if depth:
                        self.finding('unknown', chain, 'unbalanced command substitution')
                        output.append(UNKNOWN); continue
                    code = source[opening:index - 1]
                    symbol = '__SHELL_SUB_' + str(len(pending_substitutions)) + '__'
                    pending_substitutions[symbol] = code
                    output.append(symbol)
                    continue
                if quote != "'" and char == '`':
                    end = index + 1
                    while end < len(source) and source[end] != '`':
                        end += 2 if source[end] == '\\' else 1
                    if end >= len(source):
                        self.finding('unknown', chain, 'unbalanced backtick execution')
                    else:
                        symbol = '__SHELL_SUB_' + str(len(pending_substitutions)) + '__'
                        pending_substitutions[symbol] = source[index + 1:end]
                        output.append(symbol)
                    index = end + 1; continue
                output.append(char); index = start + 1
            return ''.join(output)
        # Function definitions must be registered before evaluating substitutions in their
        # bodies; execution occurs only at a callsite with its own argument bindings.
        definition = re.compile(r'(?m)^\s*(?:function\s+)?([A-Za-z_]\w*)\s*\(\)\s*\{')
        position = 0
        while match := definition.search(text, position):
            index = match.end(); opening = index; depth = 1; quote = None
            while index < len(text) and depth:
                char = text[index]
                if char == '\\' and quote != "'": index += 2; continue
                if char in ('"', "'"):
                    if quote == char: quote = None
                    elif quote is None: quote = char
                elif quote is None:
                    if char == '{': depth += 1
                    elif char == '}': depth -= 1
                index += 1
            if depth:
                self.finding('unknown', chain, 'unparseable shell function: ' + match[1]); break
            functions[match[1]] = (text[opening:index - 1], base)
            text = text[:match.start()] + '\n' + text[index:]
            position = match.start() + 1
        text = substitutions(text)
        try:
            lexer = shlex.shlex(text, posix=True, punctuation_chars=';&|()<>\n')
            if windows: lexer.escape = ''
            lexer.whitespace = ' \t\r'
            lexer.whitespace_split = True
            lexer.commenters = ''
            tokens = list(lexer)
        except ValueError:
            self.finding('unknown', chain, 'unparseable shell execution'); return
        # Newlines abutting words are returned inside those tokens by shlex.
        # Punctuation runs may combine a semicolon and newlines; split those, not quoted data.
        tokens = [part for token in tokens for part in (re.split(r'(\n+)', token) if token and all(char in ';&|()<>\n' for char in token) else [token]) if part]
        command = []; in_case_header = False; in_case_pattern = False; in_for = False; bracket = None

        def execute(words):
            if not words: return
            words = list(words)
            while words and words[0] in ('if', 'then', 'elif', 'else', 'do', '{', '!', '@'):
                words.pop(0)
            if not words: return
            if words[0] in ('fi', 'done', 'esac', '}', 'break', 'continue', 'return'):
                return
            if windows and words[:2] == ['not', 'exist'] and len(words) > 3:
                words = words[3:]
            elif windows and words[:2] == ['not', 'defined'] and len(words) > 3:
                words = words[3:]
            declaration = words[0] in ('local', 'export', 'readonly')
            if declaration:
                words.pop(0)
            while words and re.match(r'^\w+=', words[0]):
                key, value = words.pop(0).split('=', 1); variables[key] = expand(value)
                if key in arrays: arrays[key] = [UNKNOWN]
            if declaration:
                for word in words:
                    if '=' in word:
                        key, value = word.split('=', 1); variables[key] = expand(value)
                return
            if words and words[0] in ('$@', '${@}'):
                words = list(args) + words[1:]
            expanded = []
            for word in words:
                array = re.fullmatch(r'\$\{([A-Za-z_]\w*)\[@\]\}', word)
                if array and array[1] in arrays:
                    expanded.extend(arrays[array[1]])
                else:
                    expanded.append(expand(word))
            words = expanded
            if not words: return
            if words[0] == 'unset':
                for variable in words[1:]:
                    if variable in arrays: arrays[variable] = [UNKNOWN]
            name = words[0]
            if name in functions:
                key = ('shell-function', name, tuple(words[1:]))
                if key in self.active:
                    self.finding('unknown', chain, 'recursive shell function: ' + name); return
                self.active.add(key)
                body, directory = functions[name]
                command_outputs.append(self.shell(body, (*chain, name + '()'), directory, words[1:], variables))
                self.active.remove(key)
            else:
                self.variables.update(variables)
                command_outputs.append(self.command(words, chain, base))
                variables.update(self.variables)
        redirection = False
        here_string = False
        arithmetic = 0
        array_depth = 0
        for token in [*tokens, '\n']:
            if token.isspace(): token = '\n'
            if array_depth:
                if token == '(': array_depth += 1
                elif token == ')': array_depth -= 1
                else: expand(token)
                continue
            if token == '(' and command and re.fullmatch(r'\w+\+?=', command[-1]):
                name = command[-1].rstrip('=+')
                variables[name] = UNKNOWN
                arrays[name] = [UNKNOWN]
                command = []; array_depth = 1; continue
            if redirection:
                if here_string and command[:1] == ['read'] and '-a' in command:
                    at = command.index('-a')
                    if at + 1 < len(command):
                        source = expand(token)
                        arrays[command[at + 1]] = shlex.split(source) if UNKNOWN not in source else [UNKNOWN]
                redirection = False
                here_string = False
                continue
            if token in ('>', '>>', '<', '<<', '<<<', '>&', '<&', '>>&'):
                if command and command[-1].isdigit(): command.pop()
                redirection = True
                here_string = token == '<<<'
                continue
            if token == '((' or arithmetic:
                if token == '((': arithmetic += 1
                elif token == '))': arithmetic -= 1
                continue
            if bracket:
                if token == bracket: bracket = None
                continue
            if token in ('[[', '['):
                bracket = ']]' if token == '[[' else ']'; continue
            if in_for:
                if token == 'do': in_for = False
                continue
            if token in ('then', 'else', 'do'):
                execute(command); command = []; continue
            if token == 'while' and not command:
                continue
            if token in ('for', 'select') and not command:
                in_for = True; continue
            if token == 'case' and not command:
                in_case_header = True; continue
            if in_case_header:
                if token == 'in': in_case_header = False; in_case_pattern = True
                continue
            if in_case_pattern:
                if token == ')': in_case_pattern = False
                elif token == 'esac': in_case_pattern = False
                continue
            if token in (';;', ';&', ';;&'):
                execute(command); command = []; in_case_pattern = True; continue
            if token == '\n' or token and all(char in ';&|()\n' for char in token):
                execute(command); command = []
            else:
                command.append(token)
        self.variables.update(variables)
        for match in re.finditer(r'echo\s+[\"\']?([A-Za-z_]\w*)=([^\"\n]*?)[\"\']?\s*>>\s*[\"\']?(?:%GITHUB_ENV%|\$GITHUB_ENV|\$\{GITHUB_ENV\})', original_text):
            self.env_exports[match[1]] = expand(match[2])
        if len(command_outputs) == 1:
            return command_outputs[0]

def selection_error(profile: str, confirm_all: bool = False) -> str | None:
    if not profile:
        return 'Select an explicit profile; a bare dispatch never runs model families.'
    if profile == 'all' and not confirm_all:
        return 'profile=all requires the additional explicit confirm_all=true confirmation.'
    return None



def selection_binding_error(profile: str, binding: str) -> str | None:
    # The existing iOS JSON carrier is consulted ONLY for the all-profile aggregate.
    if profile != 'all':
        return selection_error(profile)
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate confirmation key')
            result[key] = value
        return result
    try:
        confirmation = json.loads(binding, object_pairs_hook=object_pairs)
    except (TypeError, ValueError):
        confirmation = None
    if not isinstance(confirmation, dict) or set(confirmation) != {'confirm_all'} or confirmation['confirm_all'] is not True:
        return 'profile=all requires exactly {"confirm_all": true} in ios_pricing_compile_binding; no extra keys are accepted.'
    return None


def selection_barriers(workflow: dict, inputs: dict) -> set[str]:
    """Recognize the source-owned fail-fast validator, not a workflow/job-name exemption."""
    barriers = set()
    for name, job in (workflow.get('jobs') or {}).items():
        if job.get('needs') or job.get('if') or job.get('continue-on-error'):
            continue
        for step in job.get('steps', []):
            if step.get('if') or step.get('continue-on-error'):
                continue
            if step.get('run', '').strip() not in ('python3.12 scripts/check_model_inference_policy.py --validate-selection',
                                                  'python3 scripts/check_model_inference_policy.py --validate-selection',
                                                  'python3 -I scripts/check_model_inference_policy.py --validate-selection'):
                continue
            environment = step.get('env', {})
            if environment.get('MODEL_PROFILE') != '${{ inputs.profile }}':
                continue
            carrier = re.fullmatch(r"\$\{\{ inputs\.profile == 'all' && inputs\.([A-Za-z_]\w*) \|\| '' \}\}", environment.get('MODEL_SELECTION_BINDING', ''))
            if carrier is None:
                carrier = re.fullmatch(r'\$\{\{ inputs\.([A-Za-z_]\w*) \}\}', environment.get('MODEL_SELECTION_BINDING', ''))
            if not carrier or inputs.get(carrier[1], {}).get('type') != 'string':
                continue
            # No default may pre-confirm the broad aggregate.
            try:
                default = json.loads(inputs[carrier[1]].get('default', ''))
            except (TypeError, ValueError):
                default = None
            if isinstance(default, dict) and default.get('confirm_all') is True:
                continue
            if any(other is not step and not str(other.get('uses', '')).startswith('actions/checkout@') for other in job.get('steps', [])):
                continue
            barriers.add(name)
    return barriers

def workflow_errors(workflow: dict, name: str, root: Path = ROOT, evidence: list | None = None, seen: frozenset = frozenset()) -> list[str]:
    if name in seen:
        return [name + ': unresolved recursive reusable workflow execution']
    seen = seen | {name}
    events = workflow.get('on', workflow.get(True, {}))
    names = {events} if isinstance(events, str) else set(events or ())
    automatic = names - {'workflow_dispatch'}
    errors = []
    jobs = workflow.get('jobs') or {}
    inputs = (events.get('workflow_dispatch') or {}).get('inputs', {}) if isinstance(events, dict) else {}
    profile = inputs.get('profile')
    if profile:
        if profile.get('default') not in ('', None) or not profile.get('required'):
            errors.append(name + ': profile must require explicit selection with an empty default')
        barriers = selection_barriers(workflow, inputs)
        if 'all' in profile.get('options', []):
            confirmation = inputs.get('confirm_all', {})
            direct_confirmation = confirmation.get('type') == 'boolean' and confirmation.get('default') is False
            if not barriers and not direct_confirmation:
                errors.append(name + ': aggregate all requires an unconfirmed-by-default selection validator or explicit boolean confirmation')
        for selected in ('', 'all'):
            context = {'inputs.profile': {selected}, 'inputs.confirm_all': {False}}
            possible = automatic_jobs(workflow, 'workflow_dispatch', context, frozenset(barriers))
            for job_name, job in jobs.items():
                if not possible.get(job_name, True):
                    continue
                # Status-function overrides may run receipt-only work after failed validation.
                # They do NOT establish a model gate: inspect every surviving execution edge.
                for step in [job] if 'uses' in job else job.get('steps', []):
                    if not can_run(step.get('if', ''), {'github.event_name': {'workflow_dispatch'}, **context}):
                        continue
                    graph = RunGraph(root)
                    if 'run' in step:
                        graph.variables = {key: str(value) for environment in (workflow.get('env', {}), job.get('env', {}), step.get('env', {})) for key, value in environment.items()}
                        if step.get('shell', '').lower() in ('powershell', 'pwsh'):
                            graph.powershell(step['run'], (f'{name}/{job_name}',))
                        else:
                            graph.shell(step['run'], (f'{name}/{job_name}',), windows=step.get('shell') == 'cmd')
                    elif 'uses' in job or str(step.get('uses', '')).startswith('./'):
                        graph.finding('unknown', (name, job_name), 'selection barrier does not dominate reusable execution')
                    if any(kind in ('model', 'proof', 'unknown') for kind, _ in graph.findings):
                        errors.append(f'{name}/{job_name}: unset/all selection model path is not dominated by successful selection validation')
    # Dispatch-only triggers already dominate every executable edge, including unknowns.
    # Only proof-concurrency checks with a shared pending group need further source tracing.
    if not automatic and ('concurrency' not in workflow or 'github.run_id' in str(workflow['concurrency'])):
        return errors
    for job_name, job in jobs.items():
        job_environment = {}
        for step in [job] if 'uses' in job else job.get('steps', []):
            graph = RunGraph(root)
            if 'run' in step:
                graph.variables = {key: str(value) for env in (workflow.get('env', {}), job.get('env', {}), job_environment, step.get('env', {})) for key, value in env.items()}
                if step.get('shell', '').lower() in ('powershell', 'pwsh'):
                    graph.powershell(step['run'], (f'{name}/{job_name}',))
                else:
                    graph.shell(step['run'], (f'{name}/{job_name}',), windows=step.get('shell') == 'cmd')
                job_environment.update(graph.env_exports)
            elif str(step.get('uses', '')).startswith('./'):
                target = str(step['uses'])[2:]
                import yaml
                path = root / target
                if path.is_dir():
                    path = next((path / file for file in ('action.yml', 'action.yaml') if (path / file).is_file()), path)
                if path.is_file():
                    child = yaml.safe_load(path.read_text(encoding='utf-8'))
                    if 'jobs' in child:
                        child = dict(child)
                        callable_events = {event for event in automatic if automatic_jobs(workflow, event).get(job_name, True) and can_run(step.get('if', ''), {'github.event_name': {event}})}
                        child['on'] = {event: None for event in callable_events or {'workflow_dispatch'}}
                        child.pop(True, None)
                        errors.extend(workflow_errors(child, target, root, evidence, seen))
                    else:
                        graph.finding('unknown', (name, job_name), 'local action execution requires resolution: ' + target)
                else:
                    graph.finding('unknown', (name, job_name), 'unresolved reusable workflow/action: ' + target)
            elif 'uses' in job:
                graph.finding('unknown', (name, job_name), 'unresolved remote reusable workflow: ' + str(job['uses']))
            if evidence is not None:
                evidence.extend({'workflow': name, 'job': job_name, 'event': event, 'kind': kind, 'source_chain': detail} for event in automatic if automatic_jobs(workflow, event).get(job_name, True) and can_run(step.get('if', ''), {'github.event_name': {event}}) for kind, detail in graph.findings if kind == 'ordinary')
            for event in automatic:
                if automatic_jobs(workflow, event).get(job_name, True) and can_run(step.get('if', ''), {'github.event_name': {event}}):
                    for kind, detail in graph.findings:
                        if kind == 'ordinary': continue
                        errors.append(f'{name}/{job_name}: automatic {event} reaches {kind} execution: {detail}')
            if any(kind == 'proof' for kind, _ in graph.findings) and 'concurrency' in workflow and 'github.run_id' not in str(workflow['concurrency']):
                errors.append(name + ': model proof execution cannot replace pending dispatches through workflow concurrency')
    return list(dict.fromkeys(errors))

def check(root: Path = ROOT) -> list[str]:
    import yaml
    registry = root / 'scripts/ci/model-inference-proofs.json'
    if not registry.is_file():
        return ['model-proof registry is missing']
    spec = importlib.util.spec_from_file_location('model_proof_policy_runner', root / 'scripts/ci/model_inference_proofs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        rows = module.load_registry(registry)
    except (ValueError, OSError) as error:
        return [str(error)]
    legacy_path = root / 'scripts/ci/ios-selected-captures.json'
    legacy = json.loads(legacy_path.read_text(encoding='utf-8')) if legacy_path.is_file() else []
    errors = registry_errors(root, rows, legacy)
    for source in sorted(set(root.glob('crates/**/*.rs')) | set(root.glob('ios-host/**/*.rs'))):
        errors.extend(model_doc_errors(source.read_text(encoding='utf-8'), source.relative_to(root).as_posix()))
    for path in sorted((root / '.github/workflows').glob('*.y*ml')):
        errors.extend(workflow_errors(yaml.safe_load(path.read_text(encoding='utf-8')), path.name, root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=ROOT)
    parser.add_argument('--validate-selection', action='store_true', help='fail fast on a bare dispatch or unconfirmed all profile')
    args = parser.parse_args()
    if args.validate_selection:
        import os
        profile = os.environ.get('MODEL_PROFILE', '')
        if 'MODEL_SELECTION_BINDING' in os.environ:
            error = selection_binding_error(profile, os.environ.get('MODEL_SELECTION_BINDING', '') if profile == 'all' else '')
        else:
            error = selection_error(profile, os.environ.get('MODEL_CONFIRM_ALL', '').lower() == 'true')
        if error:
            print('::error::' + error, file=sys.stderr)
        return bool(error)
    errors = check(args.root.resolve())
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print('Model-inference source/workflow routing OK; compiler listings and explicit runtime proofs are separate')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
