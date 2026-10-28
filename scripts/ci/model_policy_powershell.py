"""Conservative, source-only PowerShell execution-edge analysis.

This is deliberately not a PowerShell interpreter. It resolves a bounded language
of source strings, arrays, functions, callbacks and subprocess setup. Unknown
execution is reported to RunGraph, never run or silently treated as metadata.
"""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import re

UNKNOWN = '<unresolved-powershell>'


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    adjacent_call: bool = False


@dataclass(frozen=True)
class Group:
    kind: str
    items: tuple


@dataclass(frozen=True)
class Block:
    items: tuple


@dataclass
class Object:
    kind: str
    fields: dict


@dataclass(frozen=True)
class Alternatives:
    values: tuple


@dataclass(frozen=True)
class Function:
    params: tuple
    body: tuple


def _lex(text):
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ' \t\r':
            i += 1
            continue
        if text.startswith('<#', i):
            end = text.find('#>', i + 2)
            if end < 0:
                raise ValueError('unclosed block comment')
            i = end + 2
            continue
        if c == '#':
            i = text.find('\n', i)
            if i < 0:
                break
            continue
        if c == '`' and text[i + 1:i + 2] in ('\n', '\r'):
            i += 2
            continue
        if text[i:i + 2] in ("@'", '@"') and text[i + 2:i + 3] in ('\r', '\n'):
            quote = text[i + 1]
            start = text.find('\n', i) + 1
            match = re.search(r'(?m)^' + re.escape(quote + '@'), text[start:])
            if not match:
                raise ValueError('unclosed here-string')
            end = start + match.start()
            tokens.append(Token('single' if quote == "'" else 'double', text[start:end].rstrip('\r\n')))
            i = start + match.end()
            continue
        if c in "'\"":
            quote, value = c, ''
            i += 1
            while i < len(text):
                if text[i] == quote:
                    if text[i + 1:i + 2] == quote:
                        value += quote
                        i += 2
                        continue
                    i += 1
                    break
                if quote == '"' and text[i] == '`' and i + 1 < len(text):
                    value += {'n': '\n', 'r': '\r', 't': '\t'}.get(text[i + 1], text[i + 1])
                    i += 2
                else:
                    value += text[i]
                    i += 1
            else:
                raise ValueError('unclosed string')
            tokens.append(Token('single' if quote == "'" else 'double', value))
            continue
        if c == '[':
            depth, end = 1, i + 1
            while end < len(text) and depth:
                if text[end] == '[': depth += 1
                if text[end] == ']': depth -= 1
                end += 1
            if depth:
                raise ValueError('unclosed type/index')
            if text[end:end + 2] == '::':
                end += 2
                while end < len(text) and (text[end].isalnum() or text[end] == '_'): end += 1
            tokens.append(Token('word', text[i:end], text[end:end + 1] == '('))
            i = end
            continue
        if text[i:i + 2] in ('@(', '@{', '$('):
            tokens.append(Token('symbol', text[i:i + 2]))
            i += 2
            continue
        if c in '\n;(){}|,&=+':
            if text[i:i + 2] in ('+=', '&&', '||'):
                tokens.append(Token('symbol', text[i:i + 2]))
                i += 2
            else:
                tokens.append(Token('symbol', c))
                i += 1
            continue
        start = i
        while i < len(text) and text[i] not in " \t\r\n;(){}|,&=+'\"":
            if text[i] == '[':
                break
            i += 1
        if i == start:
            i += 1
        tokens.append(Token('word', text[start:i], text[i:i + 1] == '('))
    return tokens


def _groups(tokens):
    index = 0
    def collect(close=None):
        nonlocal index
        result = []
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if token.value in (')', '}') and token.kind == 'symbol':
                if token.value != close:
                    raise ValueError('unmatched closing delimiter')
                return tuple(result)
            if token.kind == 'symbol' and token.value in ('(', '@(', '$(', '{', '@{'):
                end = ')' if token.value.endswith('(') else '}'
                result.append(Group(token.value, collect(end)))
            else:
                result.append(token)
        if close:
            raise ValueError('unclosed delimiter')
        return tuple(result)
    return collect()


def _is(item, value):
    return isinstance(item, Token) and item.kind in ('word', 'symbol') and item.value.lower() == value.lower()


def _split(items, separators):
    result, current = [], []
    for item in items:
        if isinstance(item, Token) and item.kind == 'symbol' and item.value in separators:
            result.append(tuple(current)); current = []
        else:
            current.append(item)
    result.append(tuple(current))
    return result


def _params(items):
    result = []
    for part in _split(items, {','}):
        name = next((t.value.lower() for t in part if isinstance(t, Token) and re.fullmatch(r'\$[\w:]+', t.value)), None)
        if name:
            result.append((name, any(isinstance(t, Token) and t.value.lower() == '[switch]' for t in part), part))
    return tuple(result)


class Analyzer:
    # Cmdlets here operate on metadata, files, streams or process inventory. Their
    # argument expressions and scriptblocks are still recursively inspected.
    CMDLETS = set('set-strictmode write-output write-warning write-error write-host out-null '
                  'out-file get-content set-content add-content clear-content remove-item '
                  'copy-item move-item new-item get-item test-path get-filehash join-path '
                  'split-path resolve-path get-location push-location pop-location set-location '
                  'get-command select-object sort-object where-object foreach-object '
                  'convertto-json convertfrom-json convertfrom-csv add-member remove-variable '
                  'get-ciminstance start-sleep format-list format-table measure-object '
                  'get-childitem get-process stop-process'.split())
    # Instance operations are bounded data/IO/process observation, never arbitrary
    # model methods. Process Start is handled separately and must resolve its argv.
    METHODS = set('tostring touniversaltime addticks addminutes addseconds addhours trim '
                  'tolowerinvariant startswith endswith contains containskey add remove '
                  'copy replace split join dispose close refresh kill waitforexit get_exitcode '
                  'get_handle readtoendasync getawaiter getresult copytoasync write read '
                  'writealltext readalltext readalllines move delete createdirectory '
                  'setenvironmentvariable getenvironmentvariable ispathrooted gettemppath '
                  'getfullpath getfilename parse matches match isnullorempty isnullorwhitespace '
                  'isnan isinfinity max min waitall getprocessbyid startnew open exists '
                  'openstandardoutput openstandarderror newguid'.split())
    TYPES = set('datetime datetimeoffset datetimekind timespan math regex string stringcomparer '
                'environment guid intptr valueType system.array system.collections.generic.hashset[string] '
                'io.path io.file io.directory io.filemode io.fileaccess io.fileshare io.fileattributes '
                'io.directoryinfo io.filenotfoundexception text.utf8encoding threading.tasks.task '
                'threading.tasks.task[] diagnostics.stopwatch diagnostics.process '
                'diagnostics.processstartinfo system.diagnostics.process system.diagnostics.processstartinfo'.lower().split())

    def __init__(self, graph, chain, base):
        self.graph, self.chain, self.base = graph, tuple(chain), Path(base or graph.root)
        self.env = {'$true': True, '$false': False, '$null': None,
                    '$psscriptroot': str(self.base), '$pshome': '__PSHOME__',
                    '$env:comspec': 'cmd.exe'}
        self.functions = {}
        self.generated = {}
        self.active = set()
        self.steps = 0

    def finding(self, detail):
        self.graph.finding('unknown', self.chain, 'PowerShell: ' + detail)

    def parse(self, text):
        try:
            return _groups(_lex(text))
        except (ValueError, RecursionError) as error:
            self.finding(str(error))
            return ()

    def analyze(self, text, args=()):
        items = self.parse(text)
        # Bind only parameters actually declared in source. Unknown switches must
        # never select an early-exit branch and hide the real entry point.
        declaration = next((items[i + 1] for i, t in enumerate(items[:-1])
                            if _is(t, 'param') and isinstance(items[i + 1], Group)), None)
        params = _params(declaration.items) if declaration else ()
        supplied = list(args)
        for name, switch, source in params:
            default = next((i for i, t in enumerate(source) if _is(t, '=')), None)
            self.env[name] = False if switch else self.value(source[default + 1:]) if default is not None else UNKNOWN
        positions = [p[0] for p in params]
        while supplied:
            argument = str(supplied.pop(0))
            if argument.startswith('-'):
                name = '$' + argument[1:].lower()
                match = next((p for p in params if p[0] == name), None)
                if not match:
                    self.finding('undeclared script parameter ' + argument)
                    continue
                self.env[name] = True if match[1] else supplied.pop(0) if supplied else UNKNOWN
                if name in positions: positions.remove(name)
            elif positions:
                self.env[positions.pop(0)] = argument
            else:
                self.finding('unbound positional script argument ' + argument)
        self.block(items)

    def variable(self, name):
        key = name.lower()
        if key in self.env:
            return self.env[key]
        if key.startswith('$env:'):
            return '__PS_ENV_' + key[5:] + '__'
        if '.' in key:
            root, field = key.rsplit('.', 1)
            obj = self.variable(root)
            if isinstance(obj, Object): return obj.fields.get(field, UNKNOWN)
            if field == 'source' and isinstance(obj, str): return obj
        return UNKNOWN

    def string(self, token):
        if token.kind == 'single': return token.value
        value = token.value
        # Interpolated subexpressions are executable even in metadata arguments.
        for match in re.finditer(r'\$\((.*?)\)', value, re.S):
            self.expr(self.parse(match.group(1)))
        value = re.sub(r'\$\(.*?\)', UNKNOWN, value, flags=re.S)
        return re.sub(r'\$[\w:]+', lambda m: str(self.variable(m.group())), value)

    def value(self, items):
        items = tuple(t for t in items if not _is(t, '\n'))
        if not items: return None
        plus = _split(items, {'+'})
        if len(plus) > 1:
            vals = [self.value(p) for p in plus]
            if any(isinstance(v, list) for v in vals):
                return [x for v in vals for x in (v if isinstance(v, list) else [v])]
            return ''.join(str(v) for v in vals)
        commas = _split(items, {','})
        if len(commas) > 1: return [self.value(p) for p in commas if p]
        if len(items) == 1:
            item = items[0]
            if isinstance(item, Group):
                if item.kind == '{': return Block(item.items)
                if item.kind == '@{': return Object('metadata', {})
                if item.kind == '@(':
                    comma_parts = _split(item.items, {','})
                    if len(comma_parts) > 1: return [self.value(p) for p in comma_parts]
                    parts = _split(item.items, {',', '\n', ';'})
                    if all(len(p) == 1 and not isinstance(p[0], Group) for p in parts if p):
                        return [self.value(p) for p in parts if p]
                    value = self.value(item.items)
                    return value if isinstance(value, list) else [value]
                return self.value(item.items)
            if item.kind in ('single', 'double'): return self.string(item)
            if item.value.startswith('$'): return self.variable(item.value)
            if re.fullmatch(r'@[\w:]+', item.value): return self.variable('$' + item.value[1:])
            if re.fullmatch(r'-?\d+', item.value): return int(item.value)
            return item.value
        # Casts do not turn metadata into execution; preserve the underlying value.
        if isinstance(items[0], Token) and re.fullmatch(r'\[[\w.\[\]]+\]', items[0].value):
            return self.value(items[1:])
        if _is(items[0], 'join-path'):
            words = self.arguments(items[1:])
            if len(words) >= 2: return str(words[0]).rstrip('/\\') + '/' + str(words[1]).lstrip('/\\')
        if _is(items[0], 'get-command') and len(items) > 1:
            return self.value(items[1:2])
        if len(items) >= 2 and isinstance(items[0], Token) and isinstance(items[1], Group):
            call = items[0].value.lower()
            if call.endswith('::gettemppath'): return '__PS_TEMP__'
            if call.endswith('::new'):
                typ = call.split(']')[0].lstrip('[')
                if typ in ('diagnostics.processstartinfo', 'system.diagnostics.processstartinfo'): return Object('startinfo', {})
                if typ in ('diagnostics.process', 'system.diagnostics.process'): return Object('process', {})
            if call.endswith('.replace'):
                obj = self.variable(items[0].value[:-8]); parts = [self.value(p) for p in _split(items[1].items, {','})]
                if isinstance(obj, str) and len(parts) == 2: return obj.replace(str(parts[0]), str(parts[1]))
            if call in ('[string]::join', '[system.string]::join'):
                parts = [self.value(p) for p in _split(items[1].items, {','})]
                if len(parts) == 2 and isinstance(parts[1], list): return str(parts[0]).join(str(x) for x in parts[1])
            # Get-Command expression followed by .Source.
        if isinstance(items[0], Group) and len(items) == 2 and _is(items[1], '.source'):
            return self.value(items[:1])
        return UNKNOWN

    def arguments(self, items):
        # Lexical adjacency is retained in a word; parenthesized expressions and
        # arrays are single arguments, as PowerShell's argument mode requires.
        result = []
        index = 0
        while index < len(items):
            item = items[index]
            index += 1
            if isinstance(item, Group) and index < len(items) and isinstance(items[index], Token) and items[index].value.lower() == '.source':
                result.append(self.value((item, items[index]))); index += 1; continue
            if _is(item, '\n') or _is(item, ','): continue
            if isinstance(item, Token) and item.value in ('*>', '>', '2>', '2>&1'):
                break
            value = self.value((item,))
            if isinstance(item, Token) and re.fullmatch(r'@[\w:]+', item.value) and isinstance(value, list): result.extend(value)
            else: result.append(value)
        return result

    def condition(self, items):
        items = tuple(t for t in items if not _is(t, '\n'))
        if len(items) == 1 and isinstance(items[0], Group): return self.condition(items[0].items)
        if len(items) == 1 and isinstance(items[0], Token):
            value = self.variable(items[0].value) if items[0].value.startswith('$') else UNKNOWN
            return value if isinstance(value, bool) else None
        for operator in ('-and', '-or'):
            at = next((i for i, t in enumerate(items) if _is(t, operator)), None)
            if at is not None:
                left, right = self.condition(items[:at]), self.condition(items[at + 1:])
                if operator == '-and': return False if left is False or right is False else True if left is True and right is True else None
                return True if left is True or right is True else False if left is False and right is False else None
        if items and _is(items[0], '-not'):
            value = self.condition(items[1:])
            return not value if value is not None else None
        for operator in ('-eq', '-ne', '-ceq', '-cne'):
            at = next((i for i, t in enumerate(items) if _is(t, operator)), None)
            if at is not None:
                left, right = self.value(items[:at]), self.value(items[at + 1:])
                if UNKNOWN not in str(left) and UNKNOWN not in str(right):
                    equal = left == right
                    return not equal if operator in ('-ne', '-cne') else equal
        return None

    def state(self):
        return deepcopy(self.env), self.functions.copy(), self.generated.copy()

    def restore(self, state):
        self.env, self.functions, self.generated = deepcopy(state[0]), state[1].copy(), state[2].copy()

    def join(self, states):
        for index in range(3):
            merged = {}
            for key in set().union(*(state[index] for state in states)):
                values = [state[index].get(key, UNKNOWN) for state in states]
                if all(value == values[0] for value in values): merged[key] = values[0]
                elif index == 0: merged[key] = UNKNOWN
                else:
                    variants = []
                    for value in values:
                        for part in value.values if isinstance(value, Alternatives) else (value,):
                            if part not in variants: variants.append(part)
                    merged[key] = Alternatives(tuple(variants))
            if index == 0: self.env = merged
            elif index == 1: self.functions = merged
            else: self.generated = merged

    def conditional(self, items):
        original = self.state()
        states, endings = [], []
        index, fallthrough = 0, True
        while index < len(items) and fallthrough:
            token = items[index]
            if _is(token, 'if') or _is(token, 'elseif'):
                if index + 2 >= len(items) or not isinstance(items[index + 1], Group) or not isinstance(items[index + 2], Group):
                    self.finding('unresolved conditional'); return False
                condition, body = items[index + 1:index + 3]
                self.restore(original)
                self.expr(condition.items)
                selected = self.condition(condition.items)
                if selected is not False:
                    endings.append(self.block(body.items)); states.append(self.state())
                if selected is True: fallthrough = False
                index += 3
            elif _is(token, 'else') and index + 1 < len(items) and isinstance(items[index + 1], Group):
                self.restore(original)
                endings.append(self.block(items[index + 1].items)); states.append(self.state())
                fallthrough = False
                index += 2
            else:
                self.finding('unresolved conditional tail'); return False
        if fallthrough:
            states.append(original); endings.append(False)
        self.join(states)
        return bool(endings) and all(endings)

    def try_blocks(self, items):
        original = self.state()
        branches, finalizers = [], []
        index = 0
        while index < len(items):
            token = items[index]
            if any(_is(token, word) for word in ('try', 'catch', 'finally')):
                label = token.value.lower()
                index += 1
                if label == 'catch' and index < len(items) and isinstance(items[index], Token) and items[index].value.startswith('['): index += 1
                if index >= len(items) or not isinstance(items[index], Group):
                    self.finding('unresolved exception block'); return False
                (finalizers if label == 'finally' else branches).append(items[index].items)
                index += 1
            else:
                self.finding('unresolved exception flow'); return False
        states = []
        for branch in branches:
            self.restore(original); self.block(branch); states.append(self.state())
        if states: self.join(states)
        for finalizer in finalizers: self.block(finalizer)
        return False

    def block(self, items):
        statements = []
        for part in _split(items, {'\n', ';'}):
            if not part: continue
            if statements and any(_is(part[0], word) for word in ('else', 'elseif', 'catch', 'finally')):
                statements[-1] += part
            else: statements.append(part)
        for statement in statements:
            if statement and self.statement(statement): return True
        return False

    def statement(self, items):
        self.steps += 1
        if self.steps > 30000:
            if self.steps == 30001: self.finding('analysis bound exceeded')
            return False
        if not items: return False
        if _is(items[0], 'function'):
            if len(items) < 3 or not isinstance(items[-1], Group):
                self.finding('unresolved function declaration'); return False
            name = items[1].value.lower()
            header = items[2] if isinstance(items[2], Group) and items[2].kind == '(' else None
            body = items[-1].items
            if header is None and body and _is(body[0], 'param') and isinstance(body[1], Group): header = body[1]
            self.functions[name] = Function(_params(header.items) if header else (), body)
            return False
        # Attributes on param declarations carry no runtime edges.
        if _is(items[0], 'param') or (isinstance(items[0], Token) and items[0].value.lower().startswith('[cmdletbinding')):
            return False
        if _is(items[0], 'if'):
            return self.conditional(items)
        if _is(items[0], 'try'):
            return self.try_blocks(items)
        if any(_is(items[0], word) for word in ('if', 'elseif', 'while', 'foreach', 'for', 'do', 'try', 'catch', 'finally')):
            index, guaranteed = 0, False
            while index < len(items):
                token = items[index]
                if _is(token, 'if') or _is(token, 'elseif') or _is(token, 'while'):
                    condition = items[index + 1] if index + 1 < len(items) else None
                    body = items[index + 2] if index + 2 < len(items) else None
                    if not isinstance(condition, Group):
                        self.finding('unresolved control flow'); return False
                    if not isinstance(body, Group) and _is(token, 'while'):
                        self.expr(condition.items); return False
                    if not isinstance(body, Group):
                        self.finding('unresolved control flow'); return False
                    self.expr(condition.items)
                    selected = self.condition(condition.items)
                    if selected is not False:
                        ended = self.block(body.items)
                        if selected is True and not _is(token, 'while'): guaranteed = ended
                    if _is(token, 'if') and selected is True: return guaranteed
                    index += 3
                elif _is(token, 'foreach'):
                    condition, body = items[index + 1:index + 3]
                    self.expr(condition.items)
                    if isinstance(condition, Group) and isinstance(body, Group):
                        name = condition.items[0].value.lower() if condition.items and isinstance(condition.items[0], Token) else ''
                        at = next((j for j, t in enumerate(condition.items) if _is(t, 'in')), None)
                        values = self.value(condition.items[at + 1:]) if at is not None else UNKNOWN
                        for value in values if isinstance(values, list) and values else [UNKNOWN]:
                            self.env[name] = value
                            self.block(body.items)
                    index += 3
                elif isinstance(token, Group):
                    self.block(token.items); index += 1
                else:
                    index += 1
            return guaranteed
        if _is(items[0], 'exit') or _is(items[0], 'return'):
            self.expr(items[1:]); return True
        if _is(items[0], 'throw'):
            self.expr(items[1:]); return False
        assignment = next((i for i, t in enumerate(items) if _is(t, '=') or _is(t, '+=')), None)
        if assignment is not None and assignment == 1 and isinstance(items[0], Token) and items[0].value.startswith('$'):
            rhs = items[assignment + 1:]
            self.expr(rhs)
            name = items[0].value.lower()
            value = self.value(rhs)
            if _is(items[assignment], '+='): value = UNKNOWN
            if '.' in name:
                root, field = name.rsplit('.', 1); obj = self.variable(root)
                if isinstance(obj, Object): obj.fields[field] = value
            if name != '$null': self.env[name] = value
            return False
        self.expr(items)
        return False

    def invoke(self, target, args):
        if isinstance(target, Block):
            params = _params(target.items[1].items) if len(target.items) > 1 and _is(target.items[0], 'param') and isinstance(target.items[1], Group) else ()
            return self.call(Function(params, target.items), args, 'scriptblock')
        if isinstance(target, str) and target.lower() in self.functions:
            return self.call(self.functions[target.lower()], args, target)
        self.native(target, args)

    def call(self, function, args, name):
        if isinstance(function, Alternatives):
            for variant in function.values:
                if isinstance(variant, Function): self.call(variant, args, name)
                else: self.finding('conditionally unresolved function ' + name)
            return
        key = (function, tuple(str(a) for a in args))
        if key in self.active:
            self.finding('recursive function execution ' + name); return
        self.active.add(key)
        saved_env, saved_functions = self.env.copy(), self.functions.copy()
        old_chain = self.chain
        self.chain = (*self.chain, 'PowerShell function ' + name)
        for index, (param, _, source) in enumerate(function.params):
            default = next((i for i, t in enumerate(source) if _is(t, '=')), None)
            self.env[param] = args[index] if index < len(args) else self.value(source[default + 1:]) if default is not None else UNKNOWN
        self.block(function.body)
        # Script/env variables and generated artifacts persist across calls.
        retained = {k: v for k, v in self.env.items() if k.startswith(('$script:', '$env:'))}
        self.env, self.functions, self.chain = saved_env, saved_functions, old_chain
        self.env.update(retained)
        self.active.remove(key)

    def expr(self, items):
        if not items: return
        if any(_is(items[0], word) for word in ('if', 'foreach', 'while', 'try', 'for', 'do')):
            self.statement(items); return
        for pipeline in _split(items, {'|', '&&', '||'}):
            if not pipeline: continue
            # Inspect nested expressions, hashtable fields and interpolations.
            for item in pipeline:
                if isinstance(item, Group):
                    if item.kind == '@{':
                        for field in _split(item.items, {';', '\n'}):
                            at = next((i for i, t in enumerate(field) if _is(t, '=')), None)
                            self.expr(field[at + 1:] if at is not None else field)
                    elif item.kind in ('(', '$(', '@('):
                        self.expr(item.items)
                    elif item.kind == '{':
                        # Cmdlet callbacks and Add-Member methods can execute later;
                        # scan their bodies conservatively in their defining scope.
                        self.block(item.items)
                elif item.kind == 'double': self.string(item)
            for position, item in enumerate(pipeline):
                if not isinstance(item, Token) or not item.adjacent_call or position + 1 >= len(pipeline) or not isinstance(pipeline[position + 1], Group) or pipeline[position + 1].kind != '(':
                    continue
                if '::' in item.value:
                    typ, method = item.value.split('::', 1)
                    self.method(typ[1:-1].lower(), method, pipeline[position + 1].items)
                elif re.fullmatch(r'\$[\w:]+(?:\.[\w]+)+', item.value):
                    owner, method = item.value.rsplit('.', 1)
                    self.method(self.variable(owner), method, pipeline[position + 1].items, static=False)
                elif re.fullmatch(r'\.[A-Za-z_]\w*', item.value):
                    self.method(UNKNOWN, item.value[1:], pipeline[position + 1].items, static=False)
            head = pipeline[0]
            if isinstance(head, Group): continue
            if head.kind != 'word' and not _is(head, '&') and not _is(head, '.'):
                continue
            word = head.value.lower()
            if _is(head, '&') or _is(head, '.'):
                if len(pipeline) < 2:
                    self.finding('missing invocation target'); continue
                target = self.value(pipeline[1:2])
                # A cast/string concatenation in invocation position remains unknown.
                self.invoke(target, self.arguments(pipeline[2:])); continue
            if word in ('param', 'throw', 'return', 'exit', 'continue', 'break', '-not', '-and', '-or') or word.startswith('[') and not '::' in word:
                continue
            if word in self.functions:
                self.call(self.functions[word], self.arguments(pipeline[1:]), word); continue
            if word == 'start-process':
                args = list(pipeline[1:]); exe, argv = UNKNOWN, []
                for i, token in enumerate(args[:-1]):
                    if _is(token, '-filepath'): exe = self.value(args[i + 1:i + 2])
                    if _is(token, '-argumentlist'):
                        value = self.value(args[i + 1:i + 2]); argv = value if isinstance(value, list) else [value]
                self.native(exe, argv); continue
            if word == 'add-type':
                args = list(pipeline[1:]); output, source = UNKNOWN, UNKNOWN
                for i, token in enumerate(args[:-1]):
                    if _is(token, '-outputassembly'): output = self.value(args[i + 1:i + 2])
                    if _is(token, '-typedefinition'): source = self.value(args[i + 1:i + 2])
                if self.csharp(source): self.generated[str(output)] = ('native', source)
                continue
            if word in ('copy-item',):
                args = list(pipeline[1:]); src, dst = UNKNOWN, UNKNOWN
                for i, token in enumerate(args[:-1]):
                    if _is(token, '-literalpath') or _is(token, '-path'): src = self.value(args[i + 1:i + 2])
                    if _is(token, '-destination'): dst = self.value(args[i + 1:i + 2])
                if str(src) in self.generated: self.generated[str(dst)] = self.generated[str(src)]
            if word in self.CMDLETS:
                self.graph.finding('ordinary', self.chain, 'PowerShell metadata cmdlet ' + word); continue
            if head.adjacent_call and re.match(r'\$[\w:]+(?:\.[\w]+)+$', word) and len(pipeline) > 1 and isinstance(pipeline[1], Group):
                continue
            if head.adjacent_call and '::' in word and len(pipeline) > 1 and isinstance(pipeline[1], Group):
                continue
            if word.startswith('$') or word.startswith('-') or re.fullmatch(r'\d+', word) or word.startswith('.'):
                continue  # data expressions; executable variable invocation requires &/.
            if '::' in word:
                typ, prop = word.split('::', 1)
                if typ[1:-1] in self.TYPES and re.fullmatch(r'[a-z_][a-z_0-9]*', prop): continue
            # A bare unresolved command is an executable edge, including .ps1/.py.
            self.native(self.value(pipeline[:1]), self.arguments(pipeline[1:]))

    def method(self, owner, method, items, static=True):
        name = method.lower()
        args = [self.value(part) for part in _split(items, {','})]
        if name == 'start':
            if isinstance(owner, Object) and owner.kind == 'process':
                info = owner.fields.get('startinfo')
                if isinstance(info, Object):
                    self.native(info.fields.get('filename', UNKNOWN), self.argv(info.fields.get('arguments', UNKNOWN)))
                else: self.finding('unresolved Process.StartInfo')
            elif isinstance(owner, str) and owner in ('diagnostics.process', 'system.diagnostics.process'):
                self.native(args[0] if args else UNKNOWN, self.argv(args[1] if len(args) > 1 else UNKNOWN))
            else: self.finding('unresolved executable method Start')
            return
        if name == 'writealltext' and owner in ('io.file', 'system.io.file') and len(args) >= 2:
            if isinstance(args[0], str) and isinstance(args[1], str): self.generated[args[0]] = ('text', args[1])
        if isinstance(owner, str) and owner.startswith('['): owner = owner.strip('[]').lower()
        if name == 'new' and isinstance(owner, str) and owner in self.TYPES: return
        if name not in self.METHODS or static and isinstance(owner, str) and owner not in self.TYPES:
            self.finding('unresolved method ' + str(owner) + '.' + method)

    def argv(self, value):
        if isinstance(value, list): return value
        if not isinstance(value, str): return [UNKNOWN]
        # Windows command-line quoting is sufficient for the fixed argument
        # arrays joined by ProcessStartInfo; unresolved pieces are retained.
        return [a or b for a, b in re.findall(r'"([^"]*)"|([^\s"]+)', value)]

    def native(self, exe, args):
        if not isinstance(exe, str):
            self.finding('unresolved invocation target'); return
        args = [str(part) for arg in args for part in (arg if isinstance(arg, list) else [arg])]
        exe = exe.strip('"'); leaf = exe.replace('\\', '/').rsplit('/', 1)[-1].lower()
        generated = self.generated.get(exe)
        if isinstance(generated, Alternatives):
            for variant in generated.values:
                if isinstance(variant, tuple):
                    self.generated[exe] = variant
                    self.native(exe, args)
                else: self.finding('conditionally unresolved generated source ' + exe)
            self.generated[exe] = generated
            return
        if generated:
            if generated[0] == 'native':
                self.graph.finding('ordinary', self.chain, 'source-inspected generated native fixture'); return
            if leaf.endswith(('.cmd', '.bat')): self.cmd(generated[1]); return
            if leaf.endswith('.ps1'):
                return self.generated_script(exe, generated[1], args)
            self.finding('unsupported generated executable source ' + exe); return
        if leaf in ('powershell', 'powershell.exe', 'pwsh', 'pwsh.exe'):
            lowered = [x.lower() for x in args]
            if '-file' in lowered:
                at = lowered.index('-file')
                if at + 1 >= len(args): self.finding('missing PowerShell -File source'); return
                self.native(args[at + 1], args[at + 2:]); return
            if '-command' in lowered or '-c' in lowered:
                at = lowered.index('-command') if '-command' in lowered else lowered.index('-c')
                if at + 1 < len(args): self.generated_script('<inline>', args[at + 1], args[at + 2:])
                else: self.finding('missing PowerShell command source')
                return
            self.finding('PowerShell invocation without resolved source'); return
        if leaf in ('cmd', 'cmd.exe'):
            match = next((i for i, a in enumerate(args) if a.lower() == '/c'), None)
            if match is None: self.finding('cmd invocation without source'); return
            self.cmd(' '.join(args[match + 1:])); return
        if leaf in ('cargo', 'cargo.exe') and args:
            candidate = 'cargo-' + args[0] + '.exe'
            prefix = str(self.env.get('$env:path', '')).split(';')[0]
            source = self.generated.get(prefix.rstrip('/\\') + '/' + candidate)
            if isinstance(source, tuple) and source[0] == 'native':
                self.graph.finding('ordinary', self.chain, 'Cargo external subcommand with source-inspected generated executable'); return
        if UNKNOWN in exe or exe.startswith('__PS_'):
            self.finding('unresolved executable ' + exe); return
        if exe.endswith('.ps1'):
            path = (self.base / exe).resolve() if not Path(exe).is_absolute() else Path(exe).resolve()
            if path.is_file() and path.is_relative_to(self.graph.root):
                self.generated_script(str(path), path.read_text(encoding='utf-8'), args); return
        self.graph.command([exe, *args], self.chain, self.base)

    def generated_script(self, path, text, args):
        key = (path, tuple(args))
        if key in self.active:
            self.finding('recursive generated script ' + path); return
        self.active.add(key)
        child = Analyzer(self.graph, (*self.chain, 'PowerShell source ' + path), Path(path).parent if path != '<inline>' else self.base)
        child.generated = self.generated
        child.active = self.active
        child.analyze(text, args)
        self.active.remove(key)

    def cmd(self, source):
        # Resolve the quoted generated batch used to import an environment. It is
        # still inspected; an injected command in its body is not exempted.
        for path, entry in self.generated.items():
            if not isinstance(entry, tuple): continue
            kind, text = entry
            if kind == 'text' and path in source and path.lower().endswith(('.bat', '.cmd')):
                residue = source.replace(path, '').replace('"', '').strip()
                if re.fullmatch(r'(?:>nul)?\s*(?:&&\s*set)?', residue, re.I):
                    self.cmd(text); return
        lines, current, quoted = [], '', False
        for char in source:
            if char == '"': quoted = not quoted
            if not quoted and char in '\r\n&|':
                if current.strip(): lines.append(current)
                current = ''
            else: current += char
        if current.strip(): lines.append(current)
        if quoted:
            self.finding('unbalanced cmd quoting'); return
        for line in lines:
            line = line.strip().lstrip('@')
            if not line: continue
            if re.fullmatch(r'(?:echo\s+off|exit(?:\s+/b)?\s+[-+]?\d+|set(?:\s+"[^"\r\n]*"|\s+[^&|<>"\r\n]+)?)', line, re.I):
                continue
            # Quoting damage is not proof of execution safety.
            if UNKNOWN in line or '__PS_' in line:
                self.finding('unresolved cmd source ' + line)
            else: self.graph.shell(line, (*self.chain, 'cmd source'), self.base, windows=True)

    def csharp(self, source):
        if not isinstance(source, str) or UNKNOWN in source or '__PS_' in source:
            self.finding('unresolved Add-Type source'); return False
        code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', ' ', source, flags=re.S)
        if re.search(r'\b(?:unsafe|extern|delegate|dynamic|typeof|assembly|DllImport|GetType|Activator|Reflection|Process|Assembly|Task|ThreadPool)\b', code):
            self.finding('unsupported generated C# execution source'); return False
        namespaces = re.findall(r'\busing\s+([\w.]+)\s*;', code)
        if any(n not in ('System', 'System.IO', 'System.Threading') for n in namespaces):
            self.finding('unsupported generated C# namespace'); return False
        # Only these data/pipe/time APIs are source-proven non-inference. All
        # other calls, including renamed model wrappers, remain unresolved.
        allowed = {'Main', 'if', 'while', 'for', 'Console.OpenStandardOutput', 'Console.OpenStandardError',
                   'DateTime.UtcNow.AddSeconds', 'File.Exists', 'Thread.Sleep'}
        streams = set(re.findall(r'\b(?:var|Stream)\s+(\w+)\s*=\s*Console\.OpenStandard(?:Output|Error)\s*\(', code))
        calls = re.findall(r'\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(', code)
        if any(call not in allowed and not (call.endswith('.Write') and call[:-6] in streams) for call in calls):
            self.finding('unsupported generated C# call'); return False
        # Constructors are executable too. Arrays of primitive bytes are the only
        # construction admitted by this bounded source grammar.
        if re.search(r'\bnew\s+(?!byte\s*\[)', code):
            self.finding('unsupported generated C# constructor'); return False
        self.graph.finding('ordinary', self.chain, 'source-inspected C# pipe/time metadata fixture')
        return True


def analyze(graph, text, chain=(), base=None, args=()):
    """Append findings to a RunGraph using source only; never execute PowerShell."""
    Analyzer(graph, chain, base).analyze(text, args)
