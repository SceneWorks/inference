#!/usr/bin/env python3
"""SW-6129 route gate: no private RoPE / adaLN-modulate arithmetic in the candle-gen provider crates.

candle-gen carries ONE shared RoPE implementation (`candle_gen::rope::{apply_rope,
apply_rope_in_place, rms_norm_rope, rms_norm_rope_in_place, apply_rope_freqs_cis}` over a
`RopeSpec`, plus `candle_gen::apply_interleaved_rope`) and ONE shared modulate implementation
(`candle_gen::modulate::{modulate, modulate_with}` composable, `adaln` / `rms_adaln` /
`layer_adaln` fused) - SW-6127 / SW-6128 / SW-6129, feature SW-F287. The family crates used to
open-code both.

This is a pure source scan (no build, no CUDA), the candle twin of
`scripts/check_mlx_rope_modulate_routes.py` (SW-6140). Every function in a provider crate's `src/`
whose name says it rotates or modulates (`rope`, `rotary`, `rotate`, `modulat`, or an `apply`
method in a `*rope*.rs` module / on a RoPE-named type) must not do raw tensor arithmetic itself
(`broadcast_mul` / `broadcast_add` / `broadcast_sub` / `mul` / `add` / `sub` / `neg` / `affine`, or
a `?`-chained `+ - *` operator): it delegates to the shared helpers, or only builds
tables/positions. The justified exceptions (trainer twins that must keep their composable ops -
epic E9 -, table builders, conventions the shared helpers do not express) are listed with a
reason in `crates/media/candle-gen/rope_modulate_routes.json`, next to the per-crate route table.
A new private rotation or modulation therefore fails this gate until it is routed through the
shared helpers or recorded with a reason; an exception that no longer matches (renamed, removed, or
no longer arithmetic) fails as stale, so the table cannot drift from the code. `#[cfg(test)]`
items and `tests.rs` / `*_tests.rs` files are not scanned.

The adaLN affine is also caught **inline**, whatever function it sits in: a
`.broadcast_mul(&(<scale> + 1.0)?)?.broadcast_add(<shift>)` chain is the private modulate under
another name, reported as `<file>::<inline modulate>` (an exception row may name that pseudo-fn for
a file whose inline modulations must stay, e.g. a trainer). A scale-only `norm·(1 + scale)` with no
shift is not the modulate affine and is not flagged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDLE_GEN = ROOT / "crates" / "media" / "candle-gen"
TABLE = CANDLE_GEN / "rope_modulate_routes.json"
EXCEPTION_ROUTES = ("trainer", "table", "kept")
INLINE_MODULATE_FN = "<inline modulate>"

NAME = re.compile(r"rope|rotary|rotate|modulat", re.IGNORECASE)
FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]")
CANDIDATE = re.compile(r"\bfn\s+(?:\w*(?:rope|rotary|rotate|modulat)\w*|apply)\b", re.IGNORECASE)
CFG_TEST = re.compile(r"#\[cfg\((?:all\()?test\b")
# Raw tensor arithmetic: the ops a private rotation / modulation is built from - candle's
# elementwise methods / free functions, and the `?`-chained operator overloads
# (`a.broadcast_mul(c)? + b`, `(scale + 1.0)?`).
ARITHMETIC = re.compile(
    r"(?<![A-Za-z0-9_])(broadcast_mul|broadcast_add|broadcast_sub|mul|add|sub|neg|affine)\s*\("
    r"|\?\s*[-+*]\s*[&(A-Za-z]"
    r"|[-+*]\s*\d+(?:\.\d*)?\s*\)\s*\?"
)
# The adaLN affine written inline: `x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)`.
INLINE_MODULATE = re.compile(
    r"\.broadcast_mul\(\s*&\s*\(\s*[^;]*?\+\s*1(?:\.0*)?\s*\)\s*\?\s*\)\s*\?\s*\.broadcast_add\("
)


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments, string and char literals, preserving offsets and newlines."""
    out: list[str] = []
    i, n = 0, len(text)
    blank = lambda s: re.sub(r"[^\n]", " ", s)  # noqa: E731
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(blank(text[i:j]))
            i = j
            continue
        raw = re.match(r"b?r(#*)\"", text[i : i + 260])
        if raw and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            close = '"' + raw.group(1)
            j = text.find(close, i + len(raw.group(0)))
            j = n if j < 0 else j + len(close)
            out.append(blank(text[i:j]))
            i = j
            continue
        if text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(n, j + 1)
            out.append(blank(text[i:j]))
            i = j
            continue
        char = re.match(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'", text[i : i + 12])
        if char:
            out.append(" " * len(char.group(0)))
            i += len(char.group(0))
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def matching_brace(code: str, open_at: int) -> int:
    depth = 0
    for k in range(open_at, len(code)):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced braces")


def test_spans(code: str) -> list[tuple[int, int]]:
    """Byte spans of `#[cfg(test)]` items (a braced item, or one ending at `;`)."""
    spans = []
    for m in CFG_TEST.finditer(code):
        j = m.end()
        semi, brace = code.find(";", j), code.find("{", j)
        if brace < 0 or 0 <= semi < brace:
            spans.append((m.start(), semi + 1))
        else:
            spans.append((m.start(), matching_brace(code, brace) + 1))
    return spans


IMPL = re.compile(r"\bimpl\b(?:\s*<[^{;]*?>)?\s+(?:[A-Za-z0-9_:<>, ]+\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)[^{;]*\{")


def rope_impl_spans(code: str) -> list[tuple[int, int]]:
    """Spans of `impl` blocks whose self type is named like a RoPE (`impl Rope`, `impl RopeTable`)."""
    spans = []
    for m in IMPL.finditer(code):
        if NAME.search(m.group(1)):
            spans.append((m.start(), matching_brace(code, m.end() - 1)))
    return spans


def functions(text: str) -> list[tuple[str, str, bool]]:
    """`(name, body, in_rope_impl)` of every non-test fn with a body (comments/strings blanked)."""
    code = strip_comments_and_strings(text)
    tests = test_spans(code)
    rope_impls = rope_impl_spans(code)
    found = []
    for m in FN.finditer(code):
        if any(a <= m.start() < b for a, b in tests):
            continue
        # The body opens at the first `{` after the signature unless a `;` (a declaration) comes
        # first. Skip `{` inside the signature's generics/where clause by tracking `<>`/`()` depth.
        depth, k = 0, m.end() - 1
        while k < len(code):
            c = code[k]
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                depth -= 1
            elif c == "-" and code.startswith("->", k):
                k += 2
                continue
            elif depth <= 0 and c == ";":
                k = -1
                break
            elif depth <= 0 and c == "{":
                break
            k += 1
        if k < 0 or k >= len(code):
            continue
        end = matching_brace(code, k)
        in_rope_impl = any(a <= m.start() < b for a, b in rope_impls)
        found.append((m.group(1), code[k : end + 1], in_rope_impl))
    return found


def in_scope(path: Path, name: str, in_rope_impl: bool = False) -> bool:
    """A fn named like a rotation/modulation, or an `apply` of a RoPE type / `*rope*.rs` module."""
    if NAME.search(name):
        return True
    return name == "apply" and (in_rope_impl or "rope" in path.stem.lower())


def scan(root: Path = ROOT) -> dict[tuple[str, str], dict]:
    """Every in-scope fn in a provider crate's `src/`: `(relative path, name) -> facts`."""
    base = root / "crates" / "media" / "candle-gen"
    found: dict[tuple[str, str], dict] = {}
    for path in sorted(base.glob("candle-gen-*/src/**/*.rs")):
        if path.stem == "tests" or path.stem.endswith("_tests"):
            continue  # `#[cfg(test)] mod tests;` bodies live in these files
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        code = strip_comments_and_strings(text)
        tests = test_spans(code)
        for m in INLINE_MODULATE.finditer(code):
            if not any(a <= m.start() < b for a, b in tests):
                entry = found.setdefault((rel, INLINE_MODULATE_FN), {"count": 0, "arithmetic": True})
                entry["count"] += 1
        if not CANDIDATE.search(text):
            continue  # cheap pre-filter: no fn here can be in scope
        for name, body, in_rope_impl in functions(text):
            if not in_scope(path, name, in_rope_impl):
                continue
            entry = found.setdefault((rel, name), {"count": 0, "arithmetic": False})
            entry["count"] += 1
            entry["arithmetic"] |= bool(ARITHMETIC.search(body))
    return found


def load_table(path: Path = TABLE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("version") != 1
        or not isinstance(data.get("crates"), dict)
        or not isinstance(data.get("exceptions"), list)
    ):
        raise ValueError(f"{path}: expected {{'version': 1, 'crates': {{...}}, 'exceptions': [...]}}")
    return data


def check(root: Path = ROOT, table: dict | None = None) -> list[str]:
    data = load_table(root / TABLE.relative_to(ROOT)) if table is None else table
    errors: list[str] = []
    base = root / "crates" / "media" / "candle-gen"
    for crate, row in sorted(data["crates"].items()):
        if not (base / crate / "src").is_dir():
            errors.append(f"route table names a crate that does not exist: {crate}")
        for field in ("rope", "modulate"):
            if len(str(row.get(field, "")).strip()) < 4:
                errors.append(f"{crate}: route table row needs a `{field}` route")
    found = scan(root)
    excepted: dict[tuple[str, str], dict] = {}
    for row in data["exceptions"]:
        key = (row.get("file", ""), row.get("fn", ""))
        if key in excepted:
            errors.append(f"duplicate exception: {key[0]}::{key[1]}")
            continue
        excepted[key] = row
        if row.get("route") not in EXCEPTION_ROUTES:
            errors.append(f"{key[0]}::{key[1]}: route must be one of {EXCEPTION_ROUTES}")
        if len(str(row.get("reason", "")).strip()) < 20:
            errors.append(f"{key[0]}::{key[1]}: an exception needs a reason (>= 20 chars)")
        facts = found.get(key)
        if facts is None:
            errors.append(f"stale exception (fn not found): {key[0]}::{key[1]}")
        elif not facts["arithmetic"]:
            errors.append(
                f"stale exception (no private arithmetic left - delete the row): {key[0]}::{key[1]}"
            )
    crates_seen = set()
    for (rel, name), facts in sorted(found.items()):
        crates_seen.add(rel.split("/")[3])
        if facts["arithmetic"] and (rel, name) not in excepted:
            errors.append(
                f"{rel}::{name}: private rotation/modulation arithmetic - route it through the shared "
                "candle_gen::rope / candle_gen::modulate helpers, or record a justified exception in "
                f"{TABLE.relative_to(ROOT).as_posix()}"
            )
    for crate in sorted(crates_seen - set(data["crates"])):
        errors.append(f"{crate}: has RoPE/modulate functions but no route-table row")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", action="store_true", help="print the scanned functions")
    args = parser.parse_args()
    if args.inventory:
        for (rel, name), facts in sorted(scan().items()):
            print(f"{rel}\t{name}\t{'arith' if facts['arithmetic'] else '-'}")
        return 0
    errors = check()
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        data = load_table()
        print(
            f"candle-gen RoPE/modulate routes OK ({len(data['crates'])} crates, "
            f"{len(data['exceptions'])} recorded exceptions)"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
