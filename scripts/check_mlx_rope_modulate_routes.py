#!/usr/bin/env python3
"""SW-6140 route gate: no private RoPE / adaLN-modulate arithmetic in the mlx-gen provider crates.

mlx-gen carries ONE shared RoPE implementation (`mlx_gen::qkv::apply_rope` / `apply_rope_eager` /
`apply_rope_1d`, over the `nn::rope_rotate` / `nn::rope_rotate_eager` / `nn::rotate_pairwise` /
`nn::rotate_half` kernels and `nn::apply_text_rope`) and ONE shared modulate implementation
(`mlx_gen::nn::modulate` / `modulate_eager` / `scale_shift` and the provider glue bodies
`modulate_matching_one_impl` / `modulate_f32_one_impl`). The family crates used to open-code both.

This is a pure source scan (no build, no MLX). Every function in a provider crate's `src/` whose
name says it rotates or modulates (`rope`, `rotary`, `rotate`, `modulat`, or an `apply` method in a
`*rope*.rs` module) must not do raw elementwise arithmetic (`add`/`subtract`/`multiply`/
`negative`/`addmm`) itself: it delegates to the shared helpers, or only builds tables/positions
without those ops. The few justified exceptions are listed, with a reason, in
`crates/media/mlx-gen/rope_modulate_routes.json`, next to the per-crate route table (which route
each crate's RoPE and modulation take). A new private rotation or modulation therefore fails this
gate until it is routed through the shared helpers or recorded with a reason; an exception that no
longer matches (renamed, removed, or no longer arithmetic) fails as stale, so the table cannot
drift from the code. `#[cfg(test)]` items and `tests.rs` / `*_tests.rs` files are not scanned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX_GEN = ROOT / "crates" / "media" / "mlx-gen"
TABLE = MLX_GEN / "rope_modulate_routes.json"
EXCEPTION_ROUTES = ("table", "kept")

NAME = re.compile(r"rope|rotary|rotate|modulat", re.IGNORECASE)
FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]")
CANDIDATE = re.compile(r"\bfn\s+(?:\w*(?:rope|rotary|rotate|modulat)\w*|apply)\b", re.IGNORECASE)
CFG_TEST = re.compile(r"#\[cfg\((?:all\()?test\b")
# Raw elementwise arithmetic: the ops a private rotation / modulation is built from.
ARITHMETIC = re.compile(r"(?<![A-Za-z0-9_])(add|subtract|multiply|negative|addmm)\s*\(")


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
    """Spans of `impl` blocks whose self type is named like a RoPE (`impl Rope`, `impl MmRope`)."""
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
    base = root / "crates" / "media" / "mlx-gen"
    found: dict[tuple[str, str], dict] = {}
    for path in sorted(base.glob("mlx-gen-*/src/**/*.rs")):
        if path.stem == "tests" or path.stem.endswith("_tests"):
            continue  # `#[cfg(test)] mod tests;` bodies live in these files
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
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
    base = root / "crates" / "media" / "mlx-gen"
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
                "mlx_gen::qkv / mlx_gen::nn RoPE and modulate helpers, or record a justified "
                f"exception in {TABLE.relative_to(ROOT).as_posix()}"
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
            f"mlx-gen RoPE/modulate routes OK ({len(data['crates'])} crates, "
            f"{len(data['exceptions'])} recorded exceptions)"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
