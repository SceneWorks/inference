#!/usr/bin/env python3
"""Find assertions whose truth depends on a WALL CLOCK.

WHY THIS EXISTS (sc-19556)
--------------------------
A test that asserts on `Instant::elapsed()` is asserting on how busy the machine was. Two
distinct defects hide under that, and this tool exists to enumerate both so they can be judged
one at a time rather than discovered by a red lane on an unrelated PR:

* **Degenerate.** The condition cannot fail for any implementation. `first < total` where
  `total` is sampled after the call that fires the `first` callback; `tps > 0.0` where `tps` is
  `n / elapsed()`. These are green for a correct implementation AND for one returning garbage,
  so they are not gates at all and their presence hides that the thing they name is ungated.
* **Contention-bound.** The condition encodes a real claim but reads it through a clock, on a
  box that routinely runs several builds at once. Whichever arm gets descheduled loses, and the
  failure is charged to whatever PR happens to be in flight.

The fix differs per site and this tool does not attempt it. Some assertions have a clock-free
equivalent that is strictly better (count the progress callbacks; read the KV cache's own
offset; compare the emitted chunks against the returned track). Some are genuinely latency
claims and must keep a duration — those want a `min` over N identical runs rather than a single
sample, because contention can only ever make a run SLOWER, so the fastest of several is a
lower bound on the hardware instead of a draw from a noisy distribution.

WHAT IT DOES
------------
A three-pass taint analysis per file:

1. **Seed.** Any binding whose initializer mentions a clock token (`Instant::now`, `.elapsed()`,
   `Duration::from_*`, `.as_secs_f64()`, `.as_millis()`, `duration_since`, ...) is tainted.
2. **Propagate** to a fixpoint: a binding whose initializer mentions a tainted name is tainted.
   Tuple destructuring taints every name bound.
3. **Report** every `assert!` / `assert_eq!` / `assert_ne!` / `debug_assert*` whose CONDITION
   mentions a clock token or a tainted name. The condition is the first argument (or first two,
   for the binary forms) split on top-level commas — the format-message arguments are excluded
   deliberately, since printing a duration in a failure message is fine and common.

KNOWN LIMITATIONS — read these before quoting a "zero remaining" number
-----------------------------------------------------------------------
* **A duration that crosses a channel, `HashMap`, `OnceLock`, struct field or any other
  container defeats the fixpoint IF it is converted to a plain number on the way.** The
  analysis is name-based, so it needs a token to follow. This still reds:

      let d = rx.recv().unwrap();
      assert!(d.as_millis() < 500, "decode step regressed");   // caught: `as_millis()`

  but this does NOT, because nothing in the assertion or its binding names a clock:

      let ms: u64 = rx.recv().unwrap();   // sender did `.as_millis()`
      assert!(ms < 500, "decode step regressed");              // MISSED

  The same hole applies to a duration stashed in a struct field and read back elsewhere, and to
  one passed through a function boundary (see below). Treat a clean run as "no site the token
  reaches", never as "no clock assertion exists".
* **Scope is the FILE, not the function.** Taint does not cross function boundaries (a helper
  returning a duration taints its callers only if the call site names a clock token), and two
  functions in one file that reuse a variable name share taint. The first causes misses; the
  second causes false positives.
* **Comments and string literals are not stripped.** A commented-out clock assertion or a
  message mentioning `elapsed` can produce a false positive. That is deliberate: this is a
  triage aid whose output is meant to be read, and under-reporting is the worse failure.

So: every hit needs a human decision, and a clean file is evidence but not proof.

USAGE
-----
    scripts/check_clock_assertions.py                 # whole repo, human-readable
    scripts/check_clock_assertions.py crates/llm      # a subtree
    scripts/check_clock_assertions.py --summary       # counts per file, no detail

VALIDATION (sc-19556)
---------------------
The tool was validated by running it against the tree BEFORE the sc-19556 edits and confirming
that every site the story fixed by hand appears in its output, then re-running after. Reproduce
with:

    git stash                                          # or check out the pre-fix revision
    scripts/check_clock_assertions.py --summary
    git stash pop
    scripts/check_clock_assertions.py --summary

Exit status is 0 when no assertion is flagged and 1 otherwise, so it can gate a lane — but note
the limitations above before wiring it to one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Tokens that introduce a wall clock. `Duration::from_*` is included because a literal duration
# compared against a measured one is still a clock assertion on the measured side.
CLOCK_TOKENS = (
    "Instant::now",
    "SystemTime::now",
    ".elapsed()",
    "duration_since",
    "Duration::from",
    "Duration::ZERO",
    ".as_secs_f64()",
    ".as_secs_f32()",
    ".as_secs()",
    ".as_millis()",
    ".as_micros()",
    ".as_nanos()",
    ".subsec_",
)

# Accessors and fields that CARRY a duration without naming one of the tokens above. A measured
# duration very often reaches an assertion through a struct — `row.wall`, `stats.fastest_step()`,
# `p.decode_ms` — and a purely token-based fixpoint loses the trail at that boundary, which is the
# `duration crossing a container` limitation in this file's header. Matching the SHAPE of the
# accessor recovers the common cases at the cost of some false positives, which is the right trade
# for a triage aid: a missed clock assertion is worse than a hit a human dismisses.
DURATION_SHAPED = re.compile(
    r"\b\w*(?:elapsed|duration|latency|fastest_step|slowest_step|_ms|_secs|_seconds|wall)\w*\b",
    re.IGNORECASE,
)

ASSERT_MACROS = (
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
)

# `let x =`, `let mut x =`, `let (a, b) =`, `let Some(x) =`
LET_RE = re.compile(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*|\([^)]*\))\s*(?::[^=]+)?=")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=[^=]")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MACRO_RE = re.compile(r"\b(" + "|".join(ASSERT_MACROS) + r")!\s*\(")


def has_clock_token(text: str) -> bool:
    return any(tok in text for tok in CLOCK_TOKENS) or bool(DURATION_SHAPED.search(text))


def names_bound(target: str) -> list[str]:
    """Names introduced by a `let` pattern (handles tuple/struct destructuring loosely)."""
    return IDENT_RE.findall(target)


def macro_args(src: str, open_paren: int) -> tuple[list[str], int]:
    """Split a macro's argument list on TOP-LEVEL commas.

    `open_paren` indexes the `(`. Returns (args, index_after_closing_paren). Tracks nesting for
    (), [], {} and skips over string/char literals so a comma inside a format string or a
    generic does not split an argument.
    """
    depth = 0
    i = open_paren
    args: list[str] = []
    start = open_paren + 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"':
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == '"':
                    break
                i += 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(src[start:i])
                return args, i + 1
        elif c == "," and depth == 1:
            args.append(src[start:i])
            start = i + 1
        i += 1
    return args, n


def condition_of(macro: str, args: list[str]) -> str:
    """The argument(s) that decide the assertion, excluding the format message."""
    if not args:
        return ""
    if macro.endswith("_eq") or macro.endswith("_ne"):
        return " , ".join(args[:2])
    return args[0]


FN_RE = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*")


def fn_bodies(src: str) -> list[tuple[int, str]]:
    """Every `fn` body in `src` as (offset_of_body, body_text), brace-matched.

    Taint is scoped to one function so that two functions reusing a name (`t`, `peak`, `last`)
    do not contaminate each other — that was the dominant false-positive source when the
    analysis was file-scoped. Nested `fn`s are covered by their enclosing body, which is fine:
    the enclosing scope is a superset.
    """
    out: list[tuple[int, str]] = []
    n = len(src)
    for m in FN_RE.finditer(src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        depth = 0
        j = i
        while j < n:
            c = src[j]
            if c == '"':
                j += 1
                while j < n:
                    if src[j] == "\\":
                        j += 2
                        continue
                    if src[j] == '"':
                        break
                    j += 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((i, src[i : j + 1]))
    return out


def analyze(path: Path) -> list[tuple[int, str, str]]:
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if "assert" not in src:
        return []

    hits: list[tuple[int, str, str]] = []
    for base, body in fn_bodies(src):
        hits.extend(analyze_scope(src, base, body))
    # One `fn` may nest inside another, so the same assertion can be reported twice.
    seen: set[int] = set()
    unique: list[tuple[int, str, str]] = []
    for h in sorted(hits):
        if h[0] not in seen:
            seen.add(h[0])
            unique.append(h)
    return unique


def analyze_scope(src: str, base: int, body: str) -> list[tuple[int, str, str]]:
    lines = body.splitlines()

    # Passes 1 + 2: seed and propagate to a fixpoint.
    tainted: set[str] = set()
    for _ in range(10):
        grew = False
        for line in lines:
            m = LET_RE.search(line)
            rhs_start = m.end() if m else None
            if m:
                rhs = line[rhs_start:]
                if has_clock_token(rhs) or any(
                    t in IDENT_RE.findall(rhs) for t in tainted
                ):
                    for name in names_bound(m.group(1)):
                        if name not in tainted:
                            tainted.add(name)
                            grew = True
            else:
                a = ASSIGN_RE.match(line)
                if a:
                    rhs = line[a.end() - 1 :]
                    if has_clock_token(rhs) or any(
                        t in IDENT_RE.findall(rhs) for t in tainted
                    ):
                        if a.group(1) not in tainted:
                            tainted.add(a.group(1))
                            grew = True
        if not grew:
            break

    # Pass 3: report assertions whose CONDITION is clock-derived.
    hits: list[tuple[int, str, str]] = []
    for m in MACRO_RE.finditer(body):
        macro = m.group(1)
        args, _ = macro_args(body, m.end() - 1)
        cond = condition_of(macro, args)
        if not cond:
            continue
        idents = set(IDENT_RE.findall(cond))
        why = ""
        if has_clock_token(cond):
            why = "clock call in the condition"
        else:
            shared = idents & tainted
            if shared:
                why = "clock-derived binding: " + ", ".join(sorted(shared))
        if why:
            line_no = src.count("\n", 0, base + m.start()) + 1
            flat = " ".join(cond.split())
            if len(flat) > 100:
                flat = flat[:97] + "..."
            hits.append((line_no, flat, why))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="*", default=["."], help="paths to scan (default: repo root)")
    ap.add_argument("--summary", action="store_true", help="counts per file only")
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.roots:
        p = Path(root)
        if p.is_file() and p.suffix == ".rs":
            files.append(p)
        else:
            files.extend(sorted(p.rglob("*.rs")))

    files = [f for f in files if "target" not in f.parts]

    total = 0
    touched = 0
    for f in files:
        hits = analyze(f)
        if not hits:
            continue
        touched += 1
        total += len(hits)
        if args.summary:
            print(f"{f}: {len(hits)}")
            continue
        print(f"\n{f}")
        for line_no, cond, why in hits:
            print(f"  {line_no}: {cond}")
            print(f"      ^ {why}")

    print(
        f"\n{total} clock-dependent assertion(s) across {touched} file(s); "
        f"{len(files)} .rs file(s) scanned.",
        file=sys.stderr,
    )
    print(
        "Each hit needs a human decision: replace with a clock-free reading where one exists, "
        "or keep the duration and reduce it with `min` over N runs. See this file's header for "
        "what the analysis CANNOT see.",
        file=sys.stderr,
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
