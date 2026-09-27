#!/usr/bin/env python3
"""Inline view of `.github/workflows/real-weights.yml` with its externalized step bodies restored.

GitHub refuses a workflow file over 512,000 bytes ("Workflow file exceeds the maximum allowed size
of 500 KB"), and every run of that file then startup-fails. `real-weights.yml` reached 509,886
bytes, so the larger step bodies now live in `scripts/ci/real-weights/<job>/<step>.<ext>` and the
step's `run:` executes the file in the step's own shell:

* default macOS shell (`bash -e`): ``run: source scripts/ci/real-weights/<job>/<step>.sh``
* ``shell: cmd``: ``run: call scripts\\ci\\real-weights\\<job>\\<step>.cmd``
* ``shell: powershell``: ``run: . ./scripts/ci/real-weights/<job>/<step>.ps1``

Each form runs the body in the step's shell process itself (``source`` / ``call`` / dot-sourcing),
so shell options, error handling, variables and exit status behave exactly as they did inline. The
file content is byte-for-byte the former `run:` scalar.

`inline_text()` reverses the indirection and returns the workflow text as if every body were still
inline, so policy tests that read step bodies keep reading what the runner executes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "real-weights.yml"
SCRIPT_DIR = "scripts/ci/real-weights"

_REFERENCE = re.compile(
    r"^(?P<lead>\s*(?:- )?)run: (?:"
    r"source (?P<sh>scripts/ci/real-weights/[^\s\\]+\.sh)"
    r"|call (?P<cmd>scripts\\ci\\real-weights\\[^\s/]+\.cmd)"
    r"|\. \./(?P<ps1>scripts/ci/real-weights/[^\s\\]+\.ps1)"
    r")$"
)


def script_references(text: str) -> list[str]:
    """Repository-relative POSIX paths of every externalized step body, in file order."""
    references = []
    for line in text.split("\n"):
        match = _REFERENCE.match(line)
        if match:
            path = match["sh"] or match["ps1"] or match["cmd"].replace("\\", "/")
            references.append(path)
    return references


def inline_text(text: str | None = None, root: Path = ROOT) -> str:
    """Return the workflow text with each externalized body restored as a `run: |` block."""
    if text is None:
        text = WORKFLOW.read_text(encoding="utf-8")
    out = []
    for line in text.split("\n"):
        match = _REFERENCE.match(line)
        if not match:
            out.append(line)
            continue
        path = match["sh"] or match["ps1"] or match["cmd"].replace("\\", "/")
        body = (root / path).read_text(encoding="utf-8").replace("\r\n", "\n")
        if not body.endswith("\n") or body.endswith("\n\n"):
            raise ValueError(f"{path} must end in exactly one newline (a `run: |` block)")
        indent = " " * (len(match["lead"]) + 2)
        out.append(f"{match['lead']}run: |")
        out.extend(indent + body_line if body_line else "" for body_line in body[:-1].split("\n"))
    return "\n".join(out)


def main() -> int:
    sys.stdout.write(inline_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
