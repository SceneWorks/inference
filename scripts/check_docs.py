#!/usr/bin/env python3
"""Check local links in repository-owned Markdown documentation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def markdown_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "README.md", "docs/**/*.md", "release/**/*.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / path for path in result.stdout.splitlines()]


def check_file(root: Path, document: Path) -> list[str]:
    errors = []
    for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
        for raw_target in LINK.findall(line):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(f"{document.relative_to(root)}:{line_number}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{document.relative_to(root)}:{line_number}: missing local link: {target}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = [error for document in markdown_files(root) for error in check_file(root, document)]
    if errors:
        print("\n".join(errors))
        return 1
    print("documentation links: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
