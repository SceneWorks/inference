#!/usr/bin/env python3
"""Validate the append-only repository-wide allocation of review finding ids."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("docs/code-review-finding-allocations.tsv")
FINDING_HEADING = re.compile(r"^#{1,6} \[F-(\d+)\]", re.MULTILINE)
IGNORED_TREE_PARTS = {".git", "target", ".claude", ".codex"}


class Allocation(NamedTuple):
    kind: str
    start: int
    end: int
    document: Path


class BaseRevisionUnavailable(Exception):
    """The requested comparison revision cannot be resolved to a commit."""

    def __init__(self, revision: str, returncode: int) -> None:
        super().__init__(revision)
        self.revision = revision
        self.returncode = returncode


# These four imported documents predate the repository-global sequence. Their
# overlapping allocations are historical evidence, so freeze rather than rewrite them.
LEGACY_ALLOCATIONS = (
    Allocation(
        "legacy", 1, 78, Path("crates/media/candle-gen/CODE_REVIEW_2026-07-01.md")
    ),
    Allocation(
        "legacy", 79, 163, Path("crates/media/candle-gen/CODE_REVIEW_2026-07-12.md")
    ),
    Allocation("legacy", 1, 150, Path("crates/media/mlx-gen/CODE_REVIEW_2026-07-01.md")),
    Allocation("legacy", 1, 181, Path("crates/media/mlx-gen/CODE_REVIEW_2026-07-11.md")),
)
LEGACY_HIGH_WATER = max(allocation.end for allocation in LEGACY_ALLOCATIONS)


def finding_ids(text: str) -> list[int]:
    """Return finding ids from canonical Markdown headings in document order."""
    return [int(match.group(1)) for match in FINDING_HEADING.finditer(text)]


def review_documents(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("**/CODE_REVIEW_*.md")
        if not (set(path.relative_to(root).parts) & IGNORED_TREE_PARTS)
    )


def parse_registry(text: str) -> tuple[list[Allocation], list[str]]:
    allocations: list[Allocation] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) != 4:
            errors.append(f"{REGISTRY}:{line_number} must contain four tab-separated fields")
            continue
        kind, start_text, end_text, document_text = fields
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            errors.append(f"{REGISTRY}:{line_number} has a non-integer allocation bound")
            continue
        if kind not in {"legacy", "review"}:
            errors.append(f"{REGISTRY}:{line_number} has unknown allocation kind {kind!r}")
        if start < 1 or end < start:
            errors.append(f"{REGISTRY}:{line_number} has invalid range {start}..{end}")
        document = Path(document_text)
        if document.is_absolute() or ".." in document.parts:
            errors.append(f"{REGISTRY}:{line_number} has unsafe document path {document_text!r}")
        allocations.append(Allocation(kind, start, end, document))
    return allocations, errors


def registry_at_revision(revision: str) -> str | None:
    if not revision or set(revision) == {"0"}:
        return None
    resolution = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if resolution.returncode != 0:
        raise BaseRevisionUnavailable(revision, resolution.returncode)
    result = subprocess.run(
        ["git", "show", f"{revision}:{REGISTRY.as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def validate(root: Path = ROOT, *, base_registry_text: str | None = None) -> list[str]:
    errors: list[str] = []
    registry_path = root / REGISTRY
    if not registry_path.is_file():
        return [f"missing allocation registry: {REGISTRY}"]

    registry_text = registry_path.read_text(encoding="utf-8")
    allocations, registry_errors = parse_registry(registry_text)
    errors.extend(registry_errors)
    if registry_errors:
        return errors

    if tuple(allocations[: len(LEGACY_ALLOCATIONS)]) != LEGACY_ALLOCATIONS:
        errors.append("allocation registry must begin with the exact four legacy ranges")

    if base_registry_text is not None:
        base_allocations, base_errors = parse_registry(base_registry_text)
        errors.extend(f"base {error}" for error in base_errors)
        if allocations[: len(base_allocations)] != base_allocations:
            errors.append("allocation registry is append-only; prior rows changed or were removed")

    expected_start = LEGACY_HIGH_WATER + 1
    documents: set[Path] = set()
    for index, allocation in enumerate(allocations):
        if index < len(LEGACY_ALLOCATIONS):
            continue
        if allocation.kind != "review":
            errors.append(f"post-legacy allocation for {allocation.document} must use kind 'review'")
        if allocation.start != expected_start:
            errors.append(
                f"allocation for {allocation.document} starts at F-{allocation.start:03d}; "
                f"expected F-{expected_start:03d}"
            )
        expected_start = allocation.end + 1
        if allocation.document in documents:
            errors.append(f"review document is registered more than once: {allocation.document}")
        documents.add(allocation.document)

    registered_documents = {allocation.document for allocation in allocations}
    for allocation in allocations:
        path = root / allocation.document
        if not path.is_file():
            errors.append(f"registered review document is missing: {allocation.document}")
            continue
        actual = sorted(finding_ids(path.read_text(encoding="utf-8")))
        expected = list(range(allocation.start, allocation.end + 1))
        if actual != expected:
            errors.append(
                f"allocation mismatch in {allocation.document}: registry requires exactly "
                f"F-{allocation.start:03d}..F-{allocation.end:03d}"
            )

    for path in review_documents(root):
        relative = path.relative_to(root)
        ids = finding_ids(path.read_text(encoding="utf-8"))
        if ids and relative not in registered_documents:
            errors.append(f"review document has unregistered finding ids: {relative}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base revision whose registry rows must remain intact")
    parser.add_argument("--next", action="store_true", help="print the next available finding id")
    args = parser.parse_args()

    try:
        base_registry_text = registry_at_revision(args.base) if args.base else None
    except BaseRevisionUnavailable as error:
        print(
            f"review finding id warning: unable to resolve base revision {error.revision!r} "
            f"(git cat-file exited {error.returncode}); validating the current tree "
            "without an append-only base comparison",
            file=sys.stderr,
        )
        base_registry_text = None
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"review finding id error: unable to read registry at base revision "
            f"{args.base!r}: {error}",
            file=sys.stderr,
        )
        return 1

    errors = validate(ROOT, base_registry_text=base_registry_text)
    if errors:
        for error in errors:
            print(f"review finding id error: {error}", file=sys.stderr)
        return 1

    allocations, _ = parse_registry((ROOT / REGISTRY).read_text(encoding="utf-8"))
    high_water = allocations[-1].end
    if args.next:
        print(f"F-{high_water + 1:03d}")
    else:
        print(
            "review finding ids: OK "
            f"(append-only registry high-water F-{high_water:03d})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
