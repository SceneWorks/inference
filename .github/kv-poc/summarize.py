#!/usr/bin/env python3.12
"""Artifact staging and the job-summary report for kv-poc-campaign.yml.

copy   --src PATH --dest PATH --max-bytes N   copy a file or tree, skipping (and listing) files
                                              larger than N so one oversized tensor cannot sink
                                              the upload of every receipt and log beside it.
report --phase P --root R [--dir D ...]       markdown: receipts produced, unaccepted rows
                                              (refused / failed / aborted), operator stops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def copy(src: Path, dest: Path, max_bytes: int) -> None:
    skipped = []
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for path in files:
        target = dest if src.is_file() else dest / path.relative_to(src)
        size = path.stat().st_size
        if size > max_bytes:
            skipped.append(f"{path} ({size} bytes)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if skipped:
        note = dest.parent / f"{dest.name}.SKIPPED-OVERSIZE.txt"
        note.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"::warning title=artifact files skipped::{len(skipped)} file(s) over {max_bytes} bytes left on the runner; see {note.name}")


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def report(phase: str, root: Path, dirs: list[Path]) -> None:
    if phase == "b":
        dirs = [root / "evidence" / name for name in (
            "sc20677-kv-llama", "sc20677-kv-llama.partial", "sc20677-comparison-llama.json",
            "sc20677-kv-qwen", "sc20677-kv-qwen.partial", "sc20677-comparison-qwen.json",
        )]
    receipts, unaccepted, stops, present = [], [], [], []
    for base in dirs:
        if not base.exists():
            continue
        present.append(base)
        for path in ([base] if base.is_file() else sorted(base.rglob("*.json"))):
            name = path.name
            # receipt.json: W1 rows and the SC-20684 bundle; campaign.json: the SC-20686 bundle.
            # Unaccepted: `<row>.unaccepted.json` (W1, SC-20684) or `failed/*/unaccepted.json` (SC-20686).
            if name in ("receipt.json", "campaign.json"):
                receipts.append(path)
            elif name.endswith(".unaccepted.json") or name == "unaccepted.json":
                unaccepted.append((path, load(path)))
            elif "operator-stop" in name:
                stops.append((path, load(path)))
    print(f"\n#### Phase {phase} evidence\n")
    if not present:
        print("_No evidence directories exist yet._")
        return
    for base in present:
        print(f"- `{base}`")
    print(f"\nReceipts produced: **{len(receipts)}**")
    for path in receipts:
        print(f"- `{path.relative_to(root)}`")
    print(f"\nUnaccepted rows: **{len(unaccepted)}**")
    if unaccepted:
        print("\n| coordinate | outcome | reason | record |\n|---|---|---|---|")
        for path, record in unaccepted:
            detail = str(record.get("detail", ""))[:160].replace("|", "/").replace("\n", " ")
            print(f"| `{record.get('coordinate', '?')}` | {record.get('outcome', '?')} | "
                  f"{record.get('reason', '?')}: {detail} | `{path.relative_to(root)}` |")
    for path, record in stops:
        print(f"\nOperator stop: before row {record.get('beforeRow', '?')} of {record.get('rowsTotal', '?')} "
              f"(`{record.get('beforeRowSlug', '?')}`), `{path.name}`")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("copy")
    c.add_argument("--src", required=True, type=Path)
    c.add_argument("--dest", required=True, type=Path)
    c.add_argument("--max-bytes", required=True, type=int)
    r = sub.add_parser("report")
    r.add_argument("--phase", required=True)
    r.add_argument("--root", required=True, type=Path)
    r.add_argument("--dir", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.command == "copy":
        copy(args.src, args.dest, args.max_bytes)
    else:
        report(args.phase, args.root, args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
