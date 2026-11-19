#!/usr/bin/env python3
"""Select the decode-speedups campaign's CUDA card by UUID (SW-7200), fail closed.

    decode_bench_select_cuda_card.py <expected GPU UUID> <evidence dir>

Queries the trusted System32 `nvidia-smi.exe` (`scripts/media_campaign_windows.trusted_nvidia_smi`,
the path the guarded CUDA lanes use; the bare name may not be on the runner's PATH), writes every
card's `index, uuid, pci.bus_id, name, memory.*` to `<evidence dir>/cuda-device-inventory.csv`,
and refuses unless exactly one card carries the expected UUID. On success it records that card in
`<evidence dir>/cuda-selected-device.json` and appends `CUDA_VISIBLE_DEVICES=<uuid>` to
`$GITHUB_ENV`, so the benchmark sees only that card (as ordinal 0) whatever its index is.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total", "memory.used", "memory.free")


def query() -> str:
    """The inventory as `nvidia-smi --format=csv,noheader,nounits` text."""
    sys.path.insert(0, str(ROOT))
    from scripts import media_campaign_windows as windows

    result = subprocess.run(
        [windows.trusted_nvidia_smi(), "--query-gpu=" + ",".join(FIELDS),
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"nvidia-smi exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def select(expected: str, inventory: str) -> tuple[dict, list[dict]]:
    """The card whose UUID is `expected`, and every card; refuses unless exactly one matches."""
    if not UUID.fullmatch(expected):
        raise ValueError(f"expected UUID {expected!r} is not a GPU UUID")
    cards = []
    for line in inventory.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(FIELDS):
            raise ValueError(f"unreadable nvidia-smi row {line!r}")
        cards.append(dict(zip(FIELDS, values)))
    if not cards:
        raise ValueError("nvidia-smi listed no CUDA card")
    matched = [card for card in cards if card["uuid"].lower() == expected.lower()]
    if len(matched) != 1:
        seen = ", ".join(f"{c['index']}={c['uuid']}" for c in cards)
        raise ValueError(f"the campaign card {expected} is not present exactly once (cards: {seen})")
    return matched[0], cards


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    expected, evidence = argv[0], Path(argv[1])
    try:
        text = query()
        (evidence / "cuda-device-inventory.csv").write_text(
            ", ".join(FIELDS) + "\n" + text, encoding="utf-8")
        card, _ = select(expected, text)
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"::error::campaign CUDA card not selected: {error}", file=sys.stderr)
        return 1
    (evidence / "cuda-selected-device.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as out:
        out.write(f"CUDA_VISIBLE_DEVICES={card['uuid']}\n")
    print(f"selected CUDA card index {card['index']} {card['uuid']} {card['pci.bus_id']} {card['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
