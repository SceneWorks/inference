#!/usr/bin/env python3
"""Fail closed unless one MiniMax-H3 VRAM probe produced one complete receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PREFIX = "[[H3_VRAM]] "
NUMBERS = {
    "peakGb", "trueMemHighGib", "denoiseMemHighGib", "decodeMemHighGib", "preDecodeGb",
    "preDecodeAbsGb", "decodeGb", "steadyGb", "loadPeakGb", "baselineGb", "middleFrameStd", "seconds",
}
INTEGERS = {"vramMeasuredPixels", "frames", "width", "height", "steps"}
REQUIRED = {"model", "tier", "peakOwner", *NUMBERS, *INTEGERS}


def no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def validate(record: object, tier: str) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != REQUIRED:
        raise ValueError(f"receipt schema differs; expected exactly {sorted(REQUIRED)}")
    if record["model"] != "minimax_h3" or record["tier"] != tier:
        raise ValueError(f"receipt model/tier must be minimax_h3/{tier}, got {record['model']!r}/{record['tier']!r}")
    if record["peakOwner"] not in {"denoise", "decode"}:
        raise ValueError(f"receipt peakOwner must be denoise or decode, got {record['peakOwner']!r}")
    for name in NUMBERS:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"receipt {name} must be a finite non-negative number")
    for name in INTEGERS:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"receipt {name} must be a positive integer")
    if record["baselineGb"] >= 1.0:
        raise ValueError(f"receipt baselineGb must be under 1 GB, got {record['baselineGb']}")
    return record


def parse_receipt(lines: list[str], tier: str) -> dict[str, object]:
    matches = [line.removeprefix(PREFIX) for line in lines if line.startswith(PREFIX)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {PREFIX.rstrip()} receipt, got {len(matches)}")
    return validate(json.loads(matches[0], object_pairs_hook=no_duplicate_object), tier)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--tier", choices=("q4", "q8", "bf16"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = parse_receipt(args.log.read_text(encoding="utf-8", errors="replace").splitlines(), args.tier)
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid H3 VRAM receipt: {error}") from error
    args.out.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
