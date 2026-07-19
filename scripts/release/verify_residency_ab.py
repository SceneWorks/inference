#!/usr/bin/env python3
"""Classify a resident/sequential real-weight VRAM A/B from SEQ_AB logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


RESULT = re.compile(r"\bSEQ_AB\b.*\bmode=(?P<mode>[\w-]+).*\bpeak_mib=(?P<peak>\d+)\b")


def read_peak(path: Path, expected_mode: str) -> int:
    matches = list(RESULT.finditer(path.read_text(encoding="utf-8", errors="replace")))
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected exactly one SEQ_AB result, found {len(matches)}")
    mode = matches[0].group("mode")
    if mode != expected_mode:
        raise RuntimeError(f"{path}: expected mode={expected_mode}, found mode={mode}")
    return int(matches[0].group("peak"))


def verify(resident_log: Path, sequential_log: Path, min_reduction_mib: int) -> tuple[int, int]:
    resident = read_peak(resident_log, "resident")
    sequential = read_peak(sequential_log, "spec-sequential")
    reduction = resident - sequential
    if reduction < min_reduction_mib:
        raise RuntimeError(
            f"sequential residency reduced peak by {reduction} MiB; required at least {min_reduction_mib} MiB "
            f"(resident={resident}, sequential={sequential})"
        )
    return resident, sequential


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resident", required=True, type=Path)
    parser.add_argument("--sequential", required=True, type=Path)
    parser.add_argument("--min-reduction-mib", required=True, type=int)
    args = parser.parse_args()
    resident, sequential = verify(args.resident, args.sequential, args.min_reduction_mib)
    print(
        f"SEQ_AB_RESULT model={args.model} verdict=pass resident_peak_mib={resident} "
        f"sequential_peak_mib={sequential} reduction_mib={resident - sequential}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"SEQ_AB_RESULT verdict=fail error={error}")
        raise SystemExit(1)
