"""Summaries for the `mlx-veda` A/B (sc-26214). Report-only: no speed threshold is applied.

  veda_ab_summary.py cell <cell dir>                  -> <cell dir>/summary.json
  veda_ab_summary.py aggregate <evidence dir> <cell>... -> <evidence dir>/ab_summary.json

A cell dir holds `off/` and `on/` `veda_ab` render dirs. `speedup` is the dense arm's mean step time
over the sparse arm's. `aggregate` exits 1 when a named cell has no summary (a render failed, was
guard-stopped, or never ran).
"""

import json
import sys
from pathlib import Path


def arm(render: Path) -> dict:
    result = json.loads((render / "result.json").read_text(encoding="utf-8"))
    timings = result.get("timings", {})
    return {
        "step_mean_s": timings.get("step_mean_s"),
        "denoise_s": timings.get("denoise_s"),
        "peak": result.get("peak"),
        "veda": result.get("veda"),
        "degenerate": result.get("degenerate"),
    }


def cell(directory: Path) -> int:
    off, on = arm(directory / "off"), arm(directory / "on")
    off.pop("veda")
    speedup = None
    if off["step_mean_s"] and on["step_mean_s"]:
        speedup = off["step_mean_s"] / on["step_mean_s"]
    summary = {"cell": directory.name, "off": off, "on": on, "speedup": speedup}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def aggregate(evidence: Path, names: list[str]) -> int:
    cells, missing = {}, []
    for name in names:
        path = evidence / name / "summary.json"
        if path.is_file():
            cells[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(name)
    summary = {"cells": cells, "missing": missing}
    (evidence / "ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if missing or not names else 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "cell":
        return cell(Path(argv[2]))
    if len(argv) >= 3 and argv[1] == "aggregate":
        return aggregate(Path(argv[2]), argv[3:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
