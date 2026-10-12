"""sc-25690 — turn one Iris memory-probe log into its classified summary row.

The two dispatch-only probe jobs (`mlx-iris-memory-probe`, `candle-iris-memory-probe`) run one
`train_memory_probe` process per shape. This script reads that process's log, classifies the
result, appends ONE row to the job's `summary.txt`, and exits 1 when the row is `FAILED` so the
runner can turn the job red. Every figure in a row is decimal GB (1e9 bytes); nothing is gated on a
number.

Classes:

* `MEASURED` — the run completed and the figures are the shape's own.
* `CACHE-SATURATED` (MLX) — the run completed, but the process footprint reached MLX's allocator GC
  limit while the buffer cache was enabled: the footprint is that ceiling, not the shape's peak.
  `active_peak_gb` (MLX's own live-byte high-water) is unaffected and is the figure to use.
* `AT-CARD-LIMIT (lower bound)` (CUDA) — the run completed, but the device was within 2% of the
  card's total memory (or the pool reserved more than the card holds): the device-level peak cannot
  rise further, so it is a lower bound. `used_high_gb` (the pool's live-byte high-water) is not
  capped by the card.
* `GUARD-STOP (lower bound)` (MLX) — the footprint guard stopped the run at its cap.
* `EXCEEDS-CARD (lower bound)` (CUDA) — the run died with a CUDA out-of-memory error.
* `FAILED` — anything else, including a passing run whose counters are missing.

Subcommands: `mlx` and `cuda` classify one log; `card` records the selected CUDA card from the
`nvidia-smi` inventory; `expect` fails unless the summary holds exactly one row per named probe (a
probe that produced no row must not leave a green job).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PASSED = "test result: ok. 1 passed"
MARKER = "[[IRIS_MEMORY_PROBE]]"
MLX_TAG = "[[IRIS_MEMORY_PROBE_MLX]]"
CUDA_TAG = "[[IRIS_MEMORY_PROBE_CUDA]]"

MEASURED = "MEASURED"
CACHE_SATURATED = "CACHE-SATURATED"
AT_CARD_LIMIT = "AT-CARD-LIMIT (lower bound)"
GUARD_STOP = "GUARD-STOP (lower bound)"
EXCEEDS_CARD = "EXCEEDS-CARD (lower bound)"
FAILED = "FAILED"

# A footprint this close to MLX's allocator GC limit is the allocator's ceiling.
CACHE_SATURATION_MARGIN_GB = 1.0
# Device memory this full cannot show a higher peak.
CARD_LIMIT_FRACTION = 0.98
MIB_AS_GB = 1024 * 1024 / 1e9

_OOM = re.compile(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY", re.IGNORECASE)
_FIELD = re.compile(r"(\w+)=(\S+)")
_GUARD = re.compile(
    r"^\[guard\] exit=(?P<exit>-?\d+) peak_phys_footprint=(?P<polled>[\d.]+) GB"
    r" lifetime_max_phys_footprint=(?P<lifetime>[\d.]+) GB wall=(?P<wall>[\d.]+)s",
    re.MULTILINE,
)


def _last_fields(log: str, kind: str) -> dict[str, str] | None:
    """The `key=value` fields of the last `[[IRIS_MEMORY_PROBE]] <kind> ...` line, if any."""
    found = None
    for line in log.splitlines():
        at = line.find(f"{MARKER} {kind} ")
        if at >= 0:
            found = dict(_FIELD.findall(line[at:]))
    return found


def _gb(fields: dict[str, str], key: str) -> float:
    return float(fields[key])


def mlx_row(name: str, log: str, rc: int) -> tuple[str, str]:
    """`(class, row text)` for one MLX probe log (the guard's and the test's output together)."""
    guards = list(_GUARD.finditer(log))
    if not guards:
        return FAILED, f"{name} {FAILED} rc={rc} (no footprint-guard result line)"
    guard = guards[-1]
    polled = float(guard["polled"])
    footprint = max(polled, float(guard["lifetime"]))
    limits = _last_fields(log, "mlx_limits")
    measured = (
        f"footprint_peak_gb={footprint:.2f} footprint_polled_gb={polled:.2f}"
        f" wall_s={guard['wall']}"
    )
    limit_text = (
        ""
        if limits is None
        else (
            f" cache_mode={limits['cache_mode']} cache_limit_gb={limits['cache_limit_gb']}"
            f" gc_limit_gb={limits['gc_limit_gb']} memory_limit_gb={limits['memory_limit_gb']}"
        )
    )

    if re.search(r"^\[guard\] SIGTERM: footprint", log, re.MULTILINE):
        # The process was killed, so the end-of-run counters never printed: the last completed
        # step's active peak is itself a lower bound.
        step = None
        for line in log.splitlines():
            if f"{MARKER} step " in line and "active_peak_gb=" in line:
                step = dict(_FIELD.findall(line))
        active = "n/a" if step is None else f">={step['active_peak_gb']}"
        return GUARD_STOP, f"{name} {GUARD_STOP} active_peak_gb={active} {measured}{limit_text}"

    if PASSED in log:
        memory = _last_fields(log, "mlx_memory")
        if limits is None or memory is None:
            return FAILED, f"{name} {FAILED} rc={rc} (passed without the MLX counters) {measured}"
        saturated = (
            _gb(limits, "cache_limit_gb") > 0
            and footprint >= _gb(limits, "gc_limit_gb") - CACHE_SATURATION_MARGIN_GB
        )
        cls = CACHE_SATURATED if saturated else MEASURED
        return cls, (
            f"{name} {cls} active_peak_gb={memory['active_peak_gb']} {measured}"
            f" cache_end_gb={memory['cache_end_gb']}{limit_text}"
        )

    return FAILED, f"{name} {FAILED} rc={rc} {measured}{limit_text}"


def _card_snapshot(snapshot: str, card: int) -> tuple[float, float, float]:
    """`(total, used, free)` GB of `card` from `index, total, used, free` MiB CSV rows."""
    for line in snapshot.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 4 and cells[0] == str(card):
            total, used, free = (int(cell) * MIB_AS_GB for cell in cells[1:])
            return total, used, free
    raise ValueError(f"card {card} is not in the nvidia-smi snapshot: {snapshot!r}")


def cuda_row(name: str, log: str, rc: int, snapshot: str, card: int) -> tuple[str, str]:
    """`(class, row text)` for one CUDA probe log and the card snapshot taken right before it."""
    total, used_before, free_before = _card_snapshot(snapshot, card)
    card_text = (
        f"card={card} card_total_gb={total:.2f} card_used_before_gb={used_before:.2f}"
        f" card_free_before_gb={free_before:.2f}"
    )
    memory = _last_fields(log, "cuda_memory")
    passed = PASSED in log
    out_of_memory = not passed and _OOM.search(log) is not None
    if memory is None:
        if out_of_memory:
            return EXCEEDS_CARD, f"{name} {EXCEEDS_CARD} (counters not printed) {card_text}"
        why = " (passed without the CUDA counters)" if passed else ""
        return FAILED, f"{name} {FAILED} rc={rc}{why} {card_text}"

    used_high = _gb(memory, "used_high_gb")
    reserved_high = _gb(memory, "reserved_high_gb")
    device_peak = _gb(memory, "device_overall_peak_gb")
    baseline = _gb(memory, "baseline_gb")
    fill = (device_peak + baseline) / total
    # Bytes the pool reserved beyond the card's own memory can only be shared system memory.
    spill = max(0.0, reserved_high + baseline - total)
    if spill > 0:
        sysmem = "engaged-and-exhausted" if out_of_memory else "engaged"
    else:
        sysmem = "not-absorbing" if out_of_memory else "not-needed"
    counters = (
        f"used_high_gb={used_high:.2f} reserved_high_gb={reserved_high:.2f}"
        f" device_peak_gb={device_peak:.2f} baseline_gb={baseline:.2f}"
        f" card_fill_pct={fill * 100:.1f} sysmem_spill_gb={spill:.2f} sysmem_fallback={sysmem}"
    )
    if passed:
        at_limit = fill >= CARD_LIMIT_FRACTION or spill > 0
        cls = AT_CARD_LIMIT if at_limit else MEASURED
        return cls, f"{name} {cls} {counters} {card_text}"
    if out_of_memory:
        return EXCEEDS_CARD, f"{name} {EXCEEDS_CARD} {counters} {card_text}"
    return FAILED, f"{name} {FAILED} rc={rc} {counters} {card_text}"


def card_line(inventory: str, card: int) -> str:
    """The selected card (and every other card's occupancy) from the `nvidia-smi` inventory CSV
    (`index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]`, with a header)."""
    parts = []
    selected = False
    for line in inventory.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        total, used, free = (int(cell.split()[0]) * MIB_AS_GB for cell in cells[2:])
        role = "selected" if int(cells[0]) == card else "other"
        selected = selected or role == "selected"
        parts.append(
            f"{role} card {cells[0]} ({cells[1]}): total {total:.2f} GB, used {used:.2f} GB,"
            f" free {free:.2f} GB"
        )
    if not selected:
        raise ValueError(f"card {card} is not in the nvidia-smi inventory: {inventory!r}")
    # ASCII only: the Windows runner's console code page prints this line.
    return "cards before the probes: " + "; ".join(parts)


def missing_rows(summary: str, tag: str, names: list[str]) -> list[str]:
    """Each expected probe name that does not have exactly one row in the summary."""
    problems = []
    for name in names:
        count = sum(1 for line in summary.splitlines() if line.startswith(f"{tag} {name} "))
        if count != 1:
            problems.append(f"{name}: {count} rows (expected exactly 1)")
    return problems


def _append(summary: Path, line: str) -> None:
    print(line, flush=True)
    with summary.open("a", encoding="utf-8") as out:
        out.write(line + "\n")


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for backend in ("mlx", "cuda"):
        row = sub.add_parser(backend)
        row.add_argument("--name", required=True)
        row.add_argument("--log", required=True)
        row.add_argument("--rc", type=int, required=True)
        row.add_argument("--summary", type=Path, required=True)
        if backend == "cuda":
            row.add_argument("--card-snapshot", required=True)
            row.add_argument("--card", type=int, required=True)
    card = sub.add_parser("card")
    card.add_argument("--inventory", required=True)
    card.add_argument("--card", type=int, required=True)
    card.add_argument("--summary", type=Path, required=True)
    expect = sub.add_parser("expect")
    expect.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    expect.add_argument("--summary", required=True)
    expect.add_argument("names", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "card":
        _append(args.summary, f"{CUDA_TAG} {card_line(_read(args.inventory), args.card)}")
        return 0
    if args.command == "expect":
        tag = MLX_TAG if args.backend == "mlx" else CUDA_TAG
        problems = missing_rows(_read(args.summary), tag, args.names)
        for problem in problems:
            print(f"::error::{tag} {problem}", flush=True)
        return 1 if problems else 0
    if args.command == "mlx":
        cls, text = mlx_row(args.name, _read(args.log), args.rc)
        _append(args.summary, f"{MLX_TAG} {text}")
    else:
        cls, text = cuda_row(
            args.name, _read(args.log), args.rc, _read(args.card_snapshot), args.card
        )
        _append(args.summary, f"{CUDA_TAG} {text}")
    return 1 if cls == FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
