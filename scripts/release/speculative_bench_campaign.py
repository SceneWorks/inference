#!/usr/bin/env python3
"""Run the decode-speedups benchmark campaign matrix on the CUDA box (epic sc-24432, sc-24446).

The `decode-speedups-bench` profile of `.github/workflows/real-weights.yml` drives this script. A
matrix row names one model snapshot and the `SPECULATIVE_BENCH_*` knobs to measure it under; each
row runs, one process after another,

* ``epic``: the epic's entry point — ``candle-llm``'s ``speculative_bench`` test
  (``core_llm_testkit::run_speculative_bench_from_env``) — in the dispatched checkout, and
* ``baseline``: the pre-epic baseline driver
  (``core-llm-testkit/baseline/speculative_bench_baseline.rs``) copied into a checkout of
  :data:`PRE_EPIC_SHA`, built into its own target directory,

each writing one ``sceneworks.decode-speedups.baseline/3`` document into the run-scoped output
directory (``<output>/<row id>/<run>.json``, its log beside it). A run counts only when its test
process exits 0 having run exactly one test, its document exists, and the document records the
schema, the row's reasoning setting and the checkout the run was meant to measure: the dispatched
commit on a clean tree (epic), or :data:`PRE_EPIC_SHA` whose only change is the copied-in driver
(baseline).

Subcommands:

  plan   validate the matrix in ``$DECODE_BENCH_MATRIX`` (untrusted dispatch input) and write the
         resolved plan; every field is checked before anything is built
  run    run every row of a plan, collect every failure, and write ``summary.json``

The matrix is JSON: ``{"rows": [ROW, ...]}``, each ``ROW`` an object with

  id                  ``[a-z0-9][a-z0-9._-]{0,63}``, unique; the row's output directory
  snapshot            ``qwen38`` / ``bonsai-mlx`` (the repository-variable snapshots the
                      Qwen3.8 / Bonsai lanes use, via ``$DECODE_BENCH_SNAPSHOT_QWEN38`` /
                      ``$DECODE_BENCH_SNAPSHOT_BONSAI_MLX``) or an absolute snapshot path
  runs                subset of ``["epic", "baseline"]`` (default both, epic first)
  format              ``bf16`` (default) / ``q8`` / ``q4`` / ``nvfp4``
  options             speculative options, the harness's wire form (default ``["off", "auto"]``)
  sampling            ``"greedy"`` (default) or a sampling object
  thinking            ``default`` / ``off`` / ``on`` / ``xhigh`` / ``medium`` / ``low``
  new_tokens, repeats non-negative integers (default: the harness's)
  warmup              boolean (default true)
  draft, mtp_head     snapshot path (epic only)
  prefix_cache_bytes  integer or ``"default"`` (a non-zero budget is epic only)
  model               the model label recorded verbatim
  switches            ``{VARIABLE: value}`` for the recorded runtime switches
                      (``CANDLE_LLM_CUDA_GRAPHS``, …); set for both runs of the row
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA = "sceneworks.decode-speedups.baseline/3"
PRE_EPIC_SHA = "c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6"
MATRIX_ENV = "DECODE_BENCH_MATRIX"
SNAPSHOT_ALIASES = {
    "qwen38": "DECODE_BENCH_SNAPSHOT_QWEN38",
    "bonsai-mlx": "DECODE_BENCH_SNAPSHOT_BONSAI_MLX",
}
RUNS = ("epic", "baseline")
FORMATS = ("bf16", "q8", "q4", "nvfp4")
THINKING = ("default", "off", "on", "xhigh", "medium", "low")
# `core_llm_testkit::BENCH_SWITCHES`.
SWITCHES = (
    "CANDLE_LLM_CUDA_GRAPHS",
    "CANDLE_LLM_CUDA_STREAM",
    "CANDLE_LLM_DEVICE_POSITIONS",
    "CANDLE_LLM_FUSED_KERNELS",
    "CANDLE_LLM_NVFP4_GEMV",
    "MLX_LLM_PIPELINING",
    "MLX_LLM_DEVICE_SAMPLER",
    "MLX_LLM_FUSED_ROTATION",
    "MLX_LLM_GDN_KERNEL",
)
ROW_KEYS = {
    "id",
    "snapshot",
    "runs",
    "format",
    "options",
    "sampling",
    "thinking",
    "new_tokens",
    "repeats",
    "warmup",
    "draft",
    "mtp_head",
    "prefix_cache_bytes",
    "model",
    "switches",
}
ROW_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_ROWS = 64
PASSED = "test result: ok. 1 passed"
# The one difference a baseline checkout may carry: the driver copied in (its header's steps).
BASELINE_CHANGES = ["?? crates/llm/candle-llm/tests/speculative_bench_baseline.rs"]
COMMANDS = {
    "epic": [
        "test",
        "--locked",
        "--release",
        "-p",
        "candle-llm",
        "--features",
        "cuda",
        "--test",
        "speculative_bench",
        "--",
        "speculative_bench_writes_the_baseline_document",
        "--exact",
        "--ignored",
        "--nocapture",
    ],
    "baseline": [
        "test",
        "--locked",
        "--release",
        "-p",
        "candle-llm",
        "--features",
        "cuda",
        "--test",
        "speculative_bench_baseline",
        "--",
        "speculative_bench_baseline_writes_the_document",
        "--exact",
        "--ignored",
        "--nocapture",
    ],
}


class MatrixError(ValueError):
    """The matrix is not a plan this script will run."""


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise MatrixError(f"{what} must be a non-empty single-line string, got {value!r}")
    return value


def _absolute(path: str) -> bool:
    return Path(path).is_absolute() or PureWindowsPath(path).is_absolute()


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MatrixError(f"{what} must be a non-negative integer, got {value!r}")
    return value


def plan_row(row: Any, environ: dict[str, str]) -> dict[str, Any]:
    """One validated row: its id, snapshot path, runs, and the environment each run gets."""
    if not isinstance(row, dict):
        raise MatrixError(f"a row must be an object, got {row!r}")
    unknown = set(row) - ROW_KEYS
    if unknown:
        raise MatrixError(f"row has unknown keys {sorted(unknown)}")
    row_id = row.get("id")
    if not isinstance(row_id, str) or not ROW_ID.fullmatch(row_id):
        raise MatrixError(f"row id must match {ROW_ID.pattern}, got {row_id!r}")
    tag = f"row {row_id}"

    snapshot = _text(row.get("snapshot"), f"{tag}: snapshot")
    if snapshot in SNAPSHOT_ALIASES:
        variable = SNAPSHOT_ALIASES[snapshot]
        resolved = environ.get(variable, "").strip()
        if not resolved:
            raise MatrixError(f"{tag}: snapshot alias {snapshot} needs ${variable}")
        snapshot = resolved
    if not _absolute(snapshot):
        raise MatrixError(f"{tag}: snapshot must be an alias or an absolute path, got {snapshot!r}")

    runs = row.get("runs", list(RUNS))
    if (
        not isinstance(runs, list)
        or not runs
        or any(run not in RUNS for run in runs)
        or len(set(runs)) != len(runs)
    ):
        raise MatrixError(f"{tag}: runs must be a non-empty subset of {list(RUNS)}, got {runs!r}")
    runs = [run for run in RUNS if run in runs]

    knobs: dict[str, str] = {"SPECULATIVE_BENCH_SNAPSHOT": snapshot}
    fmt = row.get("format", "bf16")
    if fmt not in FORMATS:
        raise MatrixError(f"{tag}: format must be one of {list(FORMATS)}, got {fmt!r}")
    knobs["SPECULATIVE_BENCH_FORMAT"] = fmt
    if "options" in row:
        options = row["options"]
        if not isinstance(options, list) or not options:
            raise MatrixError(f"{tag}: options must be a non-empty list, got {options!r}")
        knobs["SPECULATIVE_BENCH_OPTIONS"] = json.dumps(options, separators=(",", ":"))
    if "sampling" in row:
        sampling = row["sampling"]
        if sampling != "greedy" and not isinstance(sampling, dict):
            raise MatrixError(f"{tag}: sampling must be \"greedy\" or an object, got {sampling!r}")
        knobs["SPECULATIVE_BENCH_SAMPLING"] = json.dumps(sampling, separators=(",", ":"))
    thinking = row.get("thinking", "default")
    if thinking not in THINKING:
        raise MatrixError(f"{tag}: thinking must be one of {list(THINKING)}, got {thinking!r}")
    knobs["SPECULATIVE_BENCH_THINKING"] = thinking
    for key, variable in (
        ("new_tokens", "SPECULATIVE_BENCH_NEW_TOKENS"),
        ("repeats", "SPECULATIVE_BENCH_REPEATS"),
    ):
        if key in row:
            knobs[variable] = str(_count(row[key], f"{tag}: {key}"))
    if "warmup" in row:
        if not isinstance(row["warmup"], bool):
            raise MatrixError(f"{tag}: warmup must be a boolean, got {row['warmup']!r}")
        if not row["warmup"]:
            knobs["SPECULATIVE_BENCH_WARMUP"] = "0"
    if "model" in row:
        knobs["SPECULATIVE_BENCH_MODEL"] = _text(row["model"], f"{tag}: model")
    epic_only = []
    for key, variable in (
        ("draft", "SPECULATIVE_BENCH_DRAFT"),
        ("mtp_head", "SPECULATIVE_BENCH_MTP_HEAD"),
    ):
        if key in row:
            path = _text(row[key], f"{tag}: {key}")
            if not _absolute(path):
                raise MatrixError(f"{tag}: {key} must be an absolute path, got {path!r}")
            knobs[variable] = path
            epic_only.append(key)
    if "prefix_cache_bytes" in row:
        budget = row["prefix_cache_bytes"]
        if budget != "default":
            budget = _count(budget, f"{tag}: prefix_cache_bytes")
        knobs["SPECULATIVE_BENCH_PREFIX_CACHE_BYTES"] = str(budget)
        if budget != 0:
            epic_only.append("prefix_cache_bytes")
    if "baseline" in runs and epic_only:
        raise MatrixError(
            f"{tag}: {epic_only} cannot run on the pre-epic baseline (no draft models, companion "
            "heads or prefix cache there); give the row runs [\"epic\"]"
        )

    switches = row.get("switches", {})
    if not isinstance(switches, dict):
        raise MatrixError(f"{tag}: switches must be an object, got {switches!r}")
    for name, value in switches.items():
        if name not in SWITCHES:
            raise MatrixError(f"{tag}: {name} is not a recorded switch ({list(SWITCHES)})")
        knobs[name] = _text(value, f"{tag}: switch {name}")
    return {"id": row_id, "runs": runs, "env": knobs}


def plan_matrix(text: str, environ: dict[str, str]) -> dict[str, Any]:
    """The validated plan of a matrix document."""
    try:
        matrix = json.loads(text)
    except json.JSONDecodeError as error:
        raise MatrixError(f"the matrix is not JSON: {error}") from error
    if not isinstance(matrix, dict) or set(matrix) != {"rows"}:
        raise MatrixError('the matrix must be an object with exactly the key "rows"')
    rows = matrix["rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise MatrixError(f"rows must be a list of 1..{MAX_ROWS} rows")
    planned = [plan_row(row, environ) for row in rows]
    ids = [row["id"] for row in planned]
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    if duplicates:
        raise MatrixError(f"duplicate row ids {duplicates}")
    return {"schema": SCHEMA, "pre_epic_sha": PRE_EPIC_SHA, "rows": planned}


def check_document(path: Path, expected_sha: str, changes: list[str], thinking: str) -> None:
    """Refuse a document that is not the run it claims to be: its schema, reasoning setting, and
    the checkout it was built from (``expected_sha`` with exactly ``changes`` in its tree)."""
    if not path.is_file():
        raise ValueError(f"{path.name} was not written")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{path.name}: schema {document.get('schema')!r}, expected {SCHEMA}")
    setting = (document.get("thinking") or {}).get("setting")
    if setting != thinking:
        raise ValueError(f"{path.name}: thinking {setting!r}, expected {thinking!r}")
    git = (document.get("provenance") or {}).get("git") or {}
    if git.get("source") != "git" or git.get("sha") != expected_sha:
        raise ValueError(
            f"{path.name}: built from {git.get('sha')!r} ({git.get('source')}), expected "
            f"{expected_sha} read from git"
        )
    if git.get("changes") != changes or git.get("dirty") is not bool(changes):
        raise ValueError(f"{path.name}: tree changes {git.get('changes')!r}, expected {changes!r}")
    if not document.get("rows"):
        raise ValueError(f"{path.name}: no rows")


def run_one(
    *,
    cargo: str,
    run: str,
    knobs: dict[str, str],
    cwd: Path,
    target_dir: str | None,
    expected_sha: str,
    directory: Path,
) -> None:
    """One benchmark process; raises with the reason it does not count."""
    output = directory / f"{run}.json"
    log = directory / f"{run}.log"
    environ = {
        name: value
        for name, value in os.environ.items()
        # A knob left in the runner's environment never leaks into a row that did not set it.
        if not name.startswith("SPECULATIVE_BENCH_") and name not in SWITCHES
    }
    environ.update(knobs)
    environ["SPECULATIVE_BENCH_OUTPUT"] = str(output)
    environ["SPECULATIVE_BENCH_GIT_SHA"] = expected_sha
    if target_dir is not None:
        environ["CARGO_TARGET_DIR"] = target_dir
    with log.open("wb") as sink:
        result = subprocess.run(
            [cargo, *COMMANDS[run]],
            cwd=cwd,
            env=environ,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log.read_bytes().decode("utf-8", errors="replace")
    sys.stdout.write(text)
    if result.returncode != 0:
        raise ValueError(f"exited {result.returncode}")
    if PASSED not in text:
        raise ValueError("did not run exactly one passing test")
    changes = BASELINE_CHANGES if run == "baseline" else []
    check_document(output, expected_sha, changes, knobs["SPECULATIVE_BENCH_THINKING"])


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def command_plan(args: argparse.Namespace) -> int:
    text = os.environ.get(MATRIX_ENV, "")
    try:
        plan = plan_matrix(text, dict(os.environ))
    except MatrixError as error:
        print(f"::error::{MATRIX_ENV}: {error}", file=sys.stderr)
        return 1
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as sink:
        json.dump(plan, sink, indent=2)
        sink.write("\n")
    print(json.dumps(plan, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    output = Path(args.output)
    roots = {"epic": Path(args.epic_root), "baseline": Path(args.pre_epic_root)}
    expected = {"epic": args.epic_sha, "baseline": PRE_EPIC_SHA}
    for run, root in roots.items():
        if any(run in row["runs"] for row in plan["rows"]) and git_head(root) != expected[run]:
            print(f"::error::{root} is not at {expected[run]}", file=sys.stderr)
            return 1
    summary = []
    for row in plan["rows"]:
        directory = output / row["id"]
        directory.mkdir(parents=True, exist_ok=False)
        for run in row["runs"]:
            print(f"::group::{row['id']} {run}")
            try:
                run_one(
                    cargo=args.cargo,
                    run=run,
                    knobs=row["env"],
                    cwd=roots[run],
                    target_dir=args.pre_epic_target_dir if run == "baseline" else None,
                    expected_sha=expected[run],
                    directory=directory,
                )
                status, error = "ok", None
            except ValueError as failure:
                status, error = "failed", str(failure)
                print(f"::error::{row['id']} {run}: {error}")
            print("::endgroup::")
            summary.append({"id": row["id"], "run": run, "status": status, "error": error})
    with (output / "summary.json").open("x", encoding="utf-8") as sink:
        json.dump({"schema": SCHEMA, "runs": summary}, sink, indent=2)
        sink.write("\n")
    failed = [entry for entry in summary if entry["status"] != "ok"]
    for entry in failed:
        print(f"{entry['id']} {entry['run']}: {entry['error']}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help=f"validate ${MATRIX_ENV} and write the plan")
    plan.add_argument("--output", required=True, help="plan path (must not exist)")
    run = commands.add_parser("run", help="run every row of a plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--epic-root", required=True, help="checkout of the dispatched ref")
    run.add_argument("--epic-sha", required=True, help="the dispatched commit")
    run.add_argument("--pre-epic-root", required=True, help=f"checkout of {PRE_EPIC_SHA}")
    run.add_argument("--pre-epic-target-dir", required=True)
    run.add_argument("--output", required=True, help="run-scoped evidence directory")
    run.add_argument("--cargo", default="cargo")
    args = parser.parse_args(argv)
    return command_plan(args) if args.command == "plan" else command_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
