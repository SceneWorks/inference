#!/usr/bin/env python3
"""Run and compare the decode-speedups benchmark campaign (epic sc-24432 AT3 / E5 / E6, sc-24446).

A matrix row names one model snapshot and the ``SPECULATIVE_BENCH_*`` knobs to measure it under.
Each row runs, as separate processes alternating ``epic, baseline, epic, baseline, …`` (its
``process_repeats``, default 3, so the run-to-run noise is measured across processes and is not
confounded with drift),

* ``epic``: the epic's entry point (``core_llm_testkit::run_speculative_bench_from_env``) —
  ``candle-llm``'s ``speculative_bench`` test or ``mlx-llm``'s ``speculative_bench`` module — in a
  checkout of the measured commit, and
* ``baseline``: the pre-epic baseline driver
  (``core-llm-testkit/baseline/speculative_bench_baseline.rs``) in a checkout of
  :data:`PRE_EPIC_SHA` carrying exactly that driver (on MLX the header's block swap plus the
  module registration in ``tests/main.rs``),

each writing one :data:`SCHEMA` document into ``<output>/<row id>/<run>-<k>.json`` (its log
beside it). Both checkouts are built once, before any row, with the commit and dirty flag
**stamped at compile time** (``SPECULATIVE_BENCH_BUILD_GIT_SHA`` / ``_DIRTY``); the harness
refuses a binary whose stamp is not the checkout it runs in. A process counts only when it exits 0
having run exactly one test, its document exists, and the document records the schema, the lane's
backend label, the row's reasoning setting, and — at compile time and at run time — the commit and
tree it was meant to measure: the measured commit on a clean tree (epic), or :data:`PRE_EPIC_SHA`
whose only changes are the driver's (baseline). The snapshot's content identity
(:func:`snapshot_identity`) is recorded per row and re-read after every process.

MLX load admission prices a load-time ``q4``/``q8`` at twice the snapshot's BF16 payload
(``mlx_llm::load_memory``, identical at both revisions), so on a 128 GiB host Qwen3.8-27B
(2 x 55.6 GB = 111.1 GB) and Qwen3.6-35B-A3B (2 x 71.9 GB) are refused before any weight is read.
Measure them from a Q4 snapshot prepared once with ``cargo run --release -p mlx-llm --example
prepare_snapshot -- <source> <out_dir> q4`` (the same quantized projections; admitted at its own
~20 GB payload) and leave ``format`` at its default; the row's snapshot identity — which hashes the
prepared ``config.json``'s ``quantization`` block — is what ``compare`` holds both runs to. The
pre-epic MLX loader refuses Qwen3.6-35B-A3B's MoE MTP head outright, so that model's epic/baseline
rows use a snapshot prepared with ``--without-mtp`` and the ``off`` option only. The pre-epic
Candle loader refuses it too ("qwen3_5 MTP tensor set is incomplete", run 37034184281); on the
Candle lanes a row says ``"without_mtp": true`` instead and the runner prepares the MTP-free copy
itself (:func:`prepare_without_mtp`), beside the Hugging Face cache, before the row's first process.

Lanes (``--lane``): ``cuda`` (Candle/CUDA, the ``decode-speedups-bench`` profile of
``.github/workflows/real-weights.yml``), ``mlx`` (Apple silicon, run locally with ``local``) and
``cpu`` (Candle on the CPU, for exercising the pipeline on a fixture).

Subcommands:

  plan     validate the matrix in ``$DECODE_BENCH_MATRIX`` (untrusted dispatch input) and write the
           resolved plan; every field is checked before anything is built
  run      build both checkouts (stamped), run every row of a plan, write ``summary.json`` —
           rewritten after every process, so a campaign stopped part-way (a job timeout) keeps
           every finished row; ``complete`` marks the summary and each row that ran to its end
  local    the whole lane on this host: create the epic and pre-epic worktrees under
           ``--work-dir``, apply the baseline driver mechanically, plan ``--matrix``, then ``run``
  paths    read campaign directories and write the paired speculative-paths comparison (SW-7200):
           per backend, tier and context, every path's full-request time, TTFT, decode rate,
           accepted tokens per pass, draft overhead and peak memory, each against the ``off``
           row of the same process, plus the inputs ``auto``'s block-draft decision reads
  compare  read campaign directories and write the E6 verdicts and the E5 defaults decisions
           (machine JSON + markdown); refuses incomplete or mismatched evidence. A row the
           campaign stopped during (``complete: false``) is set aside by name, never judged. A row id two
           campaigns of one lane measured is taken from the campaign whose epic commit descends
           from the other's (``--repo`` orders them; unordered commits are refused), the older
           one listed as superseded

The matrix is JSON: ``{"rows": [ROW, ...]}``, each ``ROW`` an object with
(the one optional top-level key besides ``rows`` is ``"runner"``, ``"rw-starvector"`` (default, the
owner's Mac) or ``"rw-llm"`` (nax-macos-2); it only selects the MLX job's runner in the workflow
and is not part of the plan)

  id                  ``[a-z0-9][a-z0-9._-]{0,63}``, unique; the row's output directory
  snapshot            ``qwen38`` / ``bonsai-mlx`` (the repository-variable snapshots the
                      Qwen3.8 / Bonsai lanes use, via ``$DECODE_BENCH_SNAPSHOT_QWEN38`` /
                      ``$DECODE_BENCH_SNAPSHOT_BONSAI_MLX``), an absolute snapshot path, or a
                      pinned Hub snapshot ``hub:<org>/<name>@<40-hex revision>[/<subfolder>]``
                      (SW-7200): the plan binds it under ``$MODEL_PROOF_HUB_ROOT`` in the Hub cache
                      layout and lists it in ``hub_snapshots``, which
                      ``scripts/ci/stage_pinned_hub_snapshots.py --plan`` stages and authenticates
                      before ``run`` (``draft`` and ``mtp_head`` take the same form)
  runs                subset of ``["epic", "baseline"]`` (default both)
  process_repeats     processes per run, alternating with the other run (default 3, at least 1)
  format              ``bf16`` (default) / ``q8`` / ``q4`` / ``nvfp4``: the projection format
                      quantized **at load**; ``bf16`` loads the snapshot as stored, so a packed or
                      prepared-Q4 snapshot keeps its own format (see below)
  options             speculative options, the harness's wire form (default ``["off", "auto"]``):
                      ``"off"``, ``"auto"``, ``{"proposer", "depth": N}``, or a path
                      ``{"proposer", "depth": N | "max" | "recommended", "verify": "linear" |
                      {"tree": {"max_rows": N | "max"}}}`` whose ``"max"`` is the loaded model's
                      advertised bound and ``"recommended"`` the depth ``auto`` runs (SW-7200,
                      ``core_llm_testkit::BenchOption``)
  memory_cap_gib      positive number (SW-7200): the row's external memory cap, substituted for
                      ``{cap_gib}`` in ``--guard`` (a guard naming ``{cap_gib}`` refuses a row
                      without one)
  sampling            ``"greedy"`` (default) or a sampling object
  thinking            ``default`` / ``off`` / ``on`` / ``xhigh`` / ``medium`` / ``low``
  new_tokens, repeats non-negative integers: tokens per request, in-process repeats (default:
                      the harness's)
  warmup              boolean (default true)
  draft, mtp_head     snapshot path (epic only)
  prefix_cache_bytes  integer or ``"default"`` (a non-zero budget is epic only)
  without_mtp         boolean (default false; Candle lanes only): measure an MTP-free copy of the
                      Hugging Face cache snapshot ``snapshot`` names — every ``mtp.*`` tensor
                      dropped, ``mtp_num_hidden_layers`` 0 — kept at :func:`without_mtp_path`
                      and prepared by ``run`` when absent (both runs measure the same copy)
  model               the model (family) label recorded verbatim
  switches            ``{VARIABLE: value}`` for the lane's recorded runtime switches
                      (``CANDLE_LLM_CUDA_GRAPHS``, ``MLX_LLM_PIPELINING``, …); set for every
                      process of the row. Rows that differ only in one switch (same options and
                      runs) are the on/off pairs ``compare`` decides that switch's default from;
                      likewise a ``prefix_cache_bytes`` row needs a twin without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.ci.stage_pinned_hub_snapshots import resolve_hub_root  # noqa: E402

SCHEMA = "sceneworks.decode-speedups.baseline/4"
CAMPAIGN_SCHEMA = "sceneworks.decode-speedups.campaign/1"
COMPARE_SCHEMA = "sceneworks.decode-speedups.compare/1"
PRE_EPIC_SHA = "c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6"
MATRIX_ENV = "DECODE_BENCH_MATRIX"
DRIVER_SOURCE = "crates/contracts/core-llm/core-llm-testkit/baseline/speculative_bench_baseline.rs"
DEFAULTS_SOURCE = Path(__file__).resolve().parents[2] / "crates/contracts/core-llm/src/defaults.rs"
# `core_llm_testkit::BENCH_BUILD_GIT_SHA_ENV` and friends.
BUILD_SHA_ENV = "SPECULATIVE_BENCH_BUILD_GIT_SHA"
BUILD_DIRTY_ENV = "SPECULATIVE_BENCH_BUILD_GIT_DIRTY"
GIT_SHA_ENV = "SPECULATIVE_BENCH_GIT_SHA"
ALLOW_SHA_OVERRIDE_ENV = "SPECULATIVE_BENCH_ALLOW_SHA_OVERRIDE"
SNAPSHOT_ALIASES = {
    "qwen38": "DECODE_BENCH_SNAPSHOT_QWEN38",
    "bonsai-mlx": "DECODE_BENCH_SNAPSHOT_BONSAI_MLX",
}
# SW-7200: a pinned Hub snapshot, bound under the stage script's hub root
# (`scripts/ci/stage_pinned_hub_snapshots.py`, `MODEL_PROOF_HUB_ROOT`).
HUB_ROOT_ENV = "MODEL_PROOF_HUB_ROOT"
HUB_PIN = re.compile(
    r"hub:(?P<repository>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"@(?P<revision>[0-9a-f]{40})(?:/(?P<subfolder>[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*))?"
)
PROPOSERS = ("mtp", "prompt_lookup", "draft_model", "block_draft")
PATHS_SCHEMA = "sceneworks.decode-speedups.paths/1"
RUNS = ("epic", "baseline")
FORMATS = ("bf16", "q8", "q4", "nvfp4")
THINKING = ("default", "off", "on", "xhigh", "medium", "low")
PROMPT_SETS = ("default", "prefix_checkpoints")
PREFIX_CHECKPOINTS = ("off", "auto")
DEFAULT_PROCESS_REPEATS = 3
CANDLE_SWITCHES = (
    "CANDLE_LLM_CUDA_GRAPHS",
    "CANDLE_LLM_CUDA_STREAM",
    "CANDLE_LLM_DEVICE_POSITIONS",
    "CANDLE_LLM_FUSED_KERNELS",
    "CANDLE_LLM_NVFP4_GEMV",
)
MLX_SWITCHES = (
    "MLX_LLM_PIPELINING",
    "MLX_LLM_DEVICE_SAMPLER",
    "MLX_LLM_FUSED_ROTATION",
    "MLX_LLM_GDN_KERNEL",
)
# `core_llm_testkit::BENCH_SWITCHES`.
SWITCHES = CANDLE_SWITCHES + MLX_SWITCHES
# Switches whose *effective* value follows another switch's while their own variable is unset:
# the CUDA stream is candle's own stream when the CUDA-graph runner is on and the legacy stream
# when it is off (`crates/llm/candle-llm/src/device.rs`, `CudaStreamKind::resolve`). Two rows that
# differ only in the graph switch therefore also differ in the stream's effective value, so the
# E5 twin key leaves a derived switch out while its variable is unset; a derived switch set
# explicitly stays in the key and still separates twins.
DERIVED_SWITCHES = {"CANDLE_LLM_CUDA_GRAPHS": ("CANDLE_LLM_CUDA_STREAM",)}
ROW_KEYS = {
    "id",
    "snapshot",
    "runs",
    "process_repeats",
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
    "without_mtp",
    "memory_cap_gib",
    "prompt_set",
    "prefix_checkpoints",
}
ROW_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_ROWS = 64
TEST_RESULT = re.compile(r"^test result: .*$", re.MULTILINE)
PASSED = re.compile(r"^test result: ok\. 1 passed; 0 failed;")
MLX_MODULE_REGISTRATION = '\n#[path = "speculative_bench_baseline.rs"]\nmod speculative_bench_baseline;\n'
MLX_MAIN = "crates/llm/mlx-llm/tests/main.rs"


class Lane(NamedTuple):
    """One backend's build and run recipe."""

    name: str
    backend: str  # the documents' `backend` label (SPECULATIVE_BENCH_BACKEND)
    package: str
    features: tuple[str, ...]
    tests: dict[str, tuple[str, str]]  # run -> (cargo test target, exact test filter)
    driver_path: str  # where the baseline driver lives in the pre-epic checkout
    baseline_changes: tuple[str, ...]  # its `git status --porcelain` lines
    switches: tuple[str, ...]
    mlx: bool


_CANDLE_TESTS = {
    "epic": ("speculative_bench", "speculative_bench_writes_the_baseline_document"),
    "baseline": ("speculative_bench_baseline", "speculative_bench_baseline_writes_the_document"),
}
_CANDLE_DRIVER = "crates/llm/candle-llm/tests/speculative_bench_baseline.rs"
_MLX_DRIVER = "crates/llm/mlx-llm/tests/speculative_bench_baseline.rs"
LANES = {
    "cuda": Lane(
        name="cuda",
        backend="candle-cuda",
        package="candle-llm",
        features=("cuda",),
        tests=_CANDLE_TESTS,
        driver_path=_CANDLE_DRIVER,
        baseline_changes=(f"?? {_CANDLE_DRIVER}",),
        switches=CANDLE_SWITCHES,
        mlx=False,
    ),
    "cpu": Lane(
        name="cpu",
        backend="candle-cpu",
        package="candle-llm",
        features=(),
        tests=_CANDLE_TESTS,
        driver_path=_CANDLE_DRIVER,
        baseline_changes=(f"?? {_CANDLE_DRIVER}",),
        switches=CANDLE_SWITCHES,
        mlx=False,
    ),
    "mlx": Lane(
        name="mlx",
        backend="mlx",
        package="mlx-llm",
        features=(),
        tests={
            "epic": (
                "integration",
                "speculative_bench::speculative_bench_writes_the_baseline_document",
            ),
            "baseline": (
                "integration",
                "speculative_bench_baseline::speculative_bench_baseline_writes_the_document",
            ),
        },
        driver_path=_MLX_DRIVER,
        baseline_changes=(f" M {MLX_MAIN}", f"?? {_MLX_DRIVER}"),
        switches=MLX_SWITCHES,
        mlx=True,
    ),
}


# The MLX job's runner labels a matrix may select (`runs-on` reads the same key in the workflow).
MATRIX_RUNNERS = ("rw-starvector", "rw-llm")


class MatrixError(ValueError):
    """The matrix is not a plan this script will run."""


class CampaignError(ValueError):
    """A checkout, build or process that does not count."""


class Refused(ValueError):
    """Evidence `compare` will not draw a verdict from."""


# ------------------------------------------------------------------------------------------- plan


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


def hub_pin(value: str, environ: dict[str, str], what: str) -> tuple[str, dict[str, str]] | None:
    """A ``hub:<org>/<name>@<revision>[/<subfolder>]`` pin's runner path — where
    ``stage_pinned_hub_snapshots.py`` stages it under ``$MODEL_PROOF_HUB_ROOT`` (``~`` expanded,
    real path) in the Hub cache layout — and the pin; ``None`` for any other value."""
    if not value.startswith("hub:"):
        return None
    match = HUB_PIN.fullmatch(value)
    subfolder = (match.group("subfolder") or "") if match else ""
    if not match or any(not part.strip(".") for part in subfolder.split("/") if subfolder):
        raise MatrixError(
            f"{what} must be hub:<org>/<name>@<40-hex revision>[/<subfolder>], got {value!r}"
        )
    root = environ.get(HUB_ROOT_ENV, "").strip()
    if not root:
        raise MatrixError(f"{what}: a Hub pin needs ${HUB_ROOT_ENV} (the runner's Hub root)")
    repository, revision = match.group("repository"), match.group("revision")
    try:
        root_path = resolve_hub_root(environ)
    except ValueError as error:
        raise MatrixError(f"{what}: {error} (got {root!r})") from error
    base = Path(os.path.realpath(root_path))
    path = base / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    if subfolder:
        path = path.joinpath(*subfolder.split("/"))
    if not _absolute(str(path)):
        raise MatrixError(f"{what}: ${HUB_ROOT_ENV} must be an absolute or ~ path, got {root!r}")
    return str(path), {"repository": repository, "revision": revision, "subfolder": subfolder}


def _option(option: Any, what: str) -> None:
    """Refuse an option the harness would refuse (``core_llm_testkit::BenchOption``): a string
    mode, a request-wire object, or a path object with a ``"max"``-able depth and a verify layout."""

    def bound(value: Any, key: str) -> None:
        named = ("max", "recommended") if key == "depth" else ("max",)
        if value not in named and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise MatrixError(f"{what}: `{key}` must be a positive integer or one of {named}, got {value!r}")

    if isinstance(option, str):
        if option not in ("off", "auto"):
            raise MatrixError(f"{what}: unknown option {option!r}")
        return
    if not isinstance(option, dict):
        raise MatrixError(f"{what}: an option is a string or an object, got {option!r}")
    if "mode" in option:
        return  # the legacy `mtp` shape; the harness's request parser checks it
    if option.get("proposer") not in PROPOSERS:
        raise MatrixError(f"{what}: proposer must be one of {list(PROPOSERS)}, got {option!r}")
    unknown = set(option) - {"proposer", "depth", "verify"}
    if unknown:
        raise MatrixError(f"{what}: option has unknown keys {sorted(unknown)}")
    bound(option.get("depth"), "depth")
    verify = option.get("verify", "linear")
    if verify == "linear":
        return
    tree = verify.get("tree") if isinstance(verify, dict) and set(verify) == {"tree"} else None
    if not isinstance(tree, dict) or set(tree) != {"max_rows"}:
        raise MatrixError(
            f"{what}: verify must be \"linear\" or {{\"tree\": {{\"max_rows\": N | \"max\"}}}}, "
            f"got {verify!r}"
        )
    bound(tree["max_rows"], "max_rows")


def _require_local(path: str, tag: str, field: str, *, file_ok: bool = False) -> None:
    """Refuse a runner-local path that does not exist here: ``run`` would otherwise fail after the
    cold release build (``snapshot_identity``: "is not a directory")."""
    if not (Path(path).is_dir() or (file_ok and Path(path).is_file())):
        kind = "file or directory" if file_ok else "directory"
        raise MatrixError(f"{tag}: {field} {path!r} is not an existing {kind} on this runner")


def plan_row(
    row: Any, environ: dict[str, str], lane: Lane, require_local: bool = False
) -> dict[str, Any]:
    """One validated row: its id, runs, process repeats, switches, and the environment each
    process gets. ``require_local`` (the ``plan`` command, which runs on the runner before the
    build) also refuses a non-``hub:`` snapshot, draft or MTP head path that does not exist;
    ``hub:`` pins are staged after the plan, so they are not checked."""
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
    pins: list[dict[str, str]] = []
    pinned = hub_pin(snapshot, environ, f"{tag}: snapshot")
    if pinned:
        snapshot = pinned[0]
        pins.append(pinned[1])
    check_snapshot = require_local and not pinned
    if snapshot in SNAPSHOT_ALIASES:
        variable = SNAPSHOT_ALIASES[snapshot]
        resolved = environ.get(variable, "").strip()
        if not resolved:
            raise MatrixError(f"{tag}: snapshot alias {snapshot} needs ${variable}")
        snapshot = resolved
    if not _absolute(snapshot):
        raise MatrixError(f"{tag}: snapshot must be an alias or an absolute path, got {snapshot!r}")
    if check_snapshot:
        _require_local(snapshot, tag, "snapshot")
    prepare = None
    without_mtp = row.get("without_mtp", False)
    if not isinstance(without_mtp, bool):
        raise MatrixError(f"{tag}: without_mtp must be a boolean, got {without_mtp!r}")
    if without_mtp:
        if lane.mlx:
            raise MatrixError(
                f"{tag}: without_mtp is a Candle-lane preparation; an MLX row names a snapshot "
                "prepared with `prepare_snapshot --without-mtp`"
            )
        derived = without_mtp_path(snapshot)
        if derived is None:
            raise MatrixError(
                f"{tag}: without_mtp needs a Hugging Face cache snapshot "
                f"(…/models--<org>--<name>/snapshots/<revision>), got {snapshot!r}"
            )
        prepare = {"without_mtp": snapshot}
        snapshot = derived

    runs = row.get("runs", list(RUNS))
    if (
        not isinstance(runs, list)
        or not runs
        or any(run not in RUNS for run in runs)
        or len(set(runs)) != len(runs)
    ):
        raise MatrixError(f"{tag}: runs must be a non-empty subset of {list(RUNS)}, got {runs!r}")
    runs = [run for run in RUNS if run in runs]
    process_repeats = _count(
        row.get("process_repeats", DEFAULT_PROCESS_REPEATS), f"{tag}: process_repeats"
    )
    if process_repeats < 1:
        raise MatrixError(f"{tag}: process_repeats must be at least 1")

    knobs: dict[str, str] = {
        "SPECULATIVE_BENCH_SNAPSHOT": snapshot,
        "SPECULATIVE_BENCH_BACKEND": lane.backend,
    }
    fmt = row.get("format", "bf16")
    if fmt not in FORMATS:
        raise MatrixError(f"{tag}: format must be one of {list(FORMATS)}, got {fmt!r}")
    knobs["SPECULATIVE_BENCH_FORMAT"] = fmt
    if "options" in row:
        options = row["options"]
        if not isinstance(options, list) or not options:
            raise MatrixError(f"{tag}: options must be a non-empty list, got {options!r}")
        for option in options:
            _option(option, f"{tag}: options")
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
            pinned = hub_pin(path, environ, f"{tag}: {key}")
            if pinned:
                path = pinned[0]
                pins.append(pinned[1])
            if not _absolute(path):
                raise MatrixError(f"{tag}: {key} must be an absolute path, got {path!r}")
            if require_local and not pinned:
                _require_local(path, tag, key, file_ok=key == "mtp_head")
            knobs[variable] = path
            epic_only.append(key)
    if "prefix_cache_bytes" in row:
        budget = row["prefix_cache_bytes"]
        if budget != "default":
            budget = _count(budget, f"{tag}: prefix_cache_bytes")
        knobs["SPECULATIVE_BENCH_PREFIX_CACHE_BYTES"] = str(budget)
        if budget != 0:
            epic_only.append("prefix_cache_bytes")
    if "prompt_set" in row:
        prompt_set = _text(row["prompt_set"], f"{tag}: prompt_set")
        if prompt_set not in PROMPT_SETS:
            raise MatrixError(
                f"{tag}: prompt_set must be one of {list(PROMPT_SETS)}, got {prompt_set!r}"
            )
        knobs["SPECULATIVE_BENCH_PROMPT_SET"] = prompt_set
        if prompt_set != "default":
            epic_only.append("prompt_set")
    if "prefix_checkpoints" in row:
        checkpoints = _text(row["prefix_checkpoints"], f"{tag}: prefix_checkpoints")
        if checkpoints not in PREFIX_CHECKPOINTS:
            raise MatrixError(
                f"{tag}: prefix_checkpoints must be one of {list(PREFIX_CHECKPOINTS)}, got {checkpoints!r}"
            )
        knobs["SPECULATIVE_BENCH_PREFIX_CHECKPOINTS"] = checkpoints
        epic_only.append("prefix_checkpoints")
    if "baseline" in runs and epic_only:
        raise MatrixError(
            f"{tag}: {epic_only} cannot run on the pre-epic baseline (no draft models, companion "
            "heads, prefix cache or prefix checkpoints there); give the row runs [\"epic\"]"
        )

    switches = row.get("switches", {})
    if not isinstance(switches, dict):
        raise MatrixError(f"{tag}: switches must be an object, got {switches!r}")
    for name, value in switches.items():
        if name not in lane.switches:
            raise MatrixError(
                f"{tag}: {name} is not a recorded {lane.name} switch ({list(lane.switches)})"
            )
        knobs[name] = _text(value, f"{tag}: switch {name}")
    planned = {
        "id": row_id,
        "runs": runs,
        "process_repeats": process_repeats,
        "switches": dict(switches),
        "env": knobs,
    }
    if "memory_cap_gib" in row:
        cap = row["memory_cap_gib"]
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 1024:
            raise MatrixError(f"{tag}: memory_cap_gib must be a positive number of GiB, got {cap!r}")
        planned["memory_cap_gib"] = cap
    if prepare:
        planned["prepare"] = prepare
    if pins:
        planned["hub"] = pins
    return planned


def plan_matrix(
    text: str, environ: dict[str, str], lane: str = "cuda", require_local: bool = False
) -> dict[str, Any]:
    """The validated plan of a matrix document for ``lane``; ``require_local`` is
    :func:`plan_row`'s."""
    if lane not in LANES:
        raise MatrixError(f"lane must be one of {sorted(LANES)}, got {lane!r}")
    try:
        matrix = json.loads(text)
    except json.JSONDecodeError as error:
        raise MatrixError(f"the matrix is not JSON: {error}") from error
    if not isinstance(matrix, dict) or not {"rows"} <= set(matrix) <= {"rows", "runner"}:
        raise MatrixError('the matrix must be an object with the key "rows" and optionally "runner"')
    if "runner" in matrix and matrix["runner"] not in MATRIX_RUNNERS:
        raise MatrixError(f"runner must be one of {list(MATRIX_RUNNERS)}, got {matrix['runner']!r}")
    rows = matrix["rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise MatrixError(f"rows must be a list of 1..{MAX_ROWS} rows")
    planned = [plan_row(row, environ, LANES[lane], require_local) for row in rows]
    ids = [row["id"] for row in planned]
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    if duplicates:
        raise MatrixError(f"duplicate row ids {duplicates}")
    plan: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "document_schema": SCHEMA,
        "lane": lane,
        "pre_epic_sha": PRE_EPIC_SHA,
        "rows": planned,
    }
    # Every distinct Hub pin once, in first-use order, in the form the stage script reads.
    pins: list[dict[str, str]] = []
    for row in planned:
        pins += [pin for pin in row.get("hub", []) if pin not in pins]
    if pins:
        plan["asset_setup"] = "hub-snapshots"
        plan["hub_snapshots"] = {f"DECODE_BENCH_HUB_{i}": pin for i, pin in enumerate(pins, 1)}
    return plan


# -------------------------------------------------------------------------------------- checkouts


def git(root: Path, *args: str) -> str | None:
    """``git -C root args`` stdout, or ``None`` when git does not answer."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def git_head(root: Path) -> str:
    return (git(root, "rev-parse", "HEAD") or "").strip()


def git_changes(root: Path) -> list[str] | None:
    """``git status --porcelain`` lines (status columns kept), or ``None``."""
    status = git(root, "status", "--porcelain")
    return None if status is None else [line for line in status.splitlines() if line.strip()]


def baseline_driver_text(lane: Lane, source: str) -> str:
    """The driver as the lane's pre-epic checkout carries it: verbatim on Candle; on MLX the
    header's ``sed`` — delete the ``// BEGIN candle backend`` … ``// END candle backend`` block
    and un-comment every ``//mlx `` line."""
    if not lane.mlx:
        return source
    out, skipping = [], False
    for line in source.splitlines(keepends=True):
        if not skipping and line.startswith("// BEGIN candle backend"):
            skipping = True
            continue
        if skipping:
            if line.rstrip("\r\n") == "// END candle backend":
                skipping = False
            continue
        out.append(line[len("//mlx ") :] if line.startswith("//mlx ") else line)
    if skipping:
        raise CampaignError("the driver's candle backend block is not terminated")
    return "".join(out)


def mlx_main_text(pre_epic_root: Path) -> str:
    """The pre-epic ``tests/main.rs`` with the driver's module registered (the header's
    ``printf >>``)."""
    original = git(pre_epic_root, "show", f"HEAD:{MLX_MAIN}")
    if original is None:
        raise CampaignError(f"cannot read {MLX_MAIN} at the pre-epic HEAD")
    return original + MLX_MODULE_REGISTRATION


def apply_baseline_driver(lane: Lane, epic_root: Path, pre_epic_root: Path) -> None:
    """Install the epic checkout's driver into the pre-epic checkout (idempotent; refuses a
    ``tests/main.rs`` it did not write)."""
    source = (epic_root / DRIVER_SOURCE).read_bytes().decode("utf-8")
    target = pre_epic_root / lane.driver_path
    target.write_bytes(baseline_driver_text(lane, source).encode("utf-8"))
    if lane.mlx:
        main = pre_epic_root / MLX_MAIN
        wanted = mlx_main_text(pre_epic_root)
        current = main.read_text(encoding="utf-8")
        if current != wanted:
            if current + MLX_MODULE_REGISTRATION != wanted:
                raise CampaignError(f"{main} has changes other than the driver's registration")
            main.write_text(wanted, encoding="utf-8")


def check_checkout(lane: Lane, run: str, root: Path, expected_sha: str, epic_root: Path) -> None:
    """Refuse a checkout that is not the commit and tree the run must measure: ``HEAD`` is
    ``expected_sha``; the epic tree is clean; the pre-epic tree differs by exactly the driver,
    which is the epic checkout's (and on MLX the registration)."""
    head = git_head(root)
    if head != expected_sha:
        raise CampaignError(f"{root} is at {head or 'no commit'}, not {expected_sha}")
    changes = git_changes(root)
    want = sorted(lane.baseline_changes) if run == "baseline" else []
    if changes is None or sorted(changes) != want:
        raise CampaignError(f"{root}: tree changes {changes!r}, expected {want!r}")
    if run == "baseline":
        source = (epic_root / DRIVER_SOURCE).read_bytes().decode("utf-8")
        installed = (root / lane.driver_path).read_bytes().decode("utf-8")
        if installed != baseline_driver_text(lane, source):
            raise CampaignError(
                f"{root / lane.driver_path} is not the epic checkout's driver for {lane.name}"
            )
        if lane.mlx and (root / MLX_MAIN).read_text(encoding="utf-8") != mlx_main_text(root):
            raise CampaignError(f"{root / MLX_MAIN} is not the pre-epic file plus the registration")


def ensure_worktree(repo: Path, path: Path, sha: str) -> None:
    """A detached worktree of ``repo`` at ``sha``: created, or reused when already there."""
    if path.exists():
        if git_head(path) != sha:
            raise CampaignError(f"{path} exists and is not a checkout of {sha}")
        return
    result = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), sha],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise CampaignError(f"git worktree add {path} {sha}: {result.stderr.strip()}")


# -------------------------------------------------------------------------------------- snapshots

_IDENTITY_BYTES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".npz", ".pt")


def snapshot_identity(path: str) -> dict[str, Any]:
    """A content identity of a snapshot directory: SHA-256 over its config / tokenizer /
    weight-index files' bytes and every weight file's relative name and size. A directory name is
    not an identity — two snapshots named alike with different weights, a re-quantized shard, or a
    swapped config all change this."""
    root = Path(path)
    if not root.is_dir():
        raise CampaignError(f"snapshot {path} is not a directory")
    digest = hashlib.sha256()
    hashed, weights = [], []
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file.relative_to(root).as_posix()
        if file.name in _IDENTITY_BYTES or file.name.endswith(".index.json"):
            digest.update(f"bytes {relative}\0".encode())
            digest.update(file.read_bytes())
            digest.update(b"\0")
            hashed.append(relative)
        elif file.name.endswith(_WEIGHT_SUFFIXES):
            digest.update(f"weight {relative} {file.stat().st_size}\0".encode())
            weights.append(relative)
    if "config.json" not in hashed and not weights:
        raise CampaignError(f"snapshot {path} has no config.json and no weight files")
    return {"sha256": digest.hexdigest(), "hashed": hashed, "weights": weights}


# ---------------------------------------------------------------------------- MTP-free snapshots

WITHOUT_MTP_DIR = "prepared-without-mtp"
WITHOUT_MTP_MARKER = ".sceneworks-without-mtp.json"
WITHOUT_MTP_VERSION = 1
_HF_SNAPSHOT_PARTS = re.compile(
    r"^(?P<base>.*?)[/\\](?P<repo>models--[^/\\]+)[/\\]snapshots[/\\](?P<rest>[^/\\]+(?:[/\\][^/\\]+)*)$"
)
_COPY_CHUNK = 64 << 20


def without_mtp_path(source: str) -> str | None:
    """Where the MTP-free copy of the Hugging Face cache snapshot ``source`` lives: the same
    ``models--<org>--<name>/snapshots/<revision>[/<subdir>]`` under ``prepared-without-mtp`` beside
    the cache's ``hub`` directory (outside any checkout, on the cache's volume so the shards it keeps
    are hard links; the cache layout keeps :func:`family_label` naming the repository). ``None``
    when ``source`` is not a cache snapshot path. Pure string work: the plan names a runner path."""
    match = _HF_SNAPSHOT_PARTS.match(source.rstrip("/\\"))
    if not match:
        return None
    sep = "\\" if "\\" in source and "/" not in source else "/"
    base = match["base"]
    head, _, last = base.replace("\\", "/").rpartition("/")
    if last == "hub" and head:
        base = base[: len(head)]
    rest = match["rest"].replace("\\", sep).replace("/", sep)
    return sep.join([base, WITHOUT_MTP_DIR, match["repo"], "snapshots", rest])


def _is_mtp(key: str) -> bool:
    return key.startswith("mtp.")


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    """The byte offset a safetensors file's data starts at, and its header."""
    with path.open("rb") as source:
        length = int.from_bytes(source.read(8), "little")
        try:
            header = json.loads(source.read(length))
        except ValueError as error:
            raise CampaignError(f"{path}: not a safetensors file ({error})") from error
    if not isinstance(header, dict):
        raise CampaignError(f"{path}: not a safetensors file")
    return 8 + length, header


def _tensor_bytes(info: dict[str, Any]) -> int:
    begin, end = info["data_offsets"]
    return end - begin


def _write_safetensors_without(source: Path, target: Path, drop) -> dict[str, int]:
    """Write ``source`` to ``target`` without the tensors ``drop`` names, the kept tensors' bytes
    unchanged and packed in their stored order; returns each dropped tensor's byte size."""
    start, header = _safetensors_header(source)
    metadata = header.pop("__metadata__", None)
    dropped = {k: _tensor_bytes(v) for k, v in header.items() if drop(k)}
    kept = sorted(
        ((k, v) for k, v in header.items() if not drop(k)), key=lambda kv: kv[1]["data_offsets"][0]
    )
    rewritten: dict[str, Any] = {} if metadata is None else {"__metadata__": metadata}
    offset, spans = 0, []
    for key, info in kept:
        size = _tensor_bytes(info)
        rewritten[key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
        }
        spans.append((start + info["data_offsets"][0], size))
        offset += size
    blob = json.dumps(rewritten, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # the data starts 8-byte aligned, as safetensors writes it
    with source.open("rb") as reader, target.open("xb") as sink:
        sink.write(len(blob).to_bytes(8, "little"))
        sink.write(blob)
        for position, size in spans:
            reader.seek(position)
            while size:
                chunk = reader.read(min(size, _COPY_CHUNK))
                if not chunk:
                    raise CampaignError(f"{source}: truncated tensor data")
                sink.write(chunk)
                size -= len(chunk)
    return dropped


def _link_or_copy(source: Path, target: Path) -> None:
    """A hard link to ``source``'s content (a cache snapshot file is a symlink into ``blobs``),
    else a copy (another volume, or a filesystem without hard links)."""
    try:
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copyfile(source, target)


def _config_without_mtp(text: str) -> str:
    config = json.loads(text)
    if not isinstance(config, dict):
        raise CampaignError("config.json is not an object")
    root = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    root["mtp_num_hidden_layers"] = 0
    return json.dumps(config, indent=2) + "\n"


def prepare_without_mtp(source: str, destination: str) -> bool:
    """Make ``destination`` the MTP-free copy of the snapshot ``source``: every ``mtp.*`` tensor
    dropped from the shards that hold one (the rest hard-linked), the weight index's map and total
    size without them, ``mtp_num_hidden_layers`` 0 in the (text) config — what both Candle loaders
    load as a target with no native head — and every other file linked. Idempotent: a copy this
    preparer made from the same source content is reused (returns ``False``); anything else at
    ``destination`` is refused, never overwritten. Built beside it and renamed into place, so a
    stopped preparation leaves no copy that looks finished."""
    src, out = Path(source), Path(destination)
    stamp = {
        "version": WITHOUT_MTP_VERSION,
        "source": source,
        "source_identity": snapshot_identity(source)["sha256"],
    }
    if out.exists():
        try:
            existing = json.loads((out / WITHOUT_MTP_MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if existing != stamp:
            raise CampaignError(
                f"{destination} exists and is not this preparer's MTP-free copy of {source}; "
                "remove it to prepare again"
            )
        return False
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    dropped: dict[str, int] = {}
    indices = []
    for file in sorted(p for p in src.rglob("*") if p.is_file()):
        relative = file.relative_to(src)
        target = partial / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.name.endswith(".index.json"):
            indices.append(relative)
        elif relative == Path("config.json"):
            try:
                target.write_text(_config_without_mtp(file.read_text(encoding="utf-8")), encoding="utf-8")
            except ValueError as error:
                raise CampaignError(f"{file}: {error}") from error
        elif file.suffix == ".safetensors" and any(map(_is_mtp, _safetensors_header(file)[1])):
            dropped.update(_write_safetensors_without(file, target, _is_mtp))
        else:
            _link_or_copy(file, target)
    if not (partial / "config.json").is_file():
        raise CampaignError(f"snapshot {source} has no config.json")
    for relative in indices:
        try:
            index = json.loads((src / relative).read_text(encoding="utf-8"))
            index["weight_map"] = {
                k: v for k, v in index["weight_map"].items() if not _is_mtp(k)
            }
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise CampaignError(f"{src / relative}: not a weight index ({error!r})") from error
        metadata = index.get("metadata")
        total = metadata.get("total_size") if isinstance(metadata, dict) else None
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            # Hugging Face writes it as a float on some checkpoints (Qwen3.6-35B-A3B).
            metadata["total_size"] = type(total)(total - sum(dropped.values()))
        (partial / relative).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (partial / WITHOUT_MTP_MARKER).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, out)
    return True


# -------------------------------------------------------------------------------------------- run


def cargo_args(lane: Lane, run: str, *, no_run: bool = False) -> list[str]:
    test, name = lane.tests[run]
    args = ["test", "--locked", "--release", "-p", lane.package]
    if lane.features:
        args += ["--features", ",".join(lane.features)]
    args += ["--test", test]
    if no_run:
        return args + ["--no-run"]
    return args + ["--", name, "--exact", "--ignored", "--nocapture"]


def check_document(
    path: Path,
    *,
    expected_sha: str,
    changes: list[str],
    thinking: str,
    backend: str,
    sha_override: bool = False,
) -> dict[str, Any]:
    """Refuse a document that is not the run it claims to be: its schema, backend label,
    reasoning setting, and the checkout it was built from and ran in — ``expected_sha`` stamped at
    compile time with the dirty flag of ``changes``, and ``git`` at run time reporting the same
    commit with exactly ``changes``. With ``sha_override`` (``--allow-sha-override``) the binary
    carries no stamp and may record the operator's SHA instead."""
    if not path.is_file():
        raise CampaignError(f"{path.name} was not written")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise CampaignError(f"{path.name}: schema {document.get('schema')!r}, expected {SCHEMA}")
    if document.get("backend") != backend:
        raise CampaignError(
            f"{path.name}: backend {document.get('backend')!r}, expected {backend!r}"
        )
    setting = (document.get("thinking") or {}).get("setting")
    if setting != thinking:
        raise CampaignError(f"{path.name}: thinking {setting!r}, expected {thinking!r}")
    git_block = (document.get("provenance") or {}).get("git") or {}
    sources = ("git", GIT_SHA_ENV) if sha_override else ("git",)
    if git_block.get("source") not in sources or git_block.get("sha") != expected_sha:
        raise CampaignError(
            f"{path.name}: ran in {git_block.get('sha')!r} ({git_block.get('source')}), expected "
            f"{expected_sha} read from git"
        )
    stamp = (git_block.get("build_sha"), git_block.get("build_dirty"))
    if sha_override:
        if stamp != (None, None):
            raise CampaignError(f"{path.name}: an --allow-sha-override run carries a stamp {stamp}")
    elif stamp != (expected_sha, bool(changes)):
        raise CampaignError(
            f"{path.name}: compiled from {stamp}, expected ({expected_sha!r}, {bool(changes)})"
        )
    if git_block.get("source") == "git" and (
        sorted(git_block.get("changes") or []) != sorted(changes)
        or git_block.get("dirty") is not bool(changes)
    ):
        raise CampaignError(
            f"{path.name}: tree changes {git_block.get('changes')!r}, expected {changes!r}"
        )
    if not document.get("rows"):
        raise CampaignError(f"{path.name}: no rows")
    return document


def base_environment() -> dict[str, str]:
    """The runner's environment minus every benchmark knob, stamp and switch: nothing left in it
    leaks into a row that did not set it."""
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("SPECULATIVE_BENCH_") and name not in SWITCHES
    }


def mlx_prebuilt_environment(root: Path, environ: dict[str, str]) -> dict[str, str]:
    """``scripts/fetch-prebuilt-mlx.sh --build-type Release`` in ``root`` (each checkout pins its
    own mlx-rs revision): the two variables it prints. Exit 1 (no prebuilt for the revision) builds
    libmlx from source; any other failure is refused."""
    result = subprocess.run(
        ["bash", "scripts/fetch-prebuilt-mlx.sh", "--build-type", "Release"],
        cwd=root,
        env=environ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode == 1:
        print(f"::warning::{root}: no prebuilt libmlx; building it from source", file=sys.stderr)
        return {}
    if result.returncode != 0:
        raise CampaignError(
            f"{root}: fetch-prebuilt-mlx.sh exited {result.returncode}: {result.stderr}"
        )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    wanted = ("PMETAL_MLX_PREBUILT_DIR", "PMETAL_METALLIB_PATH")
    if any(not values.get(name) for name in wanted):
        raise CampaignError(f"{root}: fetch-prebuilt-mlx.sh printed {result.stdout!r}")
    return {name: values[name] for name in wanted}


def build_environment(
    lane: Lane,
    run: str,
    root: Path,
    expected_sha: str,
    target_dir: str | None,
    sha_override: bool,
) -> dict[str, str]:
    """The environment every build and process of ``run`` gets (the knobs are added per row): the
    compile-time stamp — ``expected_sha`` and the dirty flag :func:`check_checkout` verified — or,
    with ``sha_override``, the operator's SHA and its permission."""
    environ = base_environment()
    if target_dir:
        environ["CARGO_TARGET_DIR"] = target_dir
    if sha_override:
        environ[GIT_SHA_ENV] = expected_sha
        environ[ALLOW_SHA_OVERRIDE_ENV] = "1"
    else:
        environ[BUILD_SHA_ENV] = expected_sha
        environ[BUILD_DIRTY_ENV] = "1" if run == "baseline" else "0"
    if lane.mlx:
        environ.update(mlx_prebuilt_environment(root, environ))
    return environ


def single_test_passed(text: str) -> bool:
    results = TEST_RESULT.findall(text)
    return len(results) == 1 and bool(PASSED.match(results[0]))


def run_process(
    *,
    cargo: str,
    lane: Lane,
    run: str,
    name: str,
    knobs: dict[str, str],
    environ: dict[str, str],
    cwd: Path,
    expected_sha: str,
    directory: Path,
    guard: list[str],
    sha_override: bool,
    cap_gib: float | None = None,
) -> None:
    """One benchmark process (under ``guard``, its ``{cap_gib}`` the row's cap); raises with the
    reason it does not count."""
    output = directory / f"{name}.json"
    log = directory / f"{name}.log"
    process_env = {**environ, **knobs, "SPECULATIVE_BENCH_OUTPUT": str(output)}
    if any("{cap_gib}" in part for part in guard):
        if cap_gib is None:
            raise CampaignError("the guard names {cap_gib} and the row sets no memory_cap_gib")
        guard = [part.replace("{cap_gib}", f"{cap_gib:g}") for part in guard]
    with log.open("wb") as sink:
        result = subprocess.run(
            [*guard, cargo, *cargo_args(lane, run)],
            cwd=cwd,
            env=process_env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log.read_bytes().decode("utf-8", errors="replace")
    sys.stdout.write(text)
    if result.returncode != 0:
        raise CampaignError(f"exited {result.returncode}")
    if not single_test_passed(text):
        raise CampaignError("did not run exactly one passing test")
    check_document(
        output,
        expected_sha=expected_sha,
        changes=sorted(lane.baseline_changes) if run == "baseline" else [],
        thinking=knobs["SPECULATIVE_BENCH_THINKING"],
        backend=lane.backend,
        sha_override=sha_override,
    )


GUARD_SUMMARY = re.compile(r"^GUARD_SUMMARY (.*)$", re.MULTILINE)


def guard_summary(log: Path) -> dict[str, Any] | None:
    """The external memory guard's closing ``GUARD_SUMMARY key=value …`` line in a process log
    (``scripts/ci/mlx_tree_guard.py``) as a dict — integers as integers — or ``None``."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = GUARD_SUMMARY.findall(text)
    if not lines:
        return None
    out: dict[str, Any] = {}
    for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', lines[-1]):
        value = value.strip('"')
        out[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return out


def write_summary(output: Path, summary: dict[str, Any]) -> None:
    """Write ``summary.json`` atomically (a temporary file, then a rename): the runner rewrites
    it after every process, so a campaign stopped part-way — a job timeout kills the runner —
    still leaves a whole summary of every process that finished."""
    temporary = output / "summary.json.tmp"
    with temporary.open("w", encoding="utf-8") as sink:
        json.dump(summary, sink, indent=2)
        sink.write("\n")
    os.replace(temporary, output / "summary.json")


def run_plan(
    plan: dict[str, Any],
    *,
    roots: dict[str, Path],
    target_dirs: dict[str, str | None],
    epic_sha: str,
    output: Path,
    cargo: str,
    guard: list[str],
    sha_override: bool,
    stop_starting_after: float | None = None,
    clock=None,
) -> int:
    """Check both checkouts, build them stamped, run every row (processes alternating between the
    runs), and write ``summary.json``. Returns the exit code.

    ``stop_starting_after`` (seconds from the end of the builds) bounds a loaned host
    (SW-7200): once it has passed, no further benchmark process starts — the one running is
    never interrupted, since signalling a GPU process mid-command-buffer can hang the device —
    the row it stopped in stays ``complete: false`` and the summary records why."""
    clock = clock or __import__("time").monotonic
    if plan.get("schema") != CAMPAIGN_SCHEMA or plan.get("document_schema") != SCHEMA:
        print(f"::error::the plan is not a {CAMPAIGN_SCHEMA} plan over {SCHEMA}", file=sys.stderr)
        return 1
    if (output / "summary.json").exists():
        print(f"::error::{output} already holds a campaign summary", file=sys.stderr)
        return 1
    lane = LANES[plan["lane"]]
    expected = {"epic": epic_sha, "baseline": PRE_EPIC_SHA}
    needed = [run for run in RUNS if any(run in row["runs"] for row in plan["rows"])]
    # `complete` stays false until the last row has run: a summary a stopped campaign left
    # behind says so, and so does the row it stopped during.
    summary: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "document_schema": SCHEMA,
        "lane": lane.name,
        "backend": lane.backend,
        "epic_sha": epic_sha,
        "pre_epic_sha": PRE_EPIC_SHA,
        "sha_override": sha_override,
        "complete": False,
        "builds": {},
        "rows": [],
    }
    environs: dict[str, dict[str, str]] = {}
    try:
        for run in needed:
            check_checkout(lane, run, roots[run], expected[run], roots["epic"])
        for run in needed:
            environs[run] = build_environment(
                lane, run, roots[run], expected[run], target_dirs.get(run), sha_override
            )
    except (CampaignError, OSError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    for run in needed:
        print(f"::group::build {run}")
        log = output / f"build-{run}.log"
        with log.open("wb") as sink:
            result = subprocess.run(
                [cargo, *cargo_args(lane, run, no_run=True)],
                cwd=roots[run],
                env=environs[run],
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
            )
        sys.stdout.write(log.read_bytes().decode("utf-8", errors="replace"))
        print("::endgroup::")
        summary["builds"][run] = {"status": "ok" if result.returncode == 0 else "failed"}
        if result.returncode != 0:
            print(f"::error::building {run} exited {result.returncode}", file=sys.stderr)
            write_summary(output, summary)
            return 1

    # The budget counts benchmark processes only: it starts once both builds are done.
    started = clock()
    failed = []
    for row in plan["rows"]:
        directory = output / row["id"]
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = row["env"]["SPECULATIVE_BENCH_SNAPSHOT"]
        entry: dict[str, Any] = {
            "id": row["id"],
            "runs": row["runs"],
            "process_repeats": row["process_repeats"],
            "switches": row["switches"],
            "format": row["env"]["SPECULATIVE_BENCH_FORMAT"],
            "env": row["env"],
            "snapshot": {"path": snapshot, "identity": None},
            "complete": False,
            "processes": [],
        }
        if row.get("prepare"):
            entry["snapshot"]["prepared_from"] = row["prepare"]["without_mtp"]
        summary["rows"].append(entry)
        write_summary(output, summary)
        try:
            if row.get("prepare"):
                prepare_without_mtp(row["prepare"]["without_mtp"], snapshot)
            identity = snapshot_identity(snapshot)["sha256"]
        except (CampaignError, OSError) as error:
            failed.append(f"{row['id']}: {error}")
            print(f"::error::{row['id']}: {error}")
            entry["complete"] = True
            write_summary(output, summary)
            continue
        entry["snapshot"]["identity"] = identity
        for index in range(1, row["process_repeats"] + 1):
            for run in row["runs"]:
                name = f"{run}-{index}"
                if stop_starting_after is not None and clock() - started > stop_starting_after:
                    reason = (
                        f"the time budget ({stop_starting_after / 60:.0f} min) passed before "
                        f"{row['id']} {name} started; no further process ran"
                    )
                    summary["stopped"] = reason
                    write_summary(output, summary)
                    print(f"::error::{reason}", file=sys.stderr)
                    for line in failed:
                        print(line, file=sys.stderr)
                    return 1
                print(f"::group::{row['id']} {name}")
                try:
                    run_process(
                        cargo=cargo,
                        lane=lane,
                        run=run,
                        name=name,
                        knobs=row["env"],
                        environ=environs[run],
                        cwd=roots[run],
                        expected_sha=expected[run],
                        directory=directory,
                        guard=guard,
                        sha_override=sha_override,
                        cap_gib=row.get("memory_cap_gib"),
                    )
                    if snapshot_identity(snapshot)["sha256"] != identity:
                        raise CampaignError("the snapshot's content changed during the process")
                    status, error = "ok", None
                except (CampaignError, OSError, ValueError) as failure:
                    status, error = "failed", str(failure)
                    failed.append(f"{row['id']} {name}: {error}")
                    print(f"::error::{row['id']} {name}: {error}")
                print("::endgroup::")
                entry["processes"].append(
                    {
                        "run": run,
                        "process": index,
                        "document": f"{row['id']}/{name}.json",
                        "status": status,
                        "error": error,
                        "snapshot_identity": identity,
                        "switches": row["switches"],
                        "guard": guard_summary(directory / f"{name}.log"),
                    }
                )
                write_summary(output, summary)
        entry["complete"] = True
        write_summary(output, summary)
    summary["complete"] = True
    write_summary(output, summary)
    for line in failed:
        print(line, file=sys.stderr)
    return 1 if failed else 0


def guard_prefix(text: str | None) -> list[str]:
    return shlex.split(text) if text else []


def command_plan(args: argparse.Namespace) -> int:
    text = os.environ.get(MATRIX_ENV, "")
    try:
        plan = plan_matrix(text, dict(os.environ), args.lane, require_local=True)
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
    if args.pre_epic_root is None and any("baseline" in row["runs"] for row in plan["rows"]):
        print("::error::a row runs the baseline: pass --pre-epic-root", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    roots = {"epic": Path(args.epic_root)}
    if args.pre_epic_root is not None:
        roots["baseline"] = Path(args.pre_epic_root)
    return run_plan(
        plan,
        roots=roots,
        target_dirs={"epic": args.epic_target_dir, "baseline": args.pre_epic_target_dir},
        epic_sha=args.epic_sha,
        output=output,
        cargo=args.cargo,
        guard=guard_prefix(args.guard),
        sha_override=args.allow_sha_override,
        stop_starting_after=(
            None if args.stop_starting_after_minutes is None
            else args.stop_starting_after_minutes * 60
        ),
    )


def command_local(args: argparse.Namespace) -> int:
    lane = LANES[args.lane]
    repo = Path(args.repo).resolve()
    work = Path(args.work_dir).resolve()
    output = Path(args.output).resolve()
    try:
        plan = plan_matrix(
            Path(args.matrix).read_text(encoding="utf-8"), dict(os.environ), lane.name
        )
    except MatrixError as error:
        print(f"::error::{args.matrix}: {error}", file=sys.stderr)
        return 1
    epic_sha = (git(repo, "rev-parse", f"{args.epic_ref}^{{commit}}") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", epic_sha):
        print(f"::error::{args.epic_ref} is not a commit of {repo}", file=sys.stderr)
        return 1
    roots = {"epic": work / "epic", "baseline": work / "pre-epic"}
    try:
        work.mkdir(parents=True, exist_ok=True)
        ensure_worktree(repo, roots["epic"], epic_sha)
        if any("baseline" in row["runs"] for row in plan["rows"]):
            ensure_worktree(repo, roots["baseline"], PRE_EPIC_SHA)
            apply_baseline_driver(lane, roots["epic"], roots["baseline"])
    except (CampaignError, OSError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x", encoding="utf-8") as sink:
        json.dump(plan, sink, indent=2)
        sink.write("\n")
    return run_plan(
        plan,
        roots=roots,
        target_dirs={"epic": str(work / "target-epic"), "baseline": str(work / "target-pre-epic")},
        epic_sha=epic_sha,
        output=output,
        cargo=args.cargo,
        guard=guard_prefix(args.guard),
        sha_override=args.allow_sha_override,
    )


# ---------------------------------------------------------------------------------------- compare

# Two-sided 95% Student-t critical values by degrees of freedom (Welch's df is floored).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}  # fmt: skip
METRICS = {"decode_tok_s": True, "ttft_ms": False}  # metric -> higher is better
NOT_COMPARABLE = {
    "auto": "`auto` changed meaning in the epic: pre-epic it runs MTP where the model has a head, "
    "else plain; the epic's falls back to prompt lookup. Reported, never given a verdict.",
}
VERDICT_ORDER = {"pass": 0, "inconclusive": 1, "regression": 2}
DEFAULT_MAX_NOISE = 0.10
REPO_ROOT = Path(__file__).resolve().parents[2]
# `models--<org>--<name>/snapshots/<revision>[/<subdir>]`: a Hugging Face cache snapshot path.
HF_SNAPSHOT = re.compile(r"models--([^/\\]+?)--([^/\\]+)[/\\]snapshots[/\\][^/\\]+((?:[/\\][^/\\]+)*)$")


def t_critical(df: float) -> float:
    if not math.isfinite(df) or df > 120:
        return 1.960
    floor = max(1, int(df))
    return _T95[max(k for k in _T95 if k <= floor)]


def series(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n if n else None
    stddev = None
    if n > 1 and mean is not None:
        stddev = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"n": n, "mean": mean, "stddev": stddev, "values": values}


def verdict(
    candidate: list[float], reference: list[float], higher_is_better: bool, max_noise: float
) -> dict[str, Any]:
    """``candidate`` vs ``reference``, each one value per process: ``regression`` when the
    candidate is worse by more than the measured noise band (Welch, two-sided 95%); ``pass`` when
    it is not worse, or worse within a band no wider than ``max_noise`` of the reference;
    ``inconclusive`` when a side has fewer than two processes or the band is too wide to resolve a
    regression."""
    a, b = series(candidate), series(reference)
    out: dict[str, Any] = {"candidate": a, "reference": b}
    if a["n"] < 2 or b["n"] < 2:
        # The band is process-level only; in-process jitter never stands in for it.
        n = min(a["n"], b["n"])
        return {**out, "verdict": "inconclusive", "reason": f"n={n}, no process-level band"}
    va, vb = a["stddev"] ** 2 / a["n"], b["stddev"] ** 2 / b["n"]
    se = math.sqrt(va + vb)
    df = se**4 / (va**2 / (a["n"] - 1) + vb**2 / (b["n"] - 1)) if se > 0 else math.inf
    margin = t_critical(df) * se
    delta = a["mean"] - b["mean"]
    worse = -delta if higher_is_better else delta
    scale = abs(b["mean"]) or 1.0
    out.update(
        {
            "delta": delta,
            "delta_pct": delta / scale * 100,
            "margin": margin,
            "margin_pct": margin / scale * 100,
        }
    )
    if worse > margin:
        return {**out, "verdict": "regression", "reason": "worse beyond the noise band"}
    if worse <= 0:
        return {**out, "verdict": "pass", "reason": "not worse"}
    if margin / scale > max_noise:
        return {
            **out,
            "verdict": "inconclusive",
            "reason": f"worse within a noise band wider than {max_noise:.0%}",
        }
    return {**out, "verdict": "pass", "reason": "worse within the measured noise band"}


def worst(verdicts: list[str]) -> str:
    return max(verdicts, key=VERDICT_ORDER.__getitem__) if verdicts else "inconclusive"


def option_key(option: Any) -> str:
    return option if isinstance(option, str) else json.dumps(option, sort_keys=True)


class Process(NamedTuple):
    campaign: str
    row: str
    run: str
    index: int
    document: dict[str, Any]
    label: dict[str, Any]  # the summary's process entry
    row_entry: dict[str, Any]  # the summary's row entry
    lane: str = ""  # the campaign summary's lane
    epic_sha: str = ""  # the campaign summary's measured commit


def family_label(process: Process) -> str:
    """The family a reader recognises: the document's model label, except that the harness's
    default label (the snapshot directory's name, ``@<format>`` unless bf16) of a Hugging Face
    cache snapshot — a revision hash or a bare subdirectory name — becomes ``<org>/<name>`` (plus
    the subdirectory and format suffix). An explicit ``SPECULATIVE_BENCH_MODEL`` is kept verbatim."""
    model = process.document.get("model")
    path = str((process.row_entry.get("snapshot") or {}).get("path") or "")
    fmt = process.row_entry.get("format") or "bf16"
    suffix = "" if fmt == "bf16" else f"@{fmt}"
    name = re.split(r"[/\\]", path.rstrip("/\\"))[-1] if path else ""
    match = HF_SNAPSHOT.search(path.rstrip("/\\"))
    if not isinstance(model, str) or not match or model != f"{name}{suffix}":
        return model if isinstance(model, str) else str(model)
    org, repo, subdir = match.groups()
    return f"{org}/{repo}{subdir.replace(chr(92), '/')}{suffix}"


def load_campaign(
    directory: Path,
) -> tuple[dict[str, Any], list[Process], list[str], list[str]]:
    """A campaign directory's summary and every process document, with every refusal, and the
    rows the campaign stopped during (``complete: false`` — a job timeout): those are set aside
    by name, never judged, so the rows that finished stay usable. A summary written before rows
    carried ``complete`` counts every row as finished."""
    refusals: list[str] = []
    incomplete: list[str] = []
    path = directory / "summary.json"
    if not path.is_file():
        return {}, [], [f"{directory}: no summary.json"], []
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != CAMPAIGN_SCHEMA or summary.get("document_schema") != SCHEMA:
        return (
            summary,
            [],
            [f"{directory}: the summary is not {CAMPAIGN_SCHEMA} over {SCHEMA}"],
            [],
        )
    for name, build in (summary.get("builds") or {}).items():
        if build.get("status") != "ok":
            refusals.append(f"{directory}: the {name} build failed")
    processes = []
    for row in summary.get("rows", []):
        tag = f"{directory.name}/{row['id']}"
        if row.get("complete") is False:
            ran = len(row.get("processes", []))
            declared = row["process_repeats"] * len(row["runs"])
            incomplete.append(
                f"{tag}: the campaign stopped during this row ({ran} of {declared} processes "
                "ran); not judged"
            )
            continue
        if not (row.get("snapshot") or {}).get("identity"):
            refusals.append(f"{tag}: no snapshot identity was recorded")
        counts = {run: 0 for run in row["runs"]}
        for entry in row.get("processes", []):
            name = f"{tag} {entry['run']}-{entry['process']}"
            if entry.get("status") != "ok":
                refusals.append(f"{name}: {entry.get('error') or entry.get('status')}")
                continue
            document_path = directory / entry["document"]
            if not document_path.is_file():
                refusals.append(f"{name}: {entry['document']} is missing")
                continue
            document = json.loads(document_path.read_text(encoding="utf-8"))
            if document.get("schema") != SCHEMA:
                refusals.append(f"{name}: schema {document.get('schema')!r}")
                continue
            if entry.get("snapshot_identity") != row["snapshot"]["identity"]:
                refusals.append(f"{name}: snapshot identity differs from the row's")
            # The commit a row is attributed to (and superseded by) is the summary's; every
            # document must have measured exactly that commit.
            measured = ((document.get("provenance") or {}).get("git") or {}).get("sha")
            expected = summary.get("epic_sha") if entry["run"] == "epic" else PRE_EPIC_SHA
            if measured != expected:
                refusals.append(f"{name}: measured {measured!r}, the summary says {expected!r}")
            counts[entry["run"]] += 1
            processes.append(
                Process(
                    directory.name,
                    row["id"],
                    entry["run"],
                    entry["process"],
                    document,
                    entry,
                    row,
                    summary.get("lane") or "",
                    summary.get("epic_sha") or "",
                )
            )
        for run, count in counts.items():
            if count != row["process_repeats"]:
                refusals.append(
                    f"{tag}: {count} {run} processes, the row declares {row['process_repeats']}"
                )
    return summary, processes, refusals, incomplete


def document_config(process: Process) -> dict[str, Any]:
    """What must be equal for two documents' timings to be compared."""
    doc = process.document
    provenance = doc.get("provenance") or {}
    return {
        "model": doc.get("model"),
        "snapshot": process.label.get("snapshot_identity"),
        "format": process.row_entry.get("format"),
        "backend": doc.get("backend"),
        "device": provenance.get("env"),
        "sampling": doc.get("sampling"),
        "thinking": (doc.get("thinking") or {}).get("setting"),
        "max_new_tokens": doc.get("max_new_tokens"),
        "repeats": doc.get("repeats"),
        "warmup": doc.get("warmup"),
        "options": [option_key(o) for o in doc.get("options") or []],
        "switch_env": {
            name: (value or {}).get("env")
            for name, value in (provenance.get("switches") or {}).items()
        },
        "prefix_checkpoints": _prefix_checkpoints_mode(process),
        "prompts": sorted({(r.get("prompt_id"), r.get("class")) for r in doc.get("rows") or []}),
    }


def rows_by_key(process: Process) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    table, duplicates = {}, []
    for row in process.document.get("rows") or []:
        key = (row.get("prompt_id"), option_key(row.get("requested")))
        if key in table:
            duplicates.append(f"duplicate row {key}")
        table[key] = row
    return table, duplicates


def check_row_evidence(processes: list[Process]) -> list[str]:
    """Every process of one campaign row (both runs): identical configuration — model, snapshot
    identity, format, backend, device, sampling, reasoning, budgets, options, switches and prompt
    set — and every (prompt, option) present exactly once."""
    refusals = []
    first = processes[0]
    reference = document_config(first)
    reference["prompts"] = sorted(
        {p for process in processes for p in document_config(process)["prompts"]}
    )
    by_run_checkpoints: dict[str, tuple[str, str | None]] = {}
    for process in processes:
        name = f"{process.campaign}/{process.row} {process.run}-{process.index}"
        config = document_config(process)
        for key, value in config.items():
            if key == "prefix_checkpoints":
                # Pre-epic baseline documents do not emit checkpoint telemetry (`None`), whereas
                # current epic documents emit their active policy (`auto`/`off`/`document`). Enforce
                # equality within each run without rejecting cross-revision epic-vs-baseline pairs.
                expected_run = by_run_checkpoints.get(process.run)
                if expected_run is None:
                    by_run_checkpoints[process.run] = (f"{process.run}-{process.index}", value)
                elif value != expected_run[1]:
                    refusals.append(
                        f"{name}: {key} {value!r} differs from {expected_run[1]!r} "
                        f"({expected_run[0]} / the row's prompt set)"
                    )
                continue
            if value != reference[key]:
                refusals.append(
                    f"{name}: {key} {value!r} differs from {reference[key]!r} "
                    f"({first.run}-{first.index} / the row's prompt set)"
                )
        table, duplicates = rows_by_key(process)
        refusals += [f"{name}: {d}" for d in duplicates]
        for prompt_id, _ in reference["prompts"]:
            for option in config["options"]:
                if (prompt_id, option) not in table:
                    refusals.append(f"{name}: missing row ({prompt_id}, {option})")
    return refusals


def _stats(process: Process, prompt_id: str, option: str, metric: str) -> dict[str, Any]:
    table, _ = rows_by_key(process)
    return (table.get((prompt_id, option)) or {}).get(metric) or {}


def metric_values(processes: list[Process], prompt_id: str, option: str, metric: str) -> list[float]:
    """One value per process: its in-process mean."""
    values = [_stats(p, prompt_id, option, metric).get("mean") for p in processes]
    return [float(v) for v in values if isinstance(v, (int, float))]


def _is_hit(sample: dict[str, Any], prompt_tokens: int) -> bool:
    """Whether a sample was a prefix-cache **hit**: the restored prefix covered at least half its
    prompt, so its prefill skipped most of it. Less — nothing, or the few chat-template tokens a
    different earlier prompt shares (3–4 on the MLX campaigns) — is a miss: its prefill ran
    (nearly) the whole prompt and it paid whatever the cache costs on that path."""
    return 2 * sample["prefix_hit_tokens"] >= prompt_tokens > 0


def sample_values(
    processes: list[Process], prompt_id: str, option: str, metric: str, hit: bool
) -> list[float]:
    """One value per process: the mean of its per-sample ``metric`` over the samples that hit the
    cross-turn prefix cache (``hit``, :func:`_is_hit`) or missed it. A process with no such
    sample contributes nothing."""
    values = []
    for process in processes:
        table, _ = rows_by_key(process)
        row = table.get((prompt_id, option)) or {}
        picked = [
            float(s[metric])
            for s in row.get("samples") or []
            if isinstance(s.get(metric), (int, float))
            and _is_hit(s, row["prompt_tokens"]) == hit
        ]
        if picked:
            values.append(sum(picked) / len(picked))
    return values


def _splits_by_hit(processes: list[Process], prompt_id: str, option: str) -> bool:
    """Whether every process's row for ``(prompt_id, option)`` records its prompt length and
    per-sample ``prefix_hit_tokens`` (a document without them is judged on its in-process
    mean)."""
    for process in processes:
        table, _ = rows_by_key(process)
        row = table.get((prompt_id, option)) or {}
        samples = row.get("samples") or []
        if not isinstance(row.get("prompt_tokens"), int) or not samples:
            return False
        if not all(isinstance(s.get("prefix_hit_tokens"), int) for s in samples):
            return False
    return True


def within_process_stddev(
    processes: list[Process], prompt_id: str, option: str, metric: str
) -> float | None:
    """The mean of the processes' in-process standard deviations (jitter, not the noise band)."""
    values = [_stats(p, prompt_id, option, metric).get("stddev") for p in processes]
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def is_defaults_row(row_entry: dict[str, Any]) -> bool:
    """A row measuring the build's defaults: no switch set and the harness's cold prefix cache."""
    env = row_entry.get("env") or {}
    return (
        not row_entry.get("switches")
        and env.get("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES", "0") == "0"
    )


def compare_epic_to_baseline(
    groups: dict[tuple[str, str], list[Process]], max_noise: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (row, prompt, comparable option, metric): the epic's processes vs the baseline's; the
    non-comparable options reported side by side."""
    comparisons, reported = [], []
    for (campaign, row_id), processes in sorted(groups.items()):
        epic = sorted((p for p in processes if p.run == "epic"), key=lambda p: p.index)
        baseline = sorted((p for p in processes if p.run == "baseline"), key=lambda p: p.index)
        if not epic or not baseline:
            continue
        config = document_config(epic[0])
        for prompt_id, prompt_class in config["prompts"]:
            for option in config["options"]:
                base = {
                    "campaign": campaign,
                    "row": row_id,
                    "epic_sha": epic[0].epic_sha,
                    "model": family_label(epic[0]),
                    "document_model": config["model"],
                    "backend": config["backend"],
                    "snapshot": config["snapshot"],
                    "defaults_row": is_defaults_row(epic[0].row_entry),
                    "prompt_id": prompt_id,
                    "class": prompt_class,
                    "option": option,
                }
                if option in NOT_COMPARABLE:
                    sides = {
                        metric: {
                            "epic": series(metric_values(epic, prompt_id, option, metric)),
                            "baseline": series(metric_values(baseline, prompt_id, option, metric)),
                        }
                        for metric in METRICS
                    }
                    reported.append({**base, "reason": NOT_COMPARABLE[option], **sides})
                    continue
                for metric, higher in METRICS.items():
                    result = verdict(
                        metric_values(epic, prompt_id, option, metric),
                        metric_values(baseline, prompt_id, option, metric),
                        higher,
                        max_noise,
                    )
                    result["epic"] = result.pop("candidate")
                    result["baseline"] = result.pop("reference")
                    for side, ps in (("epic", epic), ("baseline", baseline)):
                        result[side]["within_process_stddev"] = within_process_stddev(
                            ps, prompt_id, option, metric
                        )
                    comparisons.append({**base, "metric": metric, **result})
    return comparisons, reported


def e6_verdicts(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The E6 table: per (family, backend, prompt class, option) over the rows measuring the
    build's defaults, each metric's worst per-prompt verdict, and the overall worst."""
    table: dict[tuple, dict[str, list[str]]] = {}
    evidence: dict[tuple, set[str]] = {}
    for c in comparisons:
        if not c["defaults_row"]:
            continue
        key = (c["model"], c["backend"], c["class"], c["option"])
        table.setdefault(key, {m: [] for m in METRICS})[c["metric"]].append(c["verdict"])
        evidence.setdefault(key, set()).add(f"{c['campaign']}/{c['row']}@{c['epic_sha'][:12]}")
    out = []
    for (model, backend, prompt_class, option), metrics in sorted(table.items()):
        per_metric = {m: worst(v) for m, v in metrics.items()}
        out.append(
            {
                "model": model,
                "backend": backend,
                "class": prompt_class,
                "option": option,
                **per_metric,
                "verdict": worst(list(per_metric.values())),
                "rows": sorted(evidence[(model, backend, prompt_class, option)]),
            }
        )
    return out


# The defaults-table fields a campaign can measure, and how: a runtime switch (epic rows differing
# in its effective state), the load's prefix cache (epic rows differing in
# `load.prefix_cache_bytes`), or the speculative option (`auto` vs `off` in the same document).
FIELD_MEASUREMENT = {
    "cuda_graphs": ("switch", "CANDLE_LLM_CUDA_GRAPHS"),
    "device_positions": ("switch", "CANDLE_LLM_DEVICE_POSITIONS"),
    "fused_kernels": ("switch", "CANDLE_LLM_FUSED_KERNELS"),
    "nvfp4_gemv": ("switch", "CANDLE_LLM_NVFP4_GEMV"),
    "pipelining": ("switch", "MLX_LLM_PIPELINING"),
    "device_sampler": ("switch", "MLX_LLM_DEVICE_SAMPLER"),
    "fused_rotation": ("switch", "MLX_LLM_FUSED_ROTATION"),
    "gdn_kernel": ("switch", "MLX_LLM_GDN_KERNEL"),
    "prefix_cache_bytes": ("prefix_cache", None),
    "prefix_checkpoints": ("prefix_checkpoints", None),
    "speculative": ("speculative", None),
}
DEFAULTS_BACKENDS = {
    "MLX": "mlx",
    "CANDLE_CUDA": "candle-cuda",
    "CANDLE_METAL": "candle-metal",
    "CANDLE_CPU": "candle-cpu",
}


def provisional_defaults(source: str) -> list[dict[str, Any]]:
    """Every ``PROVISIONAL`` entry of the defaults table (``core-llm/src/defaults.rs``): its row
    and field, its current value read as on/off, and how a campaign measures it."""
    return [e for e in default_entries(source) if e["provisional"]]


def default_entries(source: str) -> list[dict[str, Any]]:
    """Every entry of the defaults table with its justification comment and whether that
    justification is ``PROVISIONAL``."""
    entries = []
    for const, backend in DEFAULTS_BACKENDS.items():
        match = re.search(
            rf"^pub const {const}: DecodeDefaults = DecodeDefaults \{{\n(.*?)^\}};",
            source,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            raise ValueError(f"no `{const}` row in the defaults table")
        comment: list[str] = []
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                comment.append(stripped)
                continue
            field = re.fullmatch(r"(\w+): (.+),", stripped)
            if field and comment:
                name, value = field.groups()
                kind, switch = FIELD_MEASUREMENT.get(name, (None, None))
                justification = " ".join(c.lstrip("/ ").strip() for c in comment)
                entries.append(
                    {
                        "entry": f"{const}.{name}",
                        "backend": backend,
                        "field": name,
                        "value": value,
                        "on": value not in ("false", "Speculative::Off", "0", "PrefixCheckpointMode::Off"),
                        "kind": kind,
                        "switch": switch,
                        "provisional": "PROVISIONAL" in justification,
                        "justification": justification.removeprefix("justification: "),
                    }
                )
            comment = []
    return entries


def _effective(process: Process, switch: str) -> Any:
    switches = (process.document.get("provenance") or {}).get("switches") or {}
    return (switches.get(switch) or {}).get("effective")


def _prefix_cache_on(process: Process) -> bool:
    return bool((process.document.get("load") or {}).get("prefix_cache_bytes") or 0)


def _prefix_checkpoints_mode(process: Process) -> str | None:
    env = process.row_entry.get("env") or {}
    mode = env.get("SPECULATIVE_BENCH_PREFIX_CHECKPOINTS")
    if mode in PREFIX_CHECKPOINTS:
        return mode
    rows = process.document.get("rows") or []
    policies = {
        (row.get("prefix_checkpoints") or {}).get("policy")
        for row in rows
        if isinstance(row.get("prefix_checkpoints"), dict)
    }
    if not policies:
        return None
    if policies == {"off"}:
        return "off"
    if policies <= {"auto", "document"}:
        return "auto"
    return None


def _context(
    process: Process,
    exclude_switch: str | None,
    include_cache: bool,
    *,
    include_checkpoints: bool = True,
) -> str:
    """Everything two epic rows of one campaign must share to be an on/off pair — twins: their
    whole configuration (options included) and process schedule (``runs``: a row alternating with
    pre-epic processes is a different measurement regime), but the setting under test."""
    config = document_config(process)
    config.pop("switch_env")
    config["runs"] = process.row_entry.get("runs")
    switches = (process.document.get("provenance") or {}).get("switches") or {}
    excluded = {exclude_switch} | {
        derived
        for derived in DERIVED_SWITCHES.get(exclude_switch or "", ())
        if (switches.get(derived) or {}).get("env") is None
    }
    config["effective"] = {
        name: (value or {}).get("effective")
        for name, value in switches.items()
        if name not in excluded
    }
    if include_cache:
        config["prefix_cache"] = _prefix_cache_on(process)
    if not include_checkpoints:
        config.pop("prefix_checkpoints", None)
    return json.dumps({"campaign": process.campaign, **config}, sort_keys=True, default=str)


def _deliberate(entry: dict[str, Any], processes: list[Process]) -> bool:
    """Whether the row set ``entry``'s setting on purpose (a switch in its ``switches``, or a
    non-zero prefix-cache budget): a row that did so and has no twin is a missing measurement."""
    if entry["kind"] == "switch":
        return entry["switch"] in (processes[0].row_entry.get("switches") or {})
    if entry["kind"] == "prefix_checkpoints":
        return _prefix_checkpoints_mode(processes[0]) == "auto"
    return entry["kind"] == "prefix_cache" and _prefix_cache_on(processes[0])


def _pairs(
    entry: dict[str, Any], rows: list[list[Process]]
) -> tuple[list[tuple[list[Process], list[Process], list[tuple[str, str]]]], list[list[Process]]]:
    """``entry``'s on/off pairs, each with the (on option, off option) pairs to judge, and the
    rows that set the setting deliberately but have no twin. The speculative option is ``auto`` vs
    ``off`` inside one row's processes; a switch or the prefix cache is two twin rows
    (:func:`_context`) differing in that setting, judged option by option."""
    if entry["kind"] == "speculative":
        pairs = [
            (processes, processes, [("auto", "off")])
            for processes in rows
            if {"auto", "off"} <= set(document_config(processes[0])["options"])
        ]
        return pairs, []
    buckets: dict[str, dict[bool, list[list[Process]]]] = {}
    for processes in rows:
        if entry["kind"] == "switch":
            state = _effective(processes[0], entry["switch"])
            if not isinstance(state, bool):
                continue
            context = _context(processes[0], entry["switch"], include_cache=True)
        elif entry["kind"] == "prefix_cache":
            state = _prefix_cache_on(processes[0])
            context = _context(processes[0], None, include_cache=False)
        elif entry["kind"] == "prefix_checkpoints":
            mode = _prefix_checkpoints_mode(processes[0])
            if mode not in PREFIX_CHECKPOINTS:
                continue
            state = mode == "auto"
            context = _context(processes[0], None, include_cache=True, include_checkpoints=False)
        else:
            return [], []
        buckets.setdefault(context, {}).setdefault(state, []).append(processes)
    pairs, unpaired = [], []
    for states in buckets.values():
        for on in states.get(True, []):
            for off in states.get(False, []):
                options = document_config(on[0])["options"]
                pairs.append((on, off, [(o, o) for o in options]))
        if len(states) == 1:
            unpaired += [p for ps in states.values() for p in ps if _deliberate(entry, p)]
    return pairs, unpaired


def _resolved_to_plain_decode(processes: list[Process], prompt_id: str, option: str) -> bool:
    """Whether ``option`` ran no proposer for ``prompt_id`` in every process (the DecodeReport's
    ``proposer`` is ``none``: e.g. ``auto`` on a model with no proposer is plain decode, so it is
    no evidence about speculation). A document without the field is not assumed either way."""
    proposers = [(rows_by_key(p)[0].get((prompt_id, option)) or {}).get("proposer") for p in processes]
    return bool(proposers) and all(proposer == "none" for proposer in proposers)


def _e5_outcome(entry: dict[str, Any], judged: list[dict[str, Any]]) -> dict[str, Any]:
    """E5: ON unless a measured on-vs-off regression (worse beyond the process-level noise band)
    justifies OFF. ``inconclusive`` and ``not_applicable`` are never a regression."""
    count = {v: sum(j["verdict"] == v for j in judged) for v in (*VERDICT_ORDER, "not_applicable")}
    unbanded = sum(j["reason"].endswith("no process-level band") for j in judged)
    tally = (
        f"{len(judged)} judged: {count['pass']} pass, {count['regression']} regression, "
        f"{count['inconclusive']} inconclusive ({unbanded} of them n<2, no process-level band), "
        f"{count['not_applicable']} not applicable"
    )
    tally += _sample_split_note(judged)
    regressions = [j for j in judged if j["verdict"] == "regression"]
    if not judged:
        reason = "no on/off pair of this setting in the campaigns"
        if not entry["kind"]:
            reason = "no measurement mapping"
        return {"outcome": "unmeasured", "recommended_on": None, "reason": reason}
    if regressions:
        where = "; ".join(
            f"{j['model']} {j['on_row']} vs {j['off_row']} {j['prompt_id']} `{j['option']}`"
            f"{_samples_label(j)} {j['metric']} {j['delta_pct']:+.1f}% "
            f"(band ±{j['margin_pct']:.1f}%)"
            for j in regressions
        )
        return {
            "outcome": "off",
            "recommended_on": False,
            "reason": f"measured on-vs-off regression ({tally}): {where}",
        }
    if count["pass"]:
        return {
            "outcome": "on",
            "recommended_on": True,
            "reason": f"no measured on-vs-off regression ({tally})",
        }
    return {
        "outcome": "on (unresolved)",
        "recommended_on": True,
        "reason": f"no on/off pair resolved and none regressed, so E5 keeps it on ({tally})",
    }


def _samples_label(judged: dict[str, Any]) -> str:
    samples = judged.get("samples")
    return f" [{samples} samples]" if samples else ""


def _sample_split_note(judged: list[dict[str, Any]]) -> str:
    """The prefix cache's miss and hit samples, each tallied with its measured Δ range per metric
    (a hit's gain is reported here; a regression on either decides the outcome)."""
    notes = []
    for samples in ("miss", "hit"):
        part = [j for j in judged if j.get("samples") == samples]
        if not part:
            continue
        verdicts = {v: sum(j["verdict"] == v for j in part) for v in VERDICT_ORDER}
        ranges = []
        for metric in METRICS:
            deltas = [j["delta_pct"] for j in part if j["metric"] == metric and j["delta_pct"] is not None]
            if deltas:
                ranges.append(f"{metric} Δ {min(deltas):+.1f}..{max(deltas):+.1f}%")
        notes.append(
            f"{samples} samples: {len(part)} judged, {verdicts['pass']} pass, "
            f"{verdicts['regression']} regression, {verdicts['inconclusive']} inconclusive"
            + (f" ({', '.join(ranges)})" if ranges else "")
        )
    return "; " + "; ".join(notes) if notes else ""


def _token_identity(
    pairs: list[tuple[list[Process], list[Process], list[tuple[str, str]]]],
) -> dict[str, Any]:
    """SW-6139: greedy token-stream identity across an on/off twin pair. Every process of both
    twins, every option (``off``, ``auto``, ``mtp``...), must stream the same token ids per prompt
    (the row's ``token_digest``): a kernel that changes accepted tokens is a defect, whatever its
    speed. Rows or documents without a digest, and non-greedy sampling, are not judged."""
    checked, mismatches = 0, []
    for on, off, _ in pairs:
        if (on[0].document.get("sampling") or {}).get("temperature") != 0.0:
            continue
        by_prompt: dict[str, dict[str, list[str]]] = {}
        for side, processes in (("on", on), ("off", off)):
            for process in processes:
                for (prompt_id, option), row in rows_by_key(process)[0].items():
                    digest = row.get("token_digest")
                    if isinstance(digest, str):
                        label = f"{process.campaign}/{process.row}:{side}:{option}"
                        by_prompt.setdefault(prompt_id, {}).setdefault(digest, []).append(label)
        for prompt_id, digests in sorted(by_prompt.items()):
            checked += 1
            if len(digests) > 1:
                mismatches.append(
                    f"{prompt_id}: "
                    + " != ".join(f"{d} ({', '.join(sorted(set(v)))})" for d, v in sorted(digests.items()))
                )
    return {"checked_prompts": checked, "mismatches": mismatches}


def decide_defaults(
    entries: list[dict[str, Any]],
    groups: dict[tuple[str, str], list[Process]],
    max_noise: float,
) -> list[dict[str, Any]]:
    """For each defaults entry: the on/off pairs of that setting the campaigns measured (epic
    processes only), each (prompt, option, metric) judged on vs off with the process-level Welch
    band E6 uses, and what E5 implies. Informational (already-justified) entries are reported only
    when a pair was measured."""
    epic_rows = []
    for processes in groups.values():
        epic = sorted((p for p in processes if p.run == "epic"), key=lambda p: p.index)
        if epic:
            epic_rows.append(epic)
    decisions = []
    for entry in entries:
        rows = [r for r in epic_rows if r[0].document.get("backend") == entry["backend"]]
        pairs, unpaired = _pairs(entry, rows)
        if not entry["provisional"] and not pairs and not unpaired:
            continue
        judged = []
        for on, off, option_pairs in pairs:
            config = document_config(on[0])
            for prompt_id, prompt_class in config["prompts"]:
                for on_option, off_option in option_pairs:
                    plain = entry["kind"] == "speculative" and _resolved_to_plain_decode(
                        on, prompt_id, on_option
                    )
                    # A prefix-cache row's samples are a cold miss (the first) and warm hits: a
                    # mean over both hides a miss-path regression behind the hits' gain, so each
                    # is judged on its own against the cache-off twin (every sample of which is
                    # a cold request, so its in-process mean is the reference for both).
                    splits: tuple[str | None, ...] = (None,)
                    if entry["kind"] == "prefix_cache" and _splits_by_hit(on, prompt_id, on_option):
                        splits = ("miss", "hit")
                    metrics_for_entry = list(METRICS.items())
                    if entry["kind"] == "prefix_checkpoints" and (
                        metric_values(on, prompt_id, on_option, "wall_ms")
                        or metric_values(off, prompt_id, off_option, "wall_ms")
                    ):
                        metrics_for_entry.append(("wall_ms", False))
                    cells = [(m, h, s) for m, h in metrics_for_entry for s in splits]
                    for metric, higher, samples in cells:
                        if plain:
                            result = {
                                "verdict": "not_applicable",
                                "reason": f"`{on_option}` resolved to plain decode (no proposer)",
                            }
                        else:
                            candidate = (
                                metric_values(on, prompt_id, on_option, metric)
                                if samples is None
                                else sample_values(
                                    on, prompt_id, on_option, metric, samples == "hit"
                                )
                            )
                            result = verdict(
                                candidate,
                                metric_values(off, prompt_id, off_option, metric),
                                higher,
                                max_noise,
                            )
                        judged.append(
                            {
                                "on_row": f"{on[0].campaign}/{on[0].row}",
                                "off_row": f"{off[0].campaign}/{off[0].row}",
                                "epic_sha": on[0].epic_sha,
                                "model": family_label(on[0]),
                                "class": prompt_class,
                                "prompt_id": prompt_id,
                                "option": f"{on_option} vs {off_option}"
                                if on_option != off_option
                                else on_option,
                                "metric": metric,
                                "samples": samples,
                                "on_n": len(on),
                                "off_n": len(off),
                                "verdict": result["verdict"],
                                "reason": result["reason"],
                                "delta_pct": result.get("delta_pct"),
                                "margin_pct": result.get("margin_pct"),
                            }
                        )
        outcome = _e5_outcome(entry, judged)
        if unpaired:
            if entry["kind"] == "prefix_cache":
                what = "the prefix cache"
            elif entry["kind"] == "prefix_checkpoints":
                what = "prefix checkpoints"
            else:
                what = entry["switch"]
            outcome["reason"] += (
                f"; unpaired, no twin row identical but for {what}: "
                + ", ".join(f"{p[0].campaign}/{p[0].row}" for p in unpaired)
            )
        identity = (
            _token_identity(pairs) if entry["kind"] == "switch" else None
        )
        if identity and identity["mismatches"]:
            outcome = {
                "outcome": "off",
                "recommended_on": False,
                "reason": "greedy token streams differ across the on/off twins: "
                + "; ".join(identity["mismatches"]),
            }
        recommended = outcome["recommended_on"]
        decisions.append(
            {
                **entry,
                "token_identity": identity,
                "informational": not entry["provisional"],
                "pairs": len(pairs),
                "unpaired": [f"{p[0].campaign}/{p[0].row}" for p in unpaired],
                **outcome,
                "flip": recommended is not None and recommended != entry["on"],
                "judged": judged,
            }
        )
    return decisions


_CHECKPOINT_NUMERIC_FIELDS = (
    "reused_tokens",
    "reread_tokens",
    "checkpoints_captured",
    "captured_bytes",
    "restore_ms",
    "capture_ms",
    "resident_bytes",
    "protected_bytes",
    "evicted_entries",
)


def _checkpoint_telemetry(processes: list[Process], prompt_id: str, option: str) -> dict[str, Any] | None:
    blocks = []
    for process in processes:
        table, _ = rows_by_key(process)
        pc = (table.get((prompt_id, option)) or {}).get("prefix_checkpoints")
        if isinstance(pc, dict):
            blocks.append(pc)
    return _summarize_checkpoint_blocks(blocks)


def _summarize_checkpoint_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not blocks:
        return None
    out: dict[str, Any] = {
        field: _mean([v for v in (_number(b.get(field)) for b in blocks) if v is not None])
        for field in _CHECKPOINT_NUMERIC_FIELDS
    }
    out["policies"] = sorted({b["policy"] for b in blocks if isinstance(b.get("policy"), str)})
    out["matched_kinds"] = sorted(
        {b["matched_kind"] for b in blocks if isinstance(b.get("matched_kind"), str)}
    )
    out["fallbacks"] = sorted(
        {f for b in blocks for f in (b.get("fallbacks") or []) if isinstance(f, str)}
    )
    return out


def prefix_checkpoints_paired(
    groups: dict[tuple[str, str], list[Process]], max_noise: float
) -> list[dict[str, Any]]:
    """Paired ``prefix_checkpoints`` on-vs-off summaries (wall latency, TTFT, decode rate,
    snapshot capture/restore overhead, reuse counters, memory footprint, and peak RAM)."""
    epic_rows = []
    for processes in groups.values():
        epic = sorted((p for p in processes if p.run == "epic"), key=lambda p: p.index)
        if epic:
            epic_rows.append(epic)
    stub_entry = {"kind": "prefix_checkpoints", "switch": None}
    out: list[dict[str, Any]] = []
    by_backend: dict[str, list[list[Process]]] = {}
    for processes in epic_rows:
        backend = processes[0].document.get("backend")
        if isinstance(backend, str):
            by_backend.setdefault(backend, []).append(processes)
    for backend, backend_rows in sorted(by_backend.items()):
        pairs, _ = _pairs(stub_entry, backend_rows)
        for on, off, option_pairs in pairs:
            config = document_config(on[0])
            for prompt_id, prompt_class in config["prompts"]:
                for on_option, off_option in option_pairs:
                    metrics: dict[str, Any] = {}
                    for metric, higher in (
                        ("wall_ms", False),
                        ("ttft_ms", False),
                        ("decode_tok_s", True),
                    ):
                        metrics[metric] = verdict(
                            metric_values(on, prompt_id, on_option, metric),
                            metric_values(off, prompt_id, off_option, metric),
                            higher,
                            max_noise,
                        )
                    on_peaks = metric_values(on, prompt_id, on_option, "peak_memory_bytes")
                    off_peaks = metric_values(off, prompt_id, off_option, "peak_memory_bytes")
                    out.append(
                        {
                            "backend": backend,
                            "model": family_label(on[0]),
                            "on_row": f"{on[0].campaign}/{on[0].row}",
                            "off_row": f"{off[0].campaign}/{off[0].row}",
                            "prompt_id": prompt_id,
                            "class": prompt_class,
                            "option": on_option if on_option == off_option else f"{on_option} vs {off_option}",
                            "on_n": len(on),
                            "off_n": len(off),
                            "wall_ms": metrics["wall_ms"],
                            "ttft_ms": metrics["ttft_ms"],
                            "decode_tok_s": metrics["decode_tok_s"],
                            "peak_memory_bytes": {
                                "on": series(on_peaks),
                                "off": series(off_peaks),
                            },
                            "prefix_checkpoints": {
                                "on": _checkpoint_telemetry(on, prompt_id, on_option),
                                "off": _checkpoint_telemetry(off, prompt_id, off_option),
                            },
                        }
                    )
    return out


_ADAPTIVE_COST_FIELDS = (
    "proposal_ms",
    "verify_ms",
    "rollback_ms",
    "sync_ms",
    "total_ms",
    "ms_per_committed_token",
)


def _summarize_adaptive_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not blocks:
        return None
    costs = [b["cost"] for b in blocks if isinstance(b.get("cost"), dict)]
    cost_summary = (
        {
            field: _mean([v for v in (_number(c.get(field)) for c in costs) if v is not None])
            for field in _ADAPTIVE_COST_FIELDS
        }
        if costs
        else None
    )
    return {
        "adaptive": any(bool(b.get("adaptive")) for b in blocks),
        "initial_proposers": sorted(
            {b["initial_proposer"] for b in blocks if isinstance(b.get("initial_proposer"), str)}
        ),
        "final_proposers": sorted(
            {b["final_proposer"] for b in blocks if isinstance(b.get("final_proposer"), str)}
        ),
        "initial_depth": _mean(
            [v for v in (_number(b.get("initial_depth")) for b in blocks) if v is not None]
        ),
        "final_depth": _mean(
            [v for v in (_number(b.get("final_depth")) for b in blocks) if v is not None]
        ),
        "depth_changes": _mean(
            [v for v in (_number(b.get("depth_changes")) for b in blocks) if v is not None]
        ),
        "proposer_switches": _mean(
            [v for v in (_number(b.get("proposer_switches")) for b in blocks) if v is not None]
        ),
        "plain_reasons": sorted(
            {b["plain_reason"] for b in blocks if isinstance(b.get("plain_reason"), str)}
        ),
        "cost": cost_summary,
    }


def _adaptive_telemetry(processes: list[Process], prompt_id: str, option: str) -> dict[str, Any] | None:
    blocks = []
    for process in processes:
        table, _ = rows_by_key(process)
        block = (table.get((prompt_id, option)) or {}).get("adaptive_speculative")
        if isinstance(block, dict):
            blocks.append(block)
    return _summarize_adaptive_blocks(blocks)


def _row_ratio_mean(processes: list[Process], prompt_id: str, option: str, field: str) -> float | None:
    vals = []
    for process in processes:
        table, _ = rows_by_key(process)
        row = table.get((prompt_id, option)) or {}
        if field == "tokens_per_pass":
            gen = _number(row.get("generated_tokens"))
            ver = _number(row.get("verify_steps"))
            if gen is not None and ver:
                vals.append(gen / ver)
        else:
            v = _number(row.get(field))
            if v is not None:
                vals.append(v)
    return _mean(vals)


def adaptive_speculative_paired(
    groups: dict[tuple[str, str], list[Process]], max_noise: float
) -> list[dict[str, Any]]:
    """Paired ``auto`` vs ``off`` (and ``epic/auto`` vs ``baseline/auto`` when present)
    adaptive speculative summaries (SW-7237 AC3): full-request time, TTFT, decode rate,
    accepted tokens per target pass, tokens per pass, peak memory, and adaptive transitions."""
    out: list[dict[str, Any]] = []
    for (_, _), processes in sorted(groups.items()):
        epic = sorted((p for p in processes if p.run == "epic"), key=lambda p: p.index)
        if not epic:
            continue
        config = document_config(epic[0])
        options = set(config["options"])
        if not ({"auto", "off"} <= options):
            continue
        has_adaptive = any(
            isinstance(
                (rows_by_key(p)[0].get((prompt_id, "auto")) or {}).get("adaptive_speculative"),
                dict,
            )
            for p in epic
            for prompt_id, _ in config["prompts"]
        )
        if not has_adaptive:
            continue
        baseline = sorted((p for p in processes if p.run == "baseline"), key=lambda p: p.index)
        backend = epic[0].document.get("backend")
        model = family_label(epic[0])
        row_label = f"{epic[0].campaign}/{epic[0].row}"
        for prompt_id, prompt_class in config["prompts"]:
            vs_off: dict[str, Any] = {}
            for metric, higher in (
                ("wall_ms", False),
                ("ttft_ms", False),
                ("decode_tok_s", True),
            ):
                vs_off[metric] = verdict(
                    metric_values(epic, prompt_id, "auto", metric),
                    metric_values(epic, prompt_id, "off", metric),
                    higher,
                    max_noise,
                )
            vs_baseline_auto: dict[str, Any] | None = None
            if baseline and "auto" in set(document_config(baseline[0])["options"]):
                vs_baseline_auto = {
                    metric: verdict(
                        metric_values(epic, prompt_id, "auto", metric),
                        metric_values(baseline, prompt_id, "auto", metric),
                        higher,
                        max_noise,
                    )
                    for metric, higher in (
                        ("wall_ms", False),
                        ("ttft_ms", False),
                        ("decode_tok_s", True),
                    )
                }
            out.append(
                {
                    "backend": backend,
                    "model": model,
                    "row": row_label,
                    "prompt_id": prompt_id,
                    "class": prompt_class,
                    "sampling": epic[0].document.get("sampling"),
                    "epic_n": len(epic),
                    "baseline_n": len(baseline),
                    "wall_ms": vs_off["wall_ms"],
                    "ttft_ms": vs_off["ttft_ms"],
                    "decode_tok_s": vs_off["decode_tok_s"],
                    "vs_baseline_auto": vs_baseline_auto,
                    "accepted_per_pass": _row_ratio_mean(
                        epic, prompt_id, "auto", "mean_accepted_length"
                    ),
                    "tokens_per_pass": _row_ratio_mean(
                        epic, prompt_id, "auto", "tokens_per_pass"
                    ),
                    "peak_memory_bytes": {
                        "auto": series(metric_values(epic, prompt_id, "auto", "peak_memory_bytes")),
                        "off": series(metric_values(epic, prompt_id, "off", "peak_memory_bytes")),
                    },
                    "adaptive_speculative": _adaptive_telemetry(epic, prompt_id, "auto"),
                }
            )
    return out


def _fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Decode-speedups campaign comparison (epic sc-24432 E5 / E6)",
        "",
        "Noise band: Welch's two-sided 95% interval over **process-level** repeats (each process's "
        "in-process mean is one sample; the in-process standard deviation is reported as "
        "within-process jitter only). A worse mean inside the band passes unless the band is "
        f"wider than {report['max_noise']:.0%} of the reference (then inconclusive). A side with "
        "fewer than two processes has no band and is inconclusive (`n=1, no process-level band`).",
        "",
        "## Campaigns",
        "",
        "| campaign | lane | epic | pre-epic | rows |",
        "|---|---|---|---|---|",
    ]
    for c in report["campaigns"]:
        lines.append(
            f"| {c['directory']} | {c['lane']} | `{c['epic_sha'][:12]}` | "
            f"`{c['pre_epic_sha'][:12]}` | {c['rows']} |"
        )
    lines += [
        "",
        "## Rows judged (each attributed to the epic commit it measured)",
        "",
        "| lane | row | campaign | epic | family | epic processes | baseline processes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['lane']} | {r['row']} | {r['campaign']} | `{r['epic_sha'][:12]}` | {r['model']} | "
            f"{r['processes']['epic']} | {r['processes']['baseline']} |"
        )
    lines += [
        "",
        "## Superseded (re-measured at a descendant commit; excluded from every verdict)",
        "",
        "| lane | row | campaign | epic | superseded by | why |",
        "|---|---|---|---|---|---|",
    ]
    for s in report["superseded"]:
        by_campaign, _, by_sha = s["superseded_by"].partition("@")
        lines.append(
            f"| {s['lane']} | {s['row']} | {s['campaign']} | `{s['epic_sha'][:12]}` | "
            f"{by_campaign} `{by_sha[:12]}` | {s['why']} |"
        )
    if not report["superseded"]:
        lines.append("| — | — | — | — | — | no row was measured by more than one campaign |")
    lines += [
        "",
        "## E6 — epic vs pre-epic baseline (rows at the build's defaults)",
        "",
        "| family | backend | class | option | decode tok/s | TTFT | verdict | rows (epic) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in report["e6"]:
        lines.append(
            f"| {e['model']} | {e['backend']} | {e['class']} | `{e['option']}` | "
            f"{e['decode_tok_s']} | {e['ttft_ms']} | **{e['verdict']}** | {', '.join(e['rows'])} |"
        )
    if not report["e6"]:
        lines.append("| — | — | — | — | — | — | no defaults row ran both revisions | — |")
    lines += [
        "",
        "### Per prompt",
        "",
        "| row | epic | prompt | option | metric | epic mean (n) | baseline mean (n) | Δ % | band % | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["comparisons"]:
        lines.append(
            f"| {c['campaign']}/{c['row']} | `{c['epic_sha'][:12]}` | {c['prompt_id']} | "
            f"`{c['option']}` | {c['metric']} | "
            f"{_fmt(c['epic']['mean'], 2)} ({c['epic']['n']}) | "
            f"{_fmt(c['baseline']['mean'], 2)} ({c['baseline']['n']}) | "
            f"{_fmt(c.get('delta_pct'))} | {_fmt(c.get('margin_pct'))} | {c['verdict']} |"
        )
    lines += [
        "",
        "## Not comparable (reported, no verdict)",
        "",
        "| row | prompt | option | epic tok/s | baseline tok/s | epic TTFT ms | baseline TTFT ms | why |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["not_comparable"]:
        lines.append(
            f"| {r['campaign']}/{r['row']} | {r['prompt_id']} | `{r['option']}` | "
            f"{_fmt(r['decode_tok_s']['epic']['mean'], 2)} | "
            f"{_fmt(r['decode_tok_s']['baseline']['mean'], 2)} | "
            f"{_fmt(r['ttft_ms']['epic']['mean'], 2)} | {_fmt(r['ttft_ms']['baseline']['mean'], 2)} | "
            f"{r['reason']} |"
        )
    lines += [
        "",
        "## E5 — defaults decided from on-vs-off pairs of the same setting",
        "",
        "PROVISIONAL entries are decided here; an already-justified entry with a measured pair is "
        "listed as *informational* beside its justification. A default is ON unless an on-vs-off "
        "pair regresses beyond the noise band; `inconclusive` and `not_applicable` never do.",
        "",
        "| entry | current | pairs | outcome | recommended | flip | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in report["defaults"]:
        recommended = {True: "on", False: "off", None: "—"}[d["recommended_on"]]
        entry = f"`{d['entry']}`"
        if d["informational"]:
            entry += f" (informational; justified: {d['justification']})"
        lines.append(
            f"| {entry} | `{d['value']}` | {d['pairs']} | {d['outcome']} | {recommended} | "
            f"{'**yes**' if d['flip'] else 'no'} | {d['reason']} |"
        )
    lines += [
        "",
        "### E5 judged pairs",
        "",
        "| entry | family | on row | off row | prompt | option | metric | n on/off | Δ % | band % | verdict | why |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in report["defaults"]:
        for j in d["judged"]:
            lines.append(
                f"| `{d['entry']}` | {j['model']} | {j['on_row']} | {j['off_row']} | "
                f"{j['prompt_id']} | `{j['option']}`{_samples_label(j)} | {j['metric']} | "
                f"{j['on_n']}/{j['off_n']} | "
                f"{_fmt(j['delta_pct'])} | {_fmt(j['margin_pct'])} | {j['verdict']} | {j['reason']} |"
            )
    if report.get("prefix_checkpoints"):
        gib = 1024**3
        lines += [
            "",
            "## Prefix checkpoints — paired on vs off rows",
            "",
            "| backend | family | on / off | prompt | option | wall ms (on vs off, Δ %) | TTFT ms (on vs off, Δ %) | decode tok/s (on vs off, Δ %) | peak mem GiB (on / off) | reused / reread | capture ms | restore ms | captured / resident / protected bytes |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for item in report["prefix_checkpoints"]:
            def _cmp_cell(m: dict[str, Any], digits: int = 1) -> str:
                c = (m.get("candidate") or {}).get("mean")
                r = (m.get("reference") or {}).get("mean")
                d = m.get("delta_pct")
                delta = "—" if d is None else f"{d:+.1f}%"
                return f"{_fmt(c, digits)} vs {_fmt(r, digits)} ({delta})"

            on_peak = ((item.get("peak_memory_bytes") or {}).get("on") or {}).get("mean")
            off_peak = ((item.get("peak_memory_bytes") or {}).get("off") or {}).get("mean")
            peak_cell = (
                f"{'—' if on_peak is None else f'{on_peak / gib:.2f}'} / "
                f"{'—' if off_peak is None else f'{off_peak / gib:.2f}'}"
            )
            on_pc = (item.get("prefix_checkpoints") or {}).get("on") or {}
            reused = _fmt(on_pc.get("reused_tokens"), 0)
            reread = _fmt(on_pc.get("reread_tokens"), 0)
            kinds = ",".join(on_pc.get("matched_kinds") or [])
            reuse_cell = f"{reused} / {reread}" + (f" ({kinds})" if kinds else "")
            cap_ms = _fmt(on_pc.get("capture_ms"), 2)
            res_ms = _fmt(on_pc.get("restore_ms"), 2)
            bytes_cell = (
                f"{_fmt(on_pc.get('captured_bytes'), 0)} / "
                f"{_fmt(on_pc.get('resident_bytes'), 0)} / "
                f"{_fmt(on_pc.get('protected_bytes'), 0)}"
            )
            lines.append(
                f"| {item['backend']} | {item['model']} | {item['on_row']} / {item['off_row']} | "
                f"{item['prompt_id']} | `{item['option']}` | "
                f"{_cmp_cell(item['wall_ms'], 1)} | {_cmp_cell(item['ttft_ms'], 1)} | "
                f"{_cmp_cell(item['decode_tok_s'], 2)} | {peak_cell} | {reuse_cell} | "
                f"{cap_ms} | {res_ms} | {bytes_cell} |"
            )
    if report.get("adaptive_speculative"):
        gib = 1024**3
        lines += [
            "",
            "## Adaptive speculative — paired auto vs off (SW-7237)",
            "",
            "| backend | family | row | prompt | wall ms (auto vs off, Δ %) | TTFT ms (auto vs off, Δ %) | decode tok/s (auto vs off, Δ %) | accepted/pass | tok/pass | peak mem GiB (auto / off) | initial -> final | depthΔ / switches |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for item in report["adaptive_speculative"]:
            def _cmp_adaptive(m: dict[str, Any], digits: int = 1) -> str:
                c = (m.get("candidate") or {}).get("mean")
                r = (m.get("reference") or {}).get("mean")
                d = m.get("delta_pct")
                delta = "—" if d is None else f"{d:+.1f}%"
                return f"{_fmt(c, digits)} vs {_fmt(r, digits)} ({delta})"

            auto_peak = ((item.get("peak_memory_bytes") or {}).get("auto") or {}).get("mean")
            off_peak = ((item.get("peak_memory_bytes") or {}).get("off") or {}).get("mean")
            peak_cell = (
                f"{'—' if auto_peak is None else f'{auto_peak / gib:.2f}'} / "
                f"{'—' if off_peak is None else f'{off_peak / gib:.2f}'}"
            )
            ad = item.get("adaptive_speculative") or {}
            init_p = ",".join(ad.get("initial_proposers") or []) or "—"
            fin_p = ",".join(ad.get("final_proposers") or []) or "—"
            transitions = (
                f"{_fmt(ad.get('depth_changes'), 1)} / {_fmt(ad.get('proposer_switches'), 1)}"
            )
            lines.append(
                f"| {item['backend']} | {item['model']} | {item['row']} | {item['prompt_id']} | "
                f"{_cmp_adaptive(item['wall_ms'], 1)} | {_cmp_adaptive(item['ttft_ms'], 1)} | "
                f"{_cmp_adaptive(item['decode_tok_s'], 2)} | {_fmt(item.get('accepted_per_pass'), 2)} | "
                f"{_fmt(item.get('tokens_per_pass'), 2)} | {peak_cell} | {init_p} -> {fin_p} | "
                f"{transitions} |"
            )
        baseline_auto_items = [
            item for item in report["adaptive_speculative"] if item.get("vs_baseline_auto")
        ]
        if baseline_auto_items:
            lines += [
                "",
                "### Adaptive speculative — candidate auto vs baseline auto",
                "",
                "| backend | family | row | prompt | wall ms (cand vs base, Δ %) | TTFT ms (cand vs base, Δ %) | decode tok/s (cand vs base, Δ %) |",
                "|---|---|---|---|---|---|---|",
            ]
            for item in baseline_auto_items:
                vba = item["vs_baseline_auto"]
                lines.append(
                    f"| {item['backend']} | {item['model']} | {item['row']} | {item['prompt_id']} | "
                    f"{_cmp_adaptive(vba['wall_ms'], 1)} | {_cmp_adaptive(vba['ttft_ms'], 1)} | "
                    f"{_cmp_adaptive(vba['decode_tok_s'], 2)} |"
                )
    if report.get("incomplete"):
        lines += [
            "",
            "## Incomplete (the campaign stopped during them; excluded from every verdict)",
            "",
        ] + [f"- {row}" for row in report["incomplete"]]
    if report["notes"]:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """``git merge-base --is-ancestor``; refuses a commit the repository does not know."""
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise Refused(
            f"cannot order {ancestor} and {descendant} in {repo}: "
            f"{result.stderr.strip() or f'git exited {result.returncode}'}"
        )
    return result.returncode == 0


def _measurement(processes: list[Process]) -> list[tuple[str, int, str]]:
    return sorted((p.run, p.index, json.dumps(p.document, sort_keys=True)) for p in processes)


def supersede(
    groups: dict[tuple[str, str], list[Process]], repo: Path
) -> tuple[dict[tuple[str, str], list[Process]], list[dict[str, Any]], list[str]]:
    """One measurement per (lane, row id): where campaigns measured the same row, the one at the
    descendant epic commit supersedes the others. Refused when the commits are unordered (neither
    is an ancestor of the other) or equal with different data; a byte-identical copy at the same
    commit is a duplicate. Rows no later campaign re-measured stay, attributed to their commit."""
    by_row: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (campaign, row_id), processes in groups.items():
        by_row.setdefault((processes[0].lane, row_id), []).append((campaign, processes[0].epic_sha))
    kept = dict(groups)
    superseded, refusals = [], []
    for (lane, row_id), measured in by_row.items():
        if len(measured) < 2:
            continue
        tag = f"{lane} row {row_id}"
        try:
            same = [
                (a, b)
                for i, a in enumerate(measured)
                for b in measured[i + 1 :]
                if a[1] == b[1]
                and _measurement(groups[(a[0], row_id)]) != _measurement(groups[(b[0], row_id)])
            ]
            if same:
                (a, sha), (b, _) = same[0]
                raise Refused(f"{tag}: {a} and {b} both measured {sha} with different data")
            newest = [
                (campaign, sha)
                for campaign, sha in measured
                if all(other == sha or is_ancestor(repo, other, sha) for _, other in measured)
            ]
            if not newest:
                shas = ", ".join(f"{c}@{s}" for c, s in measured)
                raise Refused(f"{tag}: no measurement descends from all the others ({shas})")
        except Refused as refused:
            refusals.append(str(refused))
            continue
        winner = newest[0]
        for campaign, sha in measured:
            if (campaign, sha) == winner:
                continue
            del kept[(campaign, row_id)]
            superseded.append(
                {
                    "lane": lane,
                    "row": row_id,
                    "campaign": campaign,
                    "epic_sha": sha,
                    "superseded_by": f"{winner[0]}@{winner[1]}",
                    "why": "duplicate at the same commit" if sha == winner[1] else "older commit",
                }
            )
    return kept, superseded, refusals


def compare_campaigns(
    directories: list[Path],
    defaults_source: str,
    max_noise: float = DEFAULT_MAX_NOISE,
    repo: Path = REPO_ROOT,
) -> dict[str, Any]:
    """The comparison report over campaign directories; raises :class:`Refused` listing every
    reason the evidence cannot be judged. ``repo`` orders the campaigns' epic commits."""
    refusals: list[str] = []
    incomplete: list[str] = []
    campaigns, groups = [], {}
    for directory in directories:
        summary, processes, problems, stopped = load_campaign(directory)
        refusals += problems
        incomplete += stopped
        campaigns.append(
            {
                "directory": directory.name,
                "lane": summary.get("lane"),
                "epic_sha": summary.get("epic_sha") or "",
                "pre_epic_sha": summary.get("pre_epic_sha") or "",
                "rows": len(summary.get("rows", [])),
                "sha_override": summary.get("sha_override"),
                "complete": summary.get("complete", True),
            }
        )
        if summary and summary.get("pre_epic_sha") != PRE_EPIC_SHA:
            refusals.append(f"{directory}: pre-epic {summary.get('pre_epic_sha')} is not {PRE_EPIC_SHA}")
        for process in processes:
            groups.setdefault((process.campaign, process.row), []).append(process)
    if len({c["directory"] for c in campaigns}) != len(campaigns):
        refusals.append("two campaign directories share a name")
    if incomplete and not groups:
        refusals.append("no row ran to its end: " + "; ".join(incomplete))
    for processes in groups.values():
        refusals += check_row_evidence(processes)
    groups, superseded, problems = supersede(groups, repo)
    refusals += problems
    if refusals:
        raise Refused("\n".join(refusals))
    rows = []
    for (campaign, row_id), processes in sorted(groups.items()):
        rows.append(
            {
                "lane": processes[0].lane,
                "row": row_id,
                "campaign": campaign,
                "epic_sha": processes[0].epic_sha,
                "model": family_label(processes[0]),
                "processes": {run: sum(p.run == run for p in processes) for run in RUNS},
            }
        )
    notes = []
    for (campaign, row_id), processes in sorted(groups.items()):
        tokens = {
            run: sorted(
                {
                    (r.get("prompt_id"), r.get("prompt_tokens"))
                    for p in processes
                    if p.run == run
                    for r in p.document["rows"]
                }
            )
            for run in RUNS
        }
        if tokens["epic"] and tokens["baseline"] and tokens["epic"] != tokens["baseline"]:
            notes.append(
                f"{campaign}/{row_id}: prompt token counts differ between the revisions "
                f"(a chat-template change?): {tokens}"
            )
    comparisons, not_comparable = compare_epic_to_baseline(groups, max_noise)
    entries = [e for e in default_entries(defaults_source) if e["provisional"] or e["kind"]]
    defaults = decide_defaults(entries, groups, max_noise)
    return {
        "schema": COMPARE_SCHEMA,
        "max_noise": max_noise,
        "confidence": 0.95,
        "campaigns": campaigns,
        "rows": rows,
        "superseded": superseded,
        "e6": e6_verdicts(comparisons),
        "comparisons": comparisons,
        "not_comparable": not_comparable,
        "defaults": defaults,
        "prefix_checkpoints": prefix_checkpoints_paired(groups, max_noise),
        "adaptive_speculative": adaptive_speculative_paired(groups, max_noise),
        "incomplete": incomplete,
        "notes": notes,
    }


# ------------------------------------------------------------------------------------------ paths
#
# SW-7200: the paired speculative-paths comparison. Every path of one campaign row runs in the same
# process on the same prompts, seeds and budgets, so each path is read against the `off` row of its
# own document (the in-process pair); rows that measured the same target, tier and context in one
# host session (a DSpine process and a DFlash2 process, say) meet through those `off` ratios.

_TIERS = re.compile(r"q\d+|bf16|nvfp4")


def path_label(row: dict[str, Any]) -> str:
    """The path a document row measured, from what it asked for: ``off``, ``auto``, the proposer
    (``mtp``, ``prompt_lookup``, ``draft_model``), and for a block draft its format (from the
    report's ``draft``) and verify layout — ``block_draft:dspine:tree``. A path that asked for a
    tree says so whatever the proposer; one asked at the recommended depth (the depth ``auto``
    runs) ends ``@recommended``."""
    requested = row.get("requested")
    if isinstance(requested, str):
        return requested
    proposer = (requested or {}).get("proposer") or (requested or {}).get("mode") or "?"
    verify = row.get("verify") or (requested or {}).get("verify") or "linear"
    tree = isinstance(verify, dict) and "tree" in verify
    suffix = "@recommended" if (requested or {}).get("depth") == "recommended" else ""
    if proposer == "block_draft":
        fmt = (row.get("draft") or {}).get("format") or "unloaded"
        return f"block_draft:{fmt}:{'tree' if tree else 'linear'}{suffix}"
    return f"{proposer}:tree{suffix}" if tree else f"{proposer}{suffix}"


def row_tier(process: Process) -> str:
    """The target's tier: the load-time format, else (a prepared snapshot loaded as stored) the
    snapshot's tier folder (``…/q4``), else ``bf16``."""
    fmt = process.row_entry.get("format") or "bf16"
    if fmt != "bf16":
        return fmt
    path = str((process.row_entry.get("snapshot") or {}).get("path") or "")
    leaf = re.split(r"[/\\]", path.rstrip("/\\"))[-1] if path else ""
    return leaf if _TIERS.fullmatch(leaf) else "bf16"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def path_measurement(row: dict[str, Any], context: str) -> dict[str, Any] | None:
    """One document row's measurement over its ``context`` samples (``cold``: the prefill ran the
    whole prompt; ``reused``: the cross-turn prefix cache restored most of it, :func:`_is_hit`), or
    ``None`` when no sample is of that context."""
    prompt_tokens = row.get("prompt_tokens")
    samples = [
        sample
        for sample in row.get("samples") or []
        if (
            isinstance(prompt_tokens, int)
            and isinstance(sample.get("prefix_hit_tokens"), int)
            and _is_hit(sample, prompt_tokens)
        )
        == (context == "reused")
    ]
    if not samples:
        return None

    def values(key: str) -> list[float]:
        return [v for v in (_number(sample.get(key)) for sample in samples) if v is not None]

    verify_steps = _number(row.get("verify_steps"))
    generated = _number(row.get("generated_tokens"))
    draft_time = row.get("draft_time") or {}
    tree = row.get("tree_verify") or {}
    decode_ms = _mean(values("decode_ms"))
    draft_ms = _mean(values("draft_ms"))
    added = [
        peak - before
        for peak, before in (
            (_number(s.get("peak_memory_bytes")), _number(s.get("memory_before_bytes")))
            for s in samples
        )
        if peak is not None and before is not None
    ]
    return {
        "samples": len(samples),
        "wall_ms": _mean(values("wall_ms")),
        "ttft_ms": _mean(values("ttft_ms")),
        "decode_tok_s": _mean(values("decode_tok_s")),
        "accepted_per_pass": _number(row.get("mean_accepted_length")),
        "tokens_per_pass": generated / verify_steps if generated and verify_steps else None,
        "accepted_per_tree_pass": _number(tree.get("accepted_per_tree_pass")),
        "rows_per_pass": _number(tree.get("rows_per_pass")),
        "draft_ms": draft_ms,
        "draft_ms_per_pass": _number(draft_time.get("per_call_ms")),
        "draft_share_of_decode": draft_ms / decode_ms if draft_ms is not None and decode_ms else None,
        "draft_time_host_synchronized": (
            draft_time.get("calls") == draft_time.get("host_synchronized_calls")
            if draft_time
            else None
        ),
        "peak_memory_bytes": max(values("peak_memory_bytes"), default=None),
        "memory_added_bytes": max(added, default=None),
        "prefix_checkpoints": (
            row.get("prefix_checkpoints")
            if isinstance(row.get("prefix_checkpoints"), dict)
            else None
        ),
        "adaptive_speculative": (
            row.get("adaptive_speculative")
            if isinstance(row.get("adaptive_speculative"), dict)
            else None
        ),
    }


def paths_report(directories: list[Path]) -> dict[str, Any]:
    """The paired speculative-paths report over campaign ``directories`` (epic documents only):
    one entry per (campaign row, path, prompt, context) with its measurement and its ratios to
    the same document's ``off`` row; a summary per (backend, tier, context, path) over prompts;
    and the block-draft inputs ``auto``'s decision reads (each block-draft path's decode-rate and
    full-request ratios to ``off`` and to MTP, the MTP ratio taken from a process of the same
    tier and context — both normalized by their own process's ``off``)."""
    refusals: list[str] = []
    incomplete: list[str] = []
    entries: list[dict[str, Any]] = []
    for directory in directories:
        summary, processes, refused, stopped = load_campaign(directory)
        refusals += refused
        incomplete += stopped
        for process in processes:
            if process.run != "epic":
                continue
            document = process.document
            env = process.row_entry.get("env") or {}
            group = {
                "backend": document.get("backend"),
                "model": family_label(process),
                "tier": row_tier(process),
                "max_new_tokens": document.get("max_new_tokens"),
                "thinking": (document.get("thinking") or {}).get("setting"),
                "sampling": document.get("sampling"),
                "prefix_cache_bytes": env.get("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES", "0"),
                "prompt_set": env.get("SPECULATIVE_BENCH_PROMPT_SET", "default"),
                "prefix_checkpoints": _prefix_checkpoints_mode(process) or "default",
            }
            load = document.get("load") or {}
            offs = {
                row.get("prompt_id"): row
                for row in document.get("rows") or []
                if row.get("requested") == "off"
            }
            for row in document.get("rows") or []:
                for context in ("cold", "reused"):
                    measured = path_measurement(row, context)
                    if measured is None:
                        continue
                    off = offs.get(row.get("prompt_id"))
                    off_measured = path_measurement(off, context) if off else None
                    ratios = {}
                    if off_measured:
                        for key, higher_better in (("decode_tok_s", True), ("wall_ms", False)):
                            a, b = measured[key], off_measured[key]
                            if a and b:
                                ratios[key] = a / b if higher_better else b / a
                    entries.append(
                        {
                            "campaign": process.campaign,
                            "row": process.row,
                            "process": process.index,
                            "group": group,
                            "context": context,
                            "draft_source": (load.get("draft") or {}).get("source"),
                            "draft_refusal": (load.get("draft") or {}).get("refusal"),
                            "path": path_label(row),
                            "requested": row.get("requested"),
                            "speculative": row.get("speculative"),
                            "verify": row.get("verify"),
                            "ran": row.get("proposer"),
                            "draft": row.get("draft"),
                            "prompt_id": row.get("prompt_id"),
                            "class": row.get("class"),
                            "fallbacks": row.get("fallbacks") or [],
                            "measurement": measured,
                            "speedup_vs_off": ratios,
                        }
                    )
    return {
        "schema": PATHS_SCHEMA,
        "campaigns": [str(d) for d in directories],
        "refusals": refusals,
        "incomplete": incomplete,
        "entries": entries,
        "summary": summarize_paths(entries),
        "auto_inputs": auto_inputs(entries),
    }


def _group_key(entry: dict[str, Any]) -> str:
    return json.dumps({**entry["group"], "context": entry["context"]}, sort_keys=True)


def summarize_paths(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (group, context, path): the mean of each measurement over its prompts and processes,
    the geometric mean of the ``off`` ratios over all prompts and per prompt class, every distinct
    fallback, and whether every prompt ran the path it asked for."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        buckets.setdefault((_group_key(entry), entry["path"]), []).append(entry)
    out = []
    for (key, path), bucket in sorted(buckets.items()):
        first = bucket[0]

        def mean_of(field: str, bucket=bucket) -> float | None:
            return _mean([e["measurement"][field] for e in bucket if e["measurement"][field] is not None])

        ratios: dict[str, Any] = {}
        for metric in ("decode_tok_s", "wall_ms"):
            ratios[metric] = {
                "all": _geomean([e["speedup_vs_off"][metric] for e in bucket if metric in e["speedup_vs_off"]]),
                **{
                    cls: _geomean(
                        [
                            e["speedup_vs_off"][metric]
                            for e in bucket
                            if e["class"] == cls and metric in e["speedup_vs_off"]
                        ]
                    )
                    for cls in ("predictable", "open_ended")
                },
            }
        out.append(
            {
                "group": first["group"],
                "context": first["context"],
                "path": path,
                "prompts": len({e["prompt_id"] for e in bucket}),
                "processes": len({(e["campaign"], e["row"], e["process"]) for e in bucket}),
                "ran": sorted({str(e["ran"]) for e in bucket}),
                "draft": first["draft"],
                "draft_refusal": first["draft_refusal"],
                "fallbacks": sorted({f for e in bucket for f in e["fallbacks"]}),
                **{
                    field: mean_of(field)
                    for field in (
                        "wall_ms",
                        "ttft_ms",
                        "decode_tok_s",
                        "accepted_per_pass",
                        "tokens_per_pass",
                        "accepted_per_tree_pass",
                        "rows_per_pass",
                        "draft_ms_per_pass",
                        "draft_share_of_decode",
                    )
                },
                "draft_time_host_synchronized": all(
                    e["measurement"]["draft_time_host_synchronized"] is not False for e in bucket
                ),
                "peak_memory_bytes": max(
                    (e["measurement"]["peak_memory_bytes"] for e in bucket if e["measurement"]["peak_memory_bytes"] is not None),
                    default=None,
                ),
                "memory_added_bytes": max(
                    (e["measurement"]["memory_added_bytes"] for e in bucket if e["measurement"]["memory_added_bytes"] is not None),
                    default=None,
                ),
                "prefix_checkpoints": _summarize_path_checkpoints(bucket),
                "adaptive_speculative": _summarize_path_adaptive(bucket),
                "speedup_vs_off": ratios,
            }
        )
    return out


def _summarize_path_adaptive(bucket: list[dict[str, Any]]) -> dict[str, Any] | None:
    blocks = [
        e["measurement"]["adaptive_speculative"]
        for e in bucket
        if isinstance(e["measurement"].get("adaptive_speculative"), dict)
    ]
    return _summarize_adaptive_blocks(blocks)


def _summarize_path_checkpoints(bucket: list[dict[str, Any]]) -> dict[str, Any] | None:
    blocks = [
        e["measurement"]["prefix_checkpoints"]
        for e in bucket
        if isinstance(e["measurement"].get("prefix_checkpoints"), dict)
    ]
    return _summarize_checkpoint_blocks(blocks)


AUTO_MTP_PATH = "mtp@recommended"


def auto_inputs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (backend, tier, context, block draft at its recommended depth, linear — what ``auto``
    would run): the decode-rate and full-request ratios to ``off`` (geometric means over prompts)
    and to MTP at its recommended depth in the same group (:data:`AUTO_MTP_PATH`; each path's ratio
    over its own process's ``off``, prompt by prompt), and whether it beat MTP on every prompt —
    the measured inputs for ``core_llm::AUTO_BLOCK_DRAFT_DECISIONS``. The ``"max"`` rows answer
    another question (the most a path can do) and are left out. A decision is the reader's, not
    this report's."""
    by_group: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = _group_key(entry)
        meta[key] = {"group": entry["group"], "context": entry["context"]}
        prompts = by_group.setdefault(key, {}).setdefault(entry["path"], {})
        bucket = prompts.setdefault(entry["prompt_id"], {})
        for metric, ratio in entry["speedup_vs_off"].items():
            bucket.setdefault(metric, [])
            bucket[metric].append(ratio)
    out = []
    for key, paths in sorted(by_group.items()):
        # What `auto` would choose between: MTP and each block draft at their recommended
        # depths, linear (auto never turns a tree on).
        mtp = paths.get(AUTO_MTP_PATH, {})
        for path, prompts in sorted(paths.items()):
            if not (path.startswith("block_draft:") and path.endswith(":linear@recommended")):
                continue
            item: dict[str, Any] = {**meta[key], "path": path}
            for metric in ("decode_tok_s", "wall_ms"):
                vs_off = {p: _mean(r.get(metric, [])) for p, r in prompts.items()}
                vs_mtp = {
                    p: ratio / _mean(mtp[p][metric])
                    for p, ratio in vs_off.items()
                    if ratio and p in mtp and _mean(mtp[p].get(metric, []))
                }
                item[metric] = {
                    "vs_off": _geomean([v for v in vs_off.values() if v]),
                    "vs_mtp": _geomean(list(vs_mtp.values())),
                    "vs_mtp_min": min(vs_mtp.values(), default=None),
                    "beats_mtp_on_every_prompt": bool(vs_mtp)
                    and len(vs_mtp) == len(vs_off)
                    and all(v > 1 for v in vs_mtp.values()),
                }
            out.append(item)
    return out


def paths_markdown(report: dict[str, Any]) -> str:
    """The report's summary as one table per (backend, tier, context)."""
    gib = 1024**3
    lines = [
        "# Speculative paths (SW-7200)",
        "",
        "Every ratio is against the `off` row of the same process. `draft ms/pass` is the "
        "proposer's host wall time per call: where the proposal comes back on the host it waited "
        "for the draft's device work and anything still queued ahead of it (the previous verify), "
        "so it is an upper bound on the draft's own cost; rows marked `enqueue-only` left their "
        "drafts on the device (Candle greedy MTP / draft model), so it is a lower bound. Peak "
        "GiB is the backend's reading (MLX allocator high-water; CUDA device-used at tokens, a "
        "lower bound) and `+request` what the request added over its start.",
        "",
    ]
    for item in report["refusals"]:
        lines.append(f"- refused: {item}")
    for item in report["incomplete"]:
        lines.append(f"- not judged: {item}")
    tables: dict[str, list[dict[str, Any]]] = {}
    for entry in report["summary"]:
        group = entry["group"]
        title = (
            f"{group['backend']} · {group['model']} · {group['tier']} · {entry['context']} context · "
            f"{group['max_new_tokens']} tokens · thinking {group['thinking']}"
        )
        tables.setdefault(title, []).append(entry)
    for title, rows in tables.items():
        lines += [
            "",
            f"## {title}",
            "",
            "| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request "
            "| TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            decode = row["speedup_vs_off"]["decode_tok_s"]
            wall = row["speedup_vs_off"]["wall_ms"]
            accepted = _fmt(row["accepted_per_pass"], 2)
            if row["accepted_per_tree_pass"] is not None:
                accepted += f" (tree {_fmt(row['accepted_per_tree_pass'], 2)})"
            share = row["draft_share_of_decode"]
            draft = _fmt(row["draft_ms_per_pass"], 2)
            if share is not None:
                draft += f" ({share:.0%})"
            if not row["draft_time_host_synchronized"]:
                draft += " enqueue-only"
            peak = row["peak_memory_bytes"]
            added = row["memory_added_bytes"]
            memory = "—" if peak is None else f"{peak / gib:.1f}"
            if added is not None:
                memory += f" (+{added / gib:.2f})"
            fallbacks = "; ".join(row["fallbacks"]) or (row["draft_refusal"] or "")
            lines.append(
                f"| {row['path']} | {','.join(row['ran'])} | {_fmt(row['decode_tok_s'])} | "
                f"{_fmt(decode['all'], 2)} ({_fmt(decode['predictable'], 2)} / "
                f"{_fmt(decode['open_ended'], 2)}) | {_fmt(row['wall_ms'], 0)} | "
                f"{_fmt(wall['all'], 2)} | {_fmt(row['ttft_ms'], 0)} | {accepted} | {draft} | "
                f"{memory} | {fallbacks} |"
            )
        if any(r.get("prefix_checkpoints") for r in rows):
            lines += [
                "",
                "### Prefix checkpoints",
                "",
                "| path | policy | reused | reread | capture ms | restore ms | captured bytes | resident bytes | protected bytes | matched kind |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for row in rows:
                pc = row.get("prefix_checkpoints")
                if not pc:
                    continue
                lines.append(
                    f"| {row['path']} | {','.join(pc.get('policies') or []) or '—'} | "
                    f"{_fmt(pc.get('reused_tokens'), 0)} | {_fmt(pc.get('reread_tokens'), 0)} | "
                    f"{_fmt(pc.get('capture_ms'), 2)} | {_fmt(pc.get('restore_ms'), 2)} | "
                    f"{_fmt(pc.get('captured_bytes'), 0)} | {_fmt(pc.get('resident_bytes'), 0)} | "
                    f"{_fmt(pc.get('protected_bytes'), 0)} | {','.join(pc.get('matched_kinds') or []) or '—'} |"
                )
        if any(r.get("adaptive_speculative") for r in rows):
            lines += [
                "",
                "### Adaptive speculative",
                "",
                "| path | adaptive | initial -> final | depth (init -> final) | depthΔ | switches | ms/tok | plain reason |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for row in rows:
                ad = row.get("adaptive_speculative")
                if not ad:
                    continue
                init_p = ",".join(ad.get("initial_proposers") or []) or "—"
                fin_p = ",".join(ad.get("final_proposers") or []) or "—"
                ms_tok = _fmt((ad.get("cost") or {}).get("ms_per_committed_token"), 2)
                plain_r = "; ".join(ad.get("plain_reasons") or []) or "—"
                lines.append(
                    f"| {row['path']} | {ad.get('adaptive')} | {init_p} -> {fin_p} | "
                    f"{_fmt(ad.get('initial_depth'), 1)} -> {_fmt(ad.get('final_depth'), 1)} | "
                    f"{_fmt(ad.get('depth_changes'), 1)} | {_fmt(ad.get('proposer_switches'), 1)} | "
                    f"{ms_tok} | {plain_r} |"
                )
    if report["auto_inputs"]:
        lines += [
            "",
            "## Block-draft inputs for `auto`",
            "",
            f"Each block draft at its recommended depth, linear, against `{AUTO_MTP_PATH}` of "
            "the same backend, tier and context.",
            "",
            "| backend | tier | context | path | decode ×off | decode ×MTP (min) | request ×MTP | "
            "beats MTP on every prompt (decode / request) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for item in report["auto_inputs"]:
            d, w = item["decode_tok_s"], item["wall_ms"]
            lines.append(
                f"| {item['group']['backend']} | {item['group']['tier']} | {item['context']} | "
                f"{item['path']} | {_fmt(d['vs_off'], 2)} | {_fmt(d['vs_mtp'], 2)} "
                f"({_fmt(d['vs_mtp_min'], 2)}) | {_fmt(w['vs_mtp'], 2)} | "
                f"{d['beats_mtp_on_every_prompt']} / {w['beats_mtp_on_every_prompt']} |"
            )
    return "\n".join(lines) + "\n"


def command_paths(args: argparse.Namespace) -> int:
    report = paths_report([Path(d) for d in args.campaign])
    with Path(args.json).open("x", encoding="utf-8") as sink:
        json.dump(report, sink, indent=2)
        sink.write("\n")
    with Path(args.markdown).open("x", encoding="utf-8") as sink:
        sink.write(paths_markdown(report))
    for line in report["refusals"]:
        print(f"::error::{line}", file=sys.stderr)
    return 1 if report["refusals"] or not report["entries"] else 0


def command_compare(args: argparse.Namespace) -> int:
    try:
        report = compare_campaigns(
            [Path(d) for d in args.campaign],
            Path(args.defaults).read_text(encoding="utf-8"),
            args.max_noise,
            Path(args.repo),
        )
    except Refused as refused:
        print("::error::refused to compare:", file=sys.stderr)
        print(str(refused), file=sys.stderr)
        return 1
    with Path(args.json).open("x", encoding="utf-8") as sink:
        json.dump(report, sink, indent=2)
        sink.write("\n")
    with Path(args.markdown).open("x", encoding="utf-8") as sink:
        sink.write(markdown_report(report))
    for e in report["e6"]:
        if e["verdict"] != "pass":
            print(
                f"E6 {e['verdict']}: {e['model']} {e['backend']} {e['class']} {e['option']}",
                file=sys.stderr,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help=f"validate ${MATRIX_ENV} and write the plan")
    plan.add_argument("--output", required=True, help="plan path (must not exist)")
    plan.add_argument("--lane", choices=sorted(LANES), default="cuda")

    def run_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--cargo", default="cargo")
        sub.add_argument(
            "--guard",
            help="a command prefix every benchmark process runs under (e.g. a memory guard that "
            "pauses, never kills), split like a POSIX shell; builds run unguarded",
        )
        sub.add_argument(
            "--allow-sha-override",
            action="store_true",
            help=f"build unstamped and pass {GIT_SHA_ENV} (the binaries record no compile-time "
            "SHA); for a host whose test binaries cannot run git",
        )

    run = commands.add_parser("run", help="build both checkouts and run every row of a plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--epic-root", required=True, help="checkout of the measured commit")
    run.add_argument("--epic-sha", required=True, help="the measured commit")
    run.add_argument("--epic-target-dir", help="CARGO_TARGET_DIR of the epic build")
    run.add_argument(
        "--pre-epic-root", help=f"checkout of {PRE_EPIC_SHA} (needed when a row runs the baseline)"
    )
    run.add_argument("--pre-epic-target-dir")
    run.add_argument("--output", required=True, help="run-scoped evidence directory")
    run.add_argument(
        "--stop-starting-after-minutes",
        type=float,
        help="start no benchmark process once this many minutes have passed since the builds "
        "finished; the running one finishes (a loaned host's budget)",
    )
    run_options(run)

    local = commands.add_parser("local", help="the whole lane on this host, worktrees included")
    local.add_argument("--lane", choices=sorted(LANES), default="mlx")
    local.add_argument("--matrix", required=True, help="matrix JSON file")
    local.add_argument("--repo", required=True, help="an inference clone holding both commits")
    local.add_argument("--epic-ref", required=True, help="the commit to measure")
    local.add_argument("--work-dir", required=True, help="worktrees and target directories")
    local.add_argument("--output", required=True, help="evidence directory (must not exist)")
    run_options(local)

    compare = commands.add_parser("compare", help="E6 verdicts and E5 decisions over campaigns")
    compare.add_argument("--campaign", action="append", required=True, help="campaign directory")
    compare.add_argument("--json", required=True, help="report path (must not exist)")
    compare.add_argument("--markdown", required=True, help="report path (must not exist)")
    compare.add_argument("--defaults", default=str(DEFAULTS_SOURCE), help="core-llm defaults.rs")
    compare.add_argument("--max-noise", type=float, default=DEFAULT_MAX_NOISE)
    compare.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="a clone holding every campaign's epic commit (orders re-measured rows)",
    )

    paths = commands.add_parser("paths", help="the paired speculative-paths report (SW-7200)")
    paths.add_argument("--campaign", action="append", required=True, help="campaign directory")
    paths.add_argument("--json", required=True, help="report path (must not exist)")
    paths.add_argument("--markdown", required=True, help="report path (must not exist)")

    args = parser.parse_args(argv)
    return {
        "plan": command_plan,
        "run": command_run,
        "local": command_local,
        "paths": command_paths,
        "compare": command_compare,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
