#!/usr/bin/env python3
"""Run and seal the `decode_bench` decode-perf suite (epic sc-24128, story sc-24129).

The suite itself is the `#[ignore]`d `decode_bench` test in
`crates/llm/candle-llm/tests/decode_bench.rs`. This wrapper reuses the sc-23942 native-comparison
plumbing (`qwen38_bonsai_terminal.py`): it launches an already-built test executable directly so
per-process RSS and GPU samples name the model process, records the source identity of the
checkout the binary was built from, pins the model snapshot against the release manifest,
snapshots the NVIDIA hardware state, and writes a write-once evidence directory holding the
suite's JSON, the sealed run record, and a Markdown table.

Subcommands:

  run              launch one binary against one pinned snapshot, sample memory, seal the evidence
  table            render a Markdown comparison table from one or more sealed runs
  campaign         run (or collect already-sealed runs of) the AT2 matrix — models x weight
                   formats x speculative modes x CUDA graphs — and seal an index with one table
                   per model plus the baseline rows (sc-24140)
  campaign-verify  re-check a sealed campaign index against its seal and every run's seal
  baseline-source  rewrite `decode_bench.rs` so it compiles against the pre-epic baseline

A **head** run requires a clean checkout at `--runtime-sha`. A **baseline** run
(`--baseline-of <head decode_bench.rs>`) requires the checkout's only change to be the untracked
`crates/llm/candle-llm/tests/decode_bench.rs`, byte-for-byte the `baseline-source` rewrite of the
named head file; both files' sha256 are recorded in `run.json`.

**Weight format is a row dimension** (sc-24140): a table may merge runs of different formats
(`bf16`, `q8`, `q4`, `nvfp4`). Every row is compared with the *baseline of its own format* — the
first `baseline` run of that format, or the table's first run when it has that format — and a
non-bf16 row with no same-format baseline (the pre-epic S1 baseline is bf16 only) is compared with
the **head's own bf16 reference row** (a bf16 run of the same source commit), labelled as such.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.release import qwen38_bonsai_terminal as terminal
    from scripts.release.verify_model_snapshot import load_model, verify_snapshot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qwen38_bonsai_terminal as terminal
    from verify_model_snapshot import load_model, verify_snapshot


SCHEMA_VERSION = 2
SUITE = "decode_bench"
CAMPAIGN_SUITE = "decode_bench_campaign"
CAMPAIGN_SCHEMA_VERSION = 1
TEST_NAME = "decode_bench"
DEFAULT_LABEL = "RTX Pro 6000 / sm_120"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "release" / "real-weight-models.toml"
DEFAULT_MODEL_KEY = "bonsai-qwen38-parent"
# The manifest pins each model's revision; the bench additionally pins the exact `config.json` it
# was calibrated against, so a re-uploaded config under the same revision name is refused too.
PINNED_CONFIG_SHA256 = {
    "bonsai-qwen38-parent": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    # sc-24138: the llama-family row set (a `CausalLm` snapshot) runs on Qwen3-8B.
    "qwen3-8b": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
}
BENCH_SOURCE_PATH = "crates/llm/candle-llm/tests/decode_bench.rs"
HEAD_ONLY_BEGIN = "// >>> head-only\n"
HEAD_ONLY_END = "// <<< head-only\n"
STUB_BEGIN = 'const BASELINE_STUB: &str = r#"\n'
STUB_END = '"#;\n'
# The projection weight formats the binary loads (sc-24140: `q8` / `q4` are the GGML load-time
# quantization both families' loaders already had; `nvfp4` is sc-24135 / sc-24140).
WEIGHT_FORMATS = ("bf16", "q8", "q4", "nvfp4")
GRAPH_SWITCHES = ("off", "on")
CUDA_GRAPHS_ENV = "CANDLE_LLM_CUDA_GRAPHS"
DEFAULT_SAMPLING = "0.7,0.9,0"
# The AT2 matrix's speculative dimension and the rows each mode runs, per model family. `None`
# is a cell the family cannot run — recorded as `n/a` in the campaign index, never as missing.
SPECULATIVE_MODES = ("off", "mtp", "ngram")
FAMILY_ROWS: dict[str, dict[str, list[str] | None]] = {
    "qwen35": {"off": ["reference", "step_model"], "mtp": ["mtp"], "ngram": ["ngram"]},
    "llama": {"off": ["reference", "step_model"], "mtp": None, "ngram": ["ngram"]},
}
NA_REASON = {("llama", "mtp"): "n/a (no MTP head)"}
SAMPLED_ROWS = ["sampled", "sampled_step_model"]
SAMPLED_PATHS = frozenset(SAMPLED_ROWS)
METRIC_COLUMNS = (
    ("decode_tokens_per_second", "tok/s", "{:.2f}"),
    ("acceptance_rate", "acceptance", "{:.3f}"),
    ("target_forwards_per_generated_token", "fwd/tok", "{:.3f}"),
    ("host_syncs_per_token", "syncs/tok", "{:.2f}"),
    ("host_syncs_per_verify_step", "syncs/verify", "{:.2f}"),
    ("target_forwards_per_verify_step", "fwd/verify", "{:.2f}"),
    ("replay_forwards", "replay forwards", "{}"),
    ("device_used_bytes_at_last_token", "device used @ last token", "gib"),
    ("cache_live_bytes", "cache live", "mib"),
    ("cache_checkpoint_bytes", "cache checkpoints", "mib"),
    ("fused_primitives", "fused primitives", "fused"),
    ("nvfp4_projections", "nvfp4 path", "nvfp4"),
    ("cuda_graphs", "cuda graphs", "graphs"),
    ("sampler", "sampler", "sampler"),
)
# Fields every run merged into one table must share, or the rows are not comparable. The weight
# format is *not* one of them since sc-24140: it is a labelled row dimension, and each row is
# compared with the baseline of its own format (see `comparison_basis`).
COMPARABLE_FIELDS = (
    ("model", "key"),
    ("model", "revision"),
    ("model", "config_sha256"),
    ("suite", "prompt_tokens"),
    ("suite", "new_tokens"),
)


def sha256_file(path: Path) -> str:
    return terminal.sha256(path)


def write_new(path: Path, value: Any) -> None:
    terminal.write_new(path, value)


def source_identity(expected_sha: str, allow_dirty: bool, cwd: Path | None) -> dict[str, Any]:
    """`qwen38_bonsai_terminal.source_identity` evaluated in the checkout the binary came from."""
    previous = Path.cwd()
    if cwd is not None:
        os.chdir(cwd)
    try:
        identity = terminal.source_identity(expected_sha, allow_dirty)
    finally:
        os.chdir(previous)
    identity["checkout"] = str(cwd if cwd is not None else previous)
    return identity


def pinned_model(manifest: Path, key: str, snapshot: Path) -> dict[str, Any]:
    """Refuse a snapshot that is not the manifest's pinned revision with the pinned config."""
    model = load_model(manifest, key)
    verify_snapshot(model, snapshot)  # revision directory / marker + expected files
    pinned_config = PINNED_CONFIG_SHA256.get(key)
    if pinned_config is None:
        raise ValueError(f"decode_bench has no pinned config sha256 for model {key!r}")
    config_sha = sha256_file(snapshot / "config.json")
    if config_sha != pinned_config:
        raise ValueError(
            f"snapshot config.json sha256 {config_sha} does not match the pinned {pinned_config} "
            f"for {key}"
        )
    return {
        "key": key,
        "repository": model["repository"],
        "revision": model["revision"],
        "config_sha256": config_sha,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }


def baseline_bench_source(checkout: Path, identity: dict[str, Any], head_source: Path) -> dict[str, Any]:
    """The baseline checkout may differ from its commit only by the rewritten bench file."""
    expected_dirty = [f"?? {BENCH_SOURCE_PATH}"]
    if identity["dirty_paths"] != expected_dirty:
        raise ValueError(
            f"baseline checkout must differ from its commit only by {expected_dirty}, "
            f"found {identity['dirty_paths']}"
        )
    bench = checkout / BENCH_SOURCE_PATH
    expected = baseline_source_text(head_source.read_text(encoding="utf-8")).encode("utf-8")
    if bench.read_bytes() != expected:
        raise ValueError(
            f"{bench} is not the baseline-source rewrite of {head_source}"
        )
    return {
        "path": BENCH_SOURCE_PATH,
        "sha256": sha256_file(bench),
        "head_source": str(head_source),
        "head_source_sha256": sha256_file(head_source),
    }


def row_qualifiers(row: dict[str, Any]) -> list[str]:
    """Which KV cache and which attention formulation produced the row (sc-24132), when the
    binary reported them; a pre-epic baseline binary reports neither."""
    qualifiers = []
    if row.get("kv_cache"):
        qualifiers.append(f"{row['kv_cache']} kv")
    if row.get("attn_formulation"):
        qualifiers.append(f"{row['attn_formulation']} attn")
    return qualifiers


def row_label(row: dict[str, Any]) -> str:
    path = row.get("path")
    qualifiers = row_qualifiers(row)
    if path == "mtp":
        base = f"MTP K={row.get('mtp_drafts')}"
        return f"{base} ({', '.join(qualifiers)})" if qualifiers else base
    if path == "ngram":
        base = f"n-gram K={row.get('drafts')}"
        return f"{base} ({', '.join(qualifiers)})" if qualifiers else base
    if path == "reference":
        return "MTP off (" + ", ".join(["reference", *qualifiers]) + ")"
    if path == "reference_unfused":
        return "MTP off (" + ", ".join(["reference", "fused off", *qualifiers]) + ")"
    if path == "reference_cublaslt":
        return "MTP off (" + ", ".join(["reference", "NVFP4 GEMV off", *qualifiers]) + ")"
    if path == "step_model":
        return "MTP off (" + ", ".join(["StepModel", *qualifiers]) + ")"
    if path == "sampled":
        return "sampled (" + ", ".join(["reference", *qualifiers]) + ")"
    if path == "sampled_step_model":
        return "sampled (" + ", ".join(["StepModel", *qualifiers]) + ")"
    return str(path)


def format_metric(value: Any, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "gib":
        return f"{value / 2**30:.2f} GiB"
    if fmt == "mib":
        return f"{value / 2**20:.1f} MiB"
    if fmt == "fused":
        # sc-24137: the row's fused-vs-reference primitive tally and the switch it ran under.
        reason = f" ({value['reference_reason']})" if value.get("reference_reason") else ""
        return f"{value['switch']}: {value['fused']} fused / {value['reference']} ref{reason}"
    if fmt == "nvfp4":
        # sc-24136: NVFP4 projection calls by path (fused decode GEMV vs cuBLASLt W4A4) and the
        # GEMV switch the row ran under; a bf16 run has no NVFP4 projections.
        if not value.get("gemv") and not value.get("cublaslt"):
            return "none"
        reason = f" ({value['cublaslt_reason']})" if value.get("cublaslt_reason") else ""
        return f"{value['switch']}: {value['gemv']} gemv / {value['cublaslt']} cuBLASLt{reason}"
    if fmt == "graphs":
        # sc-24134: the row's CUDA-graph tally, the switch it ran under and the fallback reason.
        reason = f" ({value['fallback_reason']})" if value.get("fallback_reason") else ""
        return (
            f"{value['switch']}: {value['replayed']} replayed / {value['eager']} eager, "
            f"{value['captured']} captured{reason}"
        )
    if fmt == "sampler":
        # sc-24140 (the S5 measurement): the row's sampler path, draws by side and the whole
        # logits rows copied to the host per generated token.
        per_token = value.get("logits_to_host_per_token")
        rows = "n/a" if per_token is None else f"{per_token:.2f}"
        return (
            f"{value['path']}: {value['device_draws']} device / {value['host_draws']} host, "
            f"{rows} logits rows->host/tok"
        )
    return fmt.format(value)


def match_text(match: bool | None, divergence: Any) -> str:
    if match is None:
        return "(ref)"
    return "yes" if match else f"no @{divergence}"


def divergence(reference: list[int], other: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(reference, other)):
        if a != b:
            return index
    if len(reference) != len(other):
        return min(len(reference), len(other))
    return None


def check_comparable(runs: list[dict[str, Any]]) -> None:
    labels = {run["label"] for run in runs}
    if len(labels) != 1:
        raise ValueError(f"runs carry different hardware labels: {sorted(labels)}")
    for section, field in COMPARABLE_FIELDS:
        values = {json.dumps(run.get(section, {}).get(field)) for run in runs}
        if len(values) != 1:
            raise ValueError(f"runs differ in {section}.{field}: {sorted(values)}")


def run_format(run: dict[str, Any]) -> str:
    """A run's projection weight format; a document older than sc-24136 is implicitly bf16."""
    return run.get("suite", {}).get("weight_format") or "bf16"


def run_graphs(run: dict[str, Any]) -> str:
    """The CUDA-graph switch a run's binary ran under (`n/a` for a binary without the runner)."""
    return run.get("suite", {}).get("cuda_graphs") or "n/a"


def row_kind(row: dict[str, Any]) -> str:
    """`sampled` for a stochastic row (sc-24140), `greedy` otherwise."""
    return "sampled" if row.get("path") in SAMPLED_PATHS or row.get("sampling") else "greedy"


BASIS_PATH = {"greedy": "reference", "sampled": "sampled"}


def basis_tokens(run: dict[str, Any], kind: str) -> list[int] | None:
    """The tokens of a run's reference row of `kind` (the greedy `reference` or the `sampled`)."""
    return next(
        (row["tokens"] for row in run["suite"]["rows"] if row.get("path") == BASIS_PATH[kind]),
        None,
    )


def same_source(a: dict[str, Any], b: dict[str, Any]) -> bool:
    sha = a.get("source", {}).get("head_sha")
    return sha is not None and sha == b.get("source", {}).get("head_sha")


def comparison_basis(
    runs: list[dict[str, Any]], run: dict[str, Any], kind: str
) -> tuple[str, list[int]] | None:
    """What a row of `run` is compared with in `match baseline ref` (sc-24140): `(label, tokens)`
    — an empty label for the baseline of the row's own format (the first `baseline` run of that
    format, else the table's first run when it has that format), else, for a non-bf16 row, the
    head's own bf16 reference row (a bf16 run built from the same source commit), labelled.
    `None` when neither exists."""
    fmt = run_format(run)
    candidates = [r for r in runs if r.get("run_kind") == "baseline" and run_format(r) == fmt]
    if run_format(runs[0]) == fmt:
        candidates.append(runs[0])
    for candidate in candidates:
        tokens = basis_tokens(candidate, kind)
        if tokens is not None:
            return "", tokens
    if fmt != "bf16":
        for candidate in runs:
            if run_format(candidate) == "bf16" and same_source(candidate, run):
                tokens = basis_tokens(candidate, kind)
                if tokens is not None:
                    return f"vs {candidate['run_name']} bf16 ref", tokens
    return None


def baseline_match_text(runs: list[dict[str, Any]], run: dict[str, Any], row: dict[str, Any]) -> str:
    basis = comparison_basis(runs, run, row_kind(row))
    if basis is None:
        return "n/a (no bf16 ref)" if run_format(run) != "bf16" else "n/a"
    label, tokens = basis
    cross = divergence(tokens, row.get("tokens", []))
    text = "yes" if cross is None else f"no @{cross}"
    return f"{text} ({label})" if label else text


def projection_format(suite: dict[str, Any]) -> str:
    """` with NVFP4 projections` for a quantized run (sc-24136); empty for the dense default."""
    fmt = suite.get("weight_format")
    return f" with {fmt.upper()} projections" if fmt and fmt != "bf16" else ""


def formats_heading(runs: list[dict[str, Any]]) -> str:
    """The table heading's projection clause: one run format as before, or every format the
    table holds (weight format being a row dimension since sc-24140)."""
    formats = sorted({run_format(run) for run in runs}, key=WEIGHT_FORMATS.index)
    if len(formats) == 1:
        return projection_format(runs[0]["suite"])
    return " with " + " / ".join(fmt.upper() for fmt in formats) + " projections (the format column)"


def sampling_heading(runs: list[dict[str, Any]]) -> str:
    sampled = next(
        (
            row["sampling"]
            for run in runs
            for row in run["suite"]["rows"]
            if row_kind(row) == "sampled" and row.get("sampling")
        ),
        None,
    )
    if sampled is None:
        return ""
    return (
        f" Sampled rows: temperature {sampled.get('temperature')}, top-p {sampled.get('top_p')}, "
        f"seed {sampled.get('seed')}; each compares with its run's `sampled` reference row."
    )


def render_table(runs: list[dict[str, Any]]) -> str:
    """One Markdown table: a row per (run, decode row), the columns the story asks for.

    `match ref` compares a row with its own run's reference row (a sampled row with its run's
    `sampled` row); `match baseline ref` compares it with the baseline of its own weight format
    (`comparison_basis`) — the cross-binary token identity that shows a head row reproduces the
    pre-change binary.
    """
    if not runs:
        raise ValueError("no runs to tabulate")
    check_comparable(runs)
    label = runs[0]["label"]
    model = runs[0]["model"]
    suite = runs[0]["suite"]
    if basis_tokens(runs[0], "greedy") is None:
        raise ValueError(f"first run {runs[0]['run_name']} has no reference row to compare against")
    header = ["run", "format", "graphs", "row", "tokens", "match ref", "match baseline ref"] + [
        name for _, name, _ in METRIC_COLUMNS
    ]
    lines = [
        f"**{label}** — {model['repository']} @ {model['revision'][:12]} "
        f"(`{model['key']}`, config sha256 {model['config_sha256'][:12]}), "
        f"{suite.get('compute_dtype', 'unknown dtype')}{formats_heading(runs)} greedy, "
        f"{suite['prompt_tokens']} prompt "
        f"tokens, {suite['new_tokens']} new tokens per row.{sampling_heading(runs)}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for run in runs:
        for row in run["suite"]["rows"]:
            cells = [
                run["run_name"],
                run_format(run),
                run_graphs(run),
                row_label(row),
                str(row.get("generated_tokens")),
                match_text(row.get("tokens_match_reference"), row.get("first_divergence")),
                baseline_match_text(runs, run, row),
            ] + [format_metric(row.get(key), fmt) for key, _, fmt in METRIC_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"format = the run's projection weight format; graphs = the CUDA-graph switch the binary "
        f"ran under (n/a where it predates the runner); match baseline ref = tokens identical to "
        f"the baseline reference row of the row's own format (`{runs[0]['run_name']}`'s when it "
        "has that format; a sampled row against the baseline's `sampled` row) — a non-bf16 row "
        "with no same-format baseline is compared with its head's own bf16 reference row, labelled "
        "`vs <run> bf16 ref`; "
        "device used @ last token = cuMemGetInfo total-free sampled at the row's last generated "
        "token while its cache is alive (device-wide, weights included); cache live / checkpoints "
        "= the StepModel row's final cache's own accounting (rollback checkpoints separately); "
        "syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where "
        "the binary predates the counter); syncs/verify = the speculative engine's transfers per "
        "verify step (n/a for non-speculative rows and where the binary predates the engine); "
        "fwd/tok = measured target forwards per generated token "
        "(n/a where the binary predates the counter); fwd/verify = measured target forwards net "
        "of the prefill per verify step (the verify forward plus any replay fallback or other "
        "extra forward; 1.00 on the per-token DeltaNet checkpoint "
        "ring, sc-24131) and replay forwards = verify steps the engine recovered by rolling back "
        "to the step start and replaying the kept prefix (0 on the ring; n/a for non-speculative "
        "rows and where the binary predates the counters); fused primitives = the switch the row ran "
        "under and how many RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel vs the "
        "op-chain reference, with the last reference reason (n/a where the binary predates the "
        "fused primitives); nvfp4 path = the NVFP4 decode-GEMV switch the row ran under and how many "
        "NVFP4 projection calls ran the fused GEMV vs the cuBLASLt W4A4 GEMM, with the last "
        "cuBLASLt reason (`rows` = a prefill; none = no NVFP4 projections; n/a where the binary "
        "predates the GEMV); "
        "cuda graphs = the CUDA-graph runner switch the row ran under, how "
        "many steps replayed a captured graph vs ran eager, how many graphs were captured, and "
        "the last fallback reason (n/a where the binary predates the runner); sampler = the "
        "row's sampler path (`device`, `host:<reason>`), its device / host draws and the whole "
        "logits rows copied to the host per generated token (n/a where the binary predates the "
        "counters)."
    )
    return "\n".join(lines) + "\n"


def validate_suite_document(doc: dict[str, Any]) -> None:
    if doc.get("suite") != SUITE or doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"binary did not write a decode_bench schema-{SCHEMA_VERSION} document")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("decode_bench document holds no rows")
    for row in rows:
        if row.get("generated_tokens") != doc.get("new_tokens"):
            raise ValueError(
                f"row {row_label(row)} generated {row.get('generated_tokens')} tokens, "
                f"expected {doc.get('new_tokens')}"
            )
        if not isinstance(row.get("decode_tokens_per_second"), (int, float)):
            raise ValueError(f"row {row_label(row)} has no decode throughput")
        # The Qwen3.5/3.6/3.8 hybrid's step cache always holds a DeltaNet rollback checkpoint
        # after a single-token step; a llama-family (`CausalLm`) cache rolls back by offset and
        # keeps none (sc-24138), so only the hybrid's zero is a bench that read a fresh cache.
        hybrid = doc.get("model_family", "qwen35") != "llama"
        step_row = row.get("path") in ("step_model", "sampled_step_model")
        if hybrid and step_row and doc.get("new_tokens", 0) >= 2:
            if not row.get("cache_checkpoint_bytes"):
                raise ValueError(
                    f"{row.get('path')} row reports no rollback-checkpoint bytes for a "
                    "multi-token run"
                )


def check_requested_dimensions(doc: dict[str, Any], weight_format: str, cuda_graphs: str | None) -> None:
    """Fail closed when the binary did not run the dimensions it was asked for (sc-24140)."""
    recorded = doc.get("weight_format") or "bf16"
    if recorded != weight_format:
        raise ValueError(
            f"binary recorded weight_format {recorded!r}, requested {weight_format!r}"
        )
    if cuda_graphs is not None and doc.get("cuda_graphs") != cuda_graphs:
        raise ValueError(
            f"binary recorded cuda_graphs {doc.get('cuda_graphs')!r}, requested {cuda_graphs!r}"
        )


def run(args: argparse.Namespace) -> int:
    # Every check runs before the evidence directory is created, so a refused run leaves nothing.
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    binary = Path(args.binary).resolve(strict=True)
    snapshot = terminal.lexical_absolute(Path(args.snapshot))
    manifest = Path(args.manifest).resolve(strict=True)
    model = pinned_model(manifest, args.model_key, snapshot)
    runtime_sha = terminal.checked_sha(args.runtime_sha, "runtime SHA")
    checkout = Path(args.checkout).resolve(strict=True) if args.checkout else Path.cwd()
    baseline = args.baseline_of is not None
    identity = source_identity(runtime_sha, baseline, checkout)
    bench_source = (
        baseline_bench_source(checkout, identity, Path(args.baseline_of).resolve(strict=True))
        if baseline
        else None
    )
    gpus, processes, hardware_reason = terminal.nvidia_hardware()
    selected, tenants = terminal.selected_gpu_state(gpus, processes, args.gpu_index)
    output.mkdir(parents=True, exist_ok=False)

    suite_path = output / "decode_bench.json"
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "DECODE_BENCH_SNAPSHOT": str(snapshot),
            "DECODE_BENCH_OUTPUT": str(suite_path),
            "DECODE_BENCH_ROWS": args.rows,
            "DECODE_BENCH_DRAFTS": args.drafts,
            "DECODE_BENCH_NGRAM_DRAFTS": args.ngram_drafts,
            "DECODE_BENCH_NEW_TOKENS": str(args.new_tokens),
            "DECODE_BENCH_LABEL": args.label,
            "DECODE_BENCH_FORMAT": args.format,
            "DECODE_BENCH_SAMPLING": args.sampling,
        }
    )
    cuda_graphs = getattr(args, "cuda_graphs", None)
    if cuda_graphs is not None:
        env[CUDA_GRAPHS_ENV] = "1" if cuda_graphs == "on" else "0"
    if args.prompt:
        env["DECODE_BENCH_PROMPT"] = args.prompt
    command = [str(binary), TEST_NAME, "--exact", "--ignored", "--nocapture", "--test-threads=1"]

    started_wall = time.time()
    started = time.monotonic()
    rss_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    gpu_reasons: set[str] = set()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        while proc.poll() is None:
            now = time.time()
            rss = terminal.rss_bytes(proc.pid)
            if rss is not None:
                rss_samples.append({"unix_seconds": now, "bytes": rss})
            used, reason = terminal.nvidia_sample(proc.pid)
            if used is not None:
                gpu_samples.append({"unix_seconds": now, "bytes": used})
            elif reason:
                gpu_reasons.add(reason)
            time.sleep(args.sample_interval)
    elapsed = time.monotonic() - started
    exit_code = proc.returncode
    if exit_code != 0:
        raise RuntimeError(f"decode_bench exited with {exit_code}; see {stderr_path}")
    if not suite_path.is_file():
        raise RuntimeError(f"decode_bench did not write {suite_path}")
    doc = json.loads(suite_path.read_text(encoding="utf-8"))
    validate_suite_document(doc)
    check_requested_dimensions(doc, args.format, cuda_graphs)
    if doc.get("label") != args.label:
        raise ValueError("binary recorded a different hardware label than requested")

    def peak(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {"available": False, "peak_bytes": None, "sample_count": 0}
        return {
            "available": True,
            "peak_bytes": max(sample["bytes"] for sample in samples),
            "first_sample_bytes": samples[0]["bytes"],
            "sample_count": len(samples),
        }

    record = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "run_name": args.run_name,
        "run_kind": "baseline" if baseline else "head",
        "label": args.label,
        "started_unix_seconds": started_wall,
        "elapsed_seconds": elapsed,
        "command": command,
        "requested": {
            "rows": args.rows,
            "drafts": args.drafts,
            "ngram_drafts": args.ngram_drafts,
            "weight_format": args.format,
            "cuda_graphs": cuda_graphs,
            "sampling": args.sampling,
        },
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "source": identity,
        "baseline_bench_source": bench_source,
        "model": model,
        "snapshot": {"path": str(snapshot), "config_sha256": model["config_sha256"]},
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "physical_memory": dict(
                zip(("total_bytes", "available_bytes", "unavailable_reason"), terminal.physical_memory())
            ),
        },
        "gpu": {
            "index": args.gpu_index,
            "selected": selected,
            "co_tenants_at_start": tenants,
            "hardware_unavailable_reason": hardware_reason,
        },
        "process_memory": {
            "rss": peak(rss_samples),
            "gpu": {**peak(gpu_samples), "unavailable_reasons": sorted(gpu_reasons)},
            "sample_interval_seconds": args.sample_interval,
            "scope": "sampled process working set / per-process GPU memory over the whole run",
        },
        "suite_document": "decode_bench.json",
        "suite_document_sha256": sha256_file(suite_path),
    }
    write_new(output / "run.json", record)
    table = render_table([{**record, "suite": doc}])
    (output / "decode_bench.md").write_text(table, encoding="utf-8", newline="\n")
    seal = {
        name: sha256_file(output / name)
        for name in ("decode_bench.json", "run.json", "decode_bench.md", "stdout.log", "stderr.log")
    }
    write_new(output / "SEAL.json", seal)
    print(table)
    return 0


def load_run(directory: Path) -> dict[str, Any]:
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    doc = json.loads((directory / "decode_bench.json").read_text(encoding="utf-8"))
    seal = json.loads((directory / "SEAL.json").read_text(encoding="utf-8"))
    for name, digest in seal.items():
        actual = sha256_file(directory / name)
        if actual != digest:
            raise ValueError(f"{directory / name} does not match its seal")
    if record.get("suite_document_sha256") != seal.get("decode_bench.json"):
        raise ValueError(f"{directory}: run record and seal disagree about the suite document")
    return {**record, "suite": doc, "directory": str(directory)}


def table(args: argparse.Namespace) -> int:
    runs = [load_run(Path(directory)) for directory in args.runs]
    text = render_table(runs)
    if args.output:
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


# ---- campaign (sc-24140): the AT2 matrix, run or collected, sealed behind one index -----------


def parse_choices(value: str, choices: tuple[str, ...], flag: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{flag} names no values")
    unknown = [item for item in items if item not in choices]
    if unknown:
        raise ValueError(f"{flag} has unknown values {unknown}; choose from {list(choices)}")
    if len(set(items)) != len(items):
        raise ValueError(f"{flag} repeats a value: {items}")
    return items


def snapshot_family(snapshot: Path) -> str:
    """`qwen35` for a Qwen3.5/3.6/3.8 hybrid snapshot, `llama` for a `CausalLm` one — the bench's
    own `is_causal_snapshot` dispatch, read from `config.json`."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    hay = " ".join(
        [str(config.get("model_type", ""))]
        + [str(a) for a in config.get("architectures", []) or []]
    ).lower()
    if "qwen3_vl" in hay or "qwen3vl" in hay:
        return "llama"
    return "qwen35" if ("qwen3_5" in hay or "qwen3_next" in hay) else "llama"


def run_family(run: dict[str, Any]) -> str:
    return "llama" if run["suite"].get("model_family") == "llama" else "qwen35"


def campaign_rows(family: str, speculative: list[str], sampled: bool) -> list[str]:
    rows: list[str] = []
    for mode in speculative:
        for row in FAMILY_ROWS[family][mode] or []:
            if row not in rows:
                rows.append(row)
    if sampled:
        rows += SAMPLED_ROWS
    return rows


def cell_status(run: dict[str, Any] | None, family: str, mode: str, drafts: list[int]) -> str:
    """One matrix cell's coverage: `ok`, `n/a (…)` for a mode the family cannot run, `missing`."""
    wanted = FAMILY_ROWS[family][mode]
    if wanted is None:
        return NA_REASON.get((family, mode), "n/a")
    if run is None:
        return "missing"
    paths = [row.get("path") for row in run["suite"]["rows"]]
    if any(path not in paths for path in wanted):
        return "missing"
    if mode == "mtp":
        have = {row.get("mtp_drafts") for row in run["suite"]["rows"] if row.get("path") == "mtp"}
        if any(k not in have for k in drafts):
            return "missing"
        return "ok (K=" + ",".join(str(k) for k in drafts) + ")"
    if mode == "ngram":
        ks = sorted(row.get("drafts") for row in run["suite"]["rows"] if row.get("path") == "ngram")
        return "ok (K=" + ",".join(str(k) for k in ks) + ")"
    return "ok"


def sampled_status(run: dict[str, Any] | None) -> str:
    if run is None:
        return "missing"
    paths = [row.get("path") for row in run["suite"]["rows"]]
    return "ok" if all(path in paths for path in SAMPLED_ROWS) else "not run"


def check_campaign_runs(heads: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> str | None:
    """Every head run in a campaign is clean and from one commit; every baseline is a baseline;
    one hardware label throughout. Returns the head runs' runtime SHA (`None` without heads)."""
    for run in baselines:
        if run.get("run_kind") != "baseline":
            raise ValueError(f"{run['run_name']} is not a baseline run")
    shas = set()
    for run in heads:
        if run.get("run_kind") != "head":
            raise ValueError(f"{run['run_name']} is a {run.get('run_kind')} run, not a head run")
        source = run.get("source", {})
        if not source.get("clean_tree"):
            raise ValueError(
                f"{run['run_name']} was built from a dirty tree ({source.get('dirty_paths')}); "
                "a campaign seals clean head runs only"
            )
        shas.add(source.get("head_sha"))
    if len(shas) > 1:
        raise ValueError(f"campaign head runs come from different commits: {sorted(map(str, shas))}")
    labels = {run["label"] for run in heads + baselines}
    if len(labels) > 1:
        raise ValueError(f"campaign runs carry different hardware labels: {sorted(labels)}")
    return shas.pop() if shas else None


def campaign(args: argparse.Namespace) -> int:
    """Run (or collect already-sealed runs of) the AT2 matrix and seal one index over it."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    formats = parse_choices(args.formats, WEIGHT_FORMATS, "--formats")
    graphs = parse_choices(args.graphs, GRAPH_SWITCHES, "--graphs")
    speculative = parse_choices(args.speculative, SPECULATIVE_MODES, "--speculative")
    drafts = [int(k) for k in args.drafts.split(",") if k.strip()]
    baselines = [load_run(Path(directory)) for directory in args.baseline]

    if args.collect:
        if args.model:
            raise ValueError("--collect takes the matrix from the sealed runs; drop --model")
        heads = [load_run(Path(directory)) for directory in args.collect]
        runtime_sha = check_campaign_runs(heads, baselines)
        if runtime_sha is None:
            raise ValueError("--collect names no head runs")
        models = []
        for head in heads:
            key = head["model"]["key"]
            if key not in [m["key"] for m in models]:
                models.append({"key": key, "family": run_family(head), "snapshot": None})
    else:
        if not args.model:
            raise ValueError("a campaign run needs at least one --model KEY=SNAPSHOT")
        for required in ("binary", "runtime_sha"):
            if not getattr(args, required):
                raise ValueError(f"a campaign run needs --{required.replace('_', '-')}")
        # Every refusal before anything runs or any directory exists: the binary, a clean
        # checkout at the runtime SHA, and every snapshot pinned.
        Path(args.binary).resolve(strict=True)
        runtime_sha = terminal.checked_sha(args.runtime_sha, "runtime SHA")
        checkout = Path(args.checkout).resolve(strict=True) if args.checkout else Path.cwd()
        source_identity(runtime_sha, False, checkout)
        # Each run re-checks that the checkout is clean, so the campaign's own evidence cannot be
        # written into it (the second run would find the first run's files).
        if checkout == output or checkout in output.parents:
            raise ValueError(
                f"campaign output {output} is inside the measured checkout {checkout}; write it "
                "elsewhere and copy the sealed directory in afterwards"
            )
        manifest = Path(args.manifest).resolve(strict=True)
        models = []
        for spec in args.model:
            key, sep, snapshot = spec.partition("=")
            if not sep or not key or not snapshot:
                raise ValueError(f"--model must be KEY=SNAPSHOT, got {spec!r}")
            if key in [m["key"] for m in models]:
                raise ValueError(f"--model {key} given twice")
            snapshot_path = terminal.lexical_absolute(Path(snapshot))
            pinned_model(manifest, key, snapshot_path)
            models.append({"key": key, "family": snapshot_family(snapshot_path), "snapshot": snapshot_path})
        check_campaign_runs([], baselines)  # baseline kind + one label among the baselines
        for baseline in baselines:
            if baseline["label"] != args.label:
                raise ValueError(
                    f"baseline {baseline['run_name']} carries label {baseline['label']!r}, the "
                    f"campaign runs under {args.label!r}"
                )
        output.mkdir(parents=True)
        heads = []
        for model in models:
            rows = campaign_rows(model["family"], speculative, args.sampled)
            if not rows:
                raise ValueError(f"no rows to run for {model['key']} ({model['family']})")
            for fmt in formats:
                for switch in graphs:
                    name = f"{model['key']}-{fmt}-graphs-{switch}"
                    run(
                        argparse.Namespace(
                            binary=args.binary,
                            run_name=f"{args.run_prefix}{name}",
                            runtime_sha=runtime_sha,
                            checkout=args.checkout,
                            snapshot=str(model["snapshot"]),
                            manifest=args.manifest,
                            model_key=model["key"],
                            output=str(output / "runs" / name),
                            rows=",".join(rows),
                            drafts=args.drafts,
                            ngram_drafts=args.ngram_drafts,
                            new_tokens=args.new_tokens,
                            prompt=args.prompt,
                            label=args.label,
                            format=fmt,
                            cuda_graphs=switch,
                            sampling=args.sampling,
                            gpu_index=args.gpu_index,
                            sample_interval=args.sample_interval,
                            baseline_of=None,
                        )
                    )
                    heads.append(load_run(output / "runs" / name))
        check_campaign_runs(heads, baselines)

    # The coverage matrix: every requested (model, format, graphs) cell, per speculative mode.
    cells = []
    missing = []
    for model in models:
        for fmt in formats:
            for switch in graphs:
                matches = [
                    head
                    for head in heads
                    if head["model"]["key"] == model["key"]
                    and run_format(head) == fmt
                    and run_graphs(head) == switch
                ]
                if len(matches) > 1:
                    raise ValueError(
                        f"more than one run for {model['key']} / {fmt} / graphs {switch}: "
                        f"{[r['run_name'] for r in matches]}"
                    )
                cell_run = matches[0] if matches else None
                status = {
                    mode: cell_status(cell_run, model["family"], mode, drafts) for mode in speculative
                }
                if args.sampled:
                    status["sampled"] = sampled_status(cell_run)
                for mode, text in status.items():
                    if text in ("missing", "not run"):
                        missing.append(f"{model['key']} / {fmt} / graphs {switch} / {mode}")
                cells.append(
                    {
                        "model": model["key"],
                        "family": model["family"],
                        "weight_format": fmt,
                        "cuda_graphs": switch,
                        "run": cell_run["run_name"] if cell_run else None,
                        "run_directory": None,  # filled once the run sits in the campaign
                        "run_seal_sha256": None,
                        "speculative": status,
                    }
                )
    stray = [h["run_name"] for h in heads if not any(c["run"] == h["run_name"] for c in cells)]
    if stray:
        raise ValueError(f"runs outside the requested matrix: {stray}")
    if missing and not args.allow_partial:
        raise ValueError("campaign matrix cells are missing: " + "; ".join(missing))

    # The campaign is self-contained: every baseline (and, collecting, every head run) is copied
    # in beside the runs it ran, re-verified from the copy, and the index names it relative to
    # the campaign directory — so the sealed directory can be moved (into the repository's
    # evidence tree) and still verify.
    output.mkdir(parents=True, exist_ok=True)
    copies = [("baselines", run) for run in baselines]
    if args.collect:
        copies += [("runs", run) for run in heads]
    for kind, entry in copies:
        source = Path(entry["directory"])
        destination = output / kind / source.name
        if destination.exists():
            raise ValueError(f"two sealed runs share the directory name {source.name!r}")
        shutil.copytree(source, destination)
        copied = load_run(destination)
        if copied["suite_document_sha256"] != entry["suite_document_sha256"]:
            raise ValueError(f"{destination} is not a faithful copy of {source}")
        entry["directory"] = str(destination)
    by_name = {run["run_name"]: run for run in heads}
    for cell in cells:
        if cell["run"]:
            directory = Path(by_name[cell["run"]]["directory"])
            cell["run_directory"] = directory.relative_to(output).as_posix()
            cell["run_seal_sha256"] = sha256_file(directory / "SEAL.json")

    # The index: the coverage table, then one comparison table per model (baselines first).
    label = (heads or baselines)[0]["label"] if (heads or baselines) else args.label
    modes = list(speculative) + (["sampled"] if args.sampled else [])
    mode_names = {"off": "speculative off", "mtp": "MTP", "ngram": "n-gram", "sampled": "sampled"}
    lines = [
        f"# decode-perf campaign — {label}",
        "",
        f"Runtime `{runtime_sha}` (clean tree); matrix: models {[m['key'] for m in models]} x "
        f"formats {formats} x speculative {speculative} x CUDA graphs {graphs}"
        + (" + the stochastic rows" if args.sampled else "")
        + ". `n/a` = a cell the family cannot run (the llama family has no MTP head).",
        "",
        "## Coverage",
        "",
        "| model | format | graphs | " + " | ".join(mode_names[m] for m in modes) + " | run |",
        "|" + "|".join("---" for _ in range(4 + len(modes))) + "|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['model']} | {cell['weight_format']} | {cell['cuda_graphs']} | "
            + " | ".join(cell["speculative"][m] for m in modes)
            + f" | {cell['run'] or 'missing'} |"
        )
    tables = {}
    for model in models:
        model_runs = [r for r in baselines if r["model"]["key"] == model["key"]] + [
            r for r in heads if r["model"]["key"] == model["key"]
        ]
        if not model_runs:
            continue
        text = render_table(model_runs)
        tables[model["key"]] = [r["run_name"] for r in model_runs]
        lines += ["", f"## {model['key']}", "", text.rstrip("\n")]
    index_md = output / "INDEX.md"
    index_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    index = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "suite": CAMPAIGN_SUITE,
        "label": label,
        "runtime_sha": runtime_sha,
        "mode": "collect" if args.collect else "run",
        "matrix": {
            "models": [m["key"] for m in models],
            "weight_formats": formats,
            "speculative": speculative,
            "mtp_drafts": drafts,
            "cuda_graphs": graphs,
            "sampled": args.sampled,
        },
        "cells": cells,
        "missing": missing,
        "baselines": [
            {
                "run": r["run_name"],
                "run_directory": Path(r["directory"]).relative_to(output).as_posix(),
                "run_seal_sha256": sha256_file(Path(r["directory"]) / "SEAL.json"),
            }
            for r in baselines
        ],
        "tables": tables,
    }
    write_new(output / "index.json", index)
    write_new(
        output / "SEAL.json",
        {name: sha256_file(output / name) for name in ("INDEX.md", "index.json")},
    )
    print((output / "INDEX.md").read_text(encoding="utf-8"))
    return 0


def verify_campaign(directory: Path) -> dict[str, Any]:
    """Re-check a sealed campaign: the index files against the campaign seal, every run and
    baseline against its own seal and the seal hash the index recorded."""
    seal = json.loads((directory / "SEAL.json").read_text(encoding="utf-8"))
    if sorted(seal) != ["INDEX.md", "index.json"]:
        raise ValueError(f"{directory}: the campaign seal must cover INDEX.md and index.json")
    for name, digest in seal.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"{directory / name} does not match the campaign seal")
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index.get("suite") != CAMPAIGN_SUITE:
        raise ValueError(f"{directory} is not a decode_bench campaign")
    entries = [c for c in index["cells"] if c.get("run_directory")] + index["baselines"]
    for entry in entries:
        relative = Path(entry["run_directory"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{directory}: {relative} is outside the campaign directory")
        run_dir = directory / relative
        load_run(run_dir)
        if sha256_file(run_dir / "SEAL.json") != entry["run_seal_sha256"]:
            raise ValueError(f"{run_dir} was re-sealed after the campaign index was written")
    return index


def campaign_verify(args: argparse.Namespace) -> int:
    index = verify_campaign(Path(args.directory))
    print(f"campaign {args.directory}: {len(index['cells'])} cells, seals verified")
    return 0


def baseline_source_text(source: str) -> str:
    """Replace the `head-only` block with the file's own `BASELINE_STUB` body."""
    begin = source.index(HEAD_ONLY_BEGIN)
    end = source.index(HEAD_ONLY_END, begin) + len(HEAD_ONLY_END)
    stub_begin = source.index(STUB_BEGIN) + len(STUB_BEGIN)
    stub_end = source.index(STUB_END, stub_begin)
    stub = source[stub_begin:stub_end]
    rewritten = source[:begin] + "// baseline stub (rewritten by decode_bench.py baseline-source)\n" + stub.lstrip("\n") + source[end:]
    if HEAD_ONLY_BEGIN in rewritten:
        raise ValueError("more than one head-only block")
    return rewritten


def baseline_source(args: argparse.Namespace) -> int:
    source = Path(args.input).read_text(encoding="utf-8")
    rewritten = baseline_source_text(source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rewritten, encoding="utf-8", newline="\n")
    return 0


def add_measurement_args(p: argparse.ArgumentParser) -> None:
    """The flags `run` and `campaign` share."""
    p.add_argument("--runtime-sha", help="40-hex commit the binary was built from")
    p.add_argument("--checkout", help="checkout to verify --runtime-sha against (default: cwd)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="pinned real-weight model manifest")
    p.add_argument("--drafts", default="1,2,3,4,5", help="MTP draft widths")
    p.add_argument("--ngram-drafts", default="3", help="n-gram draft widths")
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--prompt")
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument(
        "--sampling",
        default=DEFAULT_SAMPLING,
        help="the stochastic rows' temperature,top_p,seed (sc-24140)",
    )
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--sample-interval", type=float, default=0.25)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = out.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="launch the decode_bench binary and seal its evidence")
    run_p.add_argument("--binary", required=True, help="built test executable containing decode_bench")
    run_p.add_argument("--run-name", required=True, help="e.g. baseline-d2b8cb335 or head-<sha>")
    run_p.add_argument("--snapshot", required=True, help="model snapshot directory (snapshots/<revision>)")
    run_p.add_argument("--model-key", default=DEFAULT_MODEL_KEY, help="manifest key the snapshot must match")
    run_p.add_argument("--output", required=True, help="new evidence directory (must not exist)")
    run_p.add_argument("--rows", default="reference,step_model,mtp")
    add_measurement_args(run_p)
    run_p.add_argument(
        "--format",
        default="bf16",
        choices=WEIGHT_FORMATS,
        help="projection weight format the binary loads, quantized at load: bf16 (dense), q8 / q4 "
        "(GGML), nvfp4 (sc-24135 / sc-24140)",
    )
    run_p.add_argument(
        "--cuda-graphs",
        choices=GRAPH_SWITCHES,
        help=f"force the CUDA-graph switch ({CUDA_GRAPHS_ENV}) for the binary and require the "
        "document to record it (default: inherit the environment)",
    )
    run_p.add_argument(
        "--baseline-of",
        help="baseline run: the head decode_bench.rs whose baseline-source rewrite must be the "
        "checkout's only (untracked) change",
    )
    run_p.set_defaults(func=run_checked)

    table_p = sub.add_parser("table", help="render a Markdown table from sealed run directories")
    table_p.add_argument("runs", nargs="+", help="sealed run directories; the first is the baseline")
    table_p.add_argument("--output")
    table_p.set_defaults(func=table)

    camp_p = sub.add_parser(
        "campaign",
        help="run (or collect sealed runs of) the AT2 matrix and seal one index over it",
    )
    camp_p.add_argument("--output", required=True, help="new campaign directory (must not exist)")
    camp_p.add_argument(
        "--model",
        action="append",
        default=[],
        help="KEY=SNAPSHOT, one per model (run mode); e.g. qwen3-8b=E:\\...\\snapshots\\<rev>",
    )
    camp_p.add_argument("--binary", help="built test executable containing decode_bench (run mode)")
    camp_p.add_argument(
        "--collect",
        action="append",
        default=[],
        help="an already-sealed head run directory to slot into the matrix (collect mode; repeat)",
    )
    camp_p.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="a sealed baseline run directory, tabulated first for its model (repeat)",
    )
    camp_p.add_argument("--formats", default="bf16,q8,nvfp4")
    camp_p.add_argument("--graphs", default="off,on")
    camp_p.add_argument("--speculative", default="off,mtp,ngram")
    camp_p.add_argument(
        "--no-sampled",
        dest="sampled",
        action="store_false",
        help="leave out the stochastic rows (sampled, sampled_step_model)",
    )
    camp_p.add_argument(
        "--allow-partial",
        action="store_true",
        help="seal an index whose matrix has missing cells (listed as missing)",
    )
    camp_p.add_argument("--run-prefix", default="head-", help="prefix of every run name")
    add_measurement_args(camp_p)
    camp_p.set_defaults(func=campaign)

    verify_p = sub.add_parser("campaign-verify", help="re-check a sealed campaign directory")
    verify_p.add_argument("directory")
    verify_p.set_defaults(func=campaign_verify)

    base_p = sub.add_parser("baseline-source", help="rewrite decode_bench.rs for the pre-epic baseline")
    base_p.add_argument("--input", required=True)
    base_p.add_argument("--output", required=True)
    base_p.set_defaults(func=baseline_source)
    return out


def run_checked(args: argparse.Namespace) -> int:
    if not args.runtime_sha:
        raise ValueError("run needs --runtime-sha")
    return run(args)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
