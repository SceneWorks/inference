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
  baseline-source  rewrite `decode_bench.rs` so it compiles against the pre-epic baseline

A **head** run requires a clean checkout at `--runtime-sha`. A **baseline** run
(`--baseline-of <head decode_bench.rs>`) requires the checkout's only change to be the untracked
`crates/llm/candle-llm/tests/decode_bench.rs`, byte-for-byte the `baseline-source` rewrite of the
named head file; both files' sha256 are recorded in `run.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
TEST_NAME = "decode_bench"
DEFAULT_LABEL = "RTX Pro 6000 / sm_120"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "release" / "real-weight-models.toml"
DEFAULT_MODEL_KEY = "bonsai-qwen38-parent"
# The manifest pins each model's revision; the bench additionally pins the exact `config.json` it
# was calibrated against, so a re-uploaded config under the same revision name is refused too.
PINNED_CONFIG_SHA256 = {
    "bonsai-qwen38-parent": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
}
BENCH_SOURCE_PATH = "crates/llm/candle-llm/tests/decode_bench.rs"
HEAD_ONLY_BEGIN = "// >>> head-only\n"
HEAD_ONLY_END = "// <<< head-only\n"
STUB_BEGIN = 'const BASELINE_STUB: &str = r#"\n'
STUB_END = '"#;\n'
METRIC_COLUMNS = (
    ("decode_tokens_per_second", "tok/s", "{:.2f}"),
    ("acceptance_rate", "acceptance", "{:.3f}"),
    ("target_forwards_per_generated_token", "fwd/tok", "{:.3f}"),
    ("host_syncs_per_token", "syncs/tok", "{:.2f}"),
    ("host_syncs_per_verify_step", "syncs/verify", "{:.2f}"),
    ("device_used_bytes_at_last_token", "device used @ last token", "gib"),
    ("cache_live_bytes", "cache live", "mib"),
    ("cache_checkpoint_bytes", "cache checkpoints", "mib"),
)
# Fields every run merged into one table must share, or the rows are not comparable.
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
    if path == "step_model":
        return "MTP off (" + ", ".join(["StepModel", *qualifiers]) + ")"
    return str(path)


def format_metric(value: Any, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "gib":
        return f"{value / 2**30:.2f} GiB"
    if fmt == "mib":
        return f"{value / 2**20:.1f} MiB"
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


def render_table(runs: list[dict[str, Any]]) -> str:
    """One Markdown table: a row per (run, decode row), the columns the story asks for.

    `match ref` compares a row with its own run's reference row; `match baseline ref` compares it
    with the **first** run's reference row — the cross-binary token identity that shows a head row
    reproduces the pre-change binary.
    """
    if not runs:
        raise ValueError("no runs to tabulate")
    check_comparable(runs)
    label = runs[0]["label"]
    model = runs[0]["model"]
    suite = runs[0]["suite"]
    baseline_ref = next(
        (row["tokens"] for row in suite["rows"] if row.get("path") == "reference"), None
    )
    if baseline_ref is None:
        raise ValueError(f"first run {runs[0]['run_name']} has no reference row to compare against")
    header = ["run", "row", "tokens", "match ref", "match baseline ref"] + [
        name for _, name, _ in METRIC_COLUMNS
    ]
    lines = [
        f"**{label}** — {model['repository']} @ {model['revision'][:12]} "
        f"(`{model['key']}`, config sha256 {model['config_sha256'][:12]}), "
        f"{suite.get('compute_dtype', 'unknown dtype')} greedy, {suite['prompt_tokens']} prompt "
        f"tokens, {suite['new_tokens']} new tokens per row.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for run in runs:
        for row in run["suite"]["rows"]:
            cross = divergence(baseline_ref, row.get("tokens", []))
            cells = [
                run["run_name"],
                row_label(row),
                str(row.get("generated_tokens")),
                match_text(row.get("tokens_match_reference"), row.get("first_divergence")),
                "yes" if cross is None else f"no @{cross}",
            ] + [format_metric(row.get(key), fmt) for key, _, fmt in METRIC_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"match baseline ref = tokens identical to `{runs[0]['run_name']}`'s reference row; "
        "device used @ last token = cuMemGetInfo total-free sampled at the row's last generated "
        "token while its cache is alive (device-wide, weights included); cache live / checkpoints "
        "= the StepModel row's final cache's own accounting (rollback checkpoints separately); "
        "syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where "
        "the binary predates the counter); syncs/verify = the speculative engine's transfers per "
        "verify step (n/a for non-speculative rows and where the binary predates the engine); "
        "fwd/tok = measured target forwards per generated token "
        "(n/a where the binary predates the counter)."
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
        if row.get("path") == "step_model" and doc.get("new_tokens", 0) >= 2:
            if not row.get("cache_checkpoint_bytes"):
                raise ValueError(
                    "step_model row reports no rollback-checkpoint bytes for a multi-token run"
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
            "DECODE_BENCH_NEW_TOKENS": str(args.new_tokens),
            "DECODE_BENCH_LABEL": args.label,
        }
    )
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
    return {**record, "suite": doc}


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


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = out.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="launch the decode_bench binary and seal its evidence")
    run_p.add_argument("--binary", required=True, help="built test executable containing decode_bench")
    run_p.add_argument("--run-name", required=True, help="e.g. baseline-d2b8cb335 or head-<sha>")
    run_p.add_argument("--runtime-sha", required=True, help="40-hex commit the binary was built from")
    run_p.add_argument("--checkout", help="checkout to verify --runtime-sha against (default: cwd)")
    run_p.add_argument("--snapshot", required=True, help="model snapshot directory (snapshots/<revision>)")
    run_p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="pinned real-weight model manifest")
    run_p.add_argument("--model-key", default=DEFAULT_MODEL_KEY, help="manifest key the snapshot must match")
    run_p.add_argument("--output", required=True, help="new evidence directory (must not exist)")
    run_p.add_argument("--rows", default="reference,step_model,mtp")
    run_p.add_argument("--drafts", default="1,2,3,4,5")
    run_p.add_argument("--new-tokens", type=int, default=256)
    run_p.add_argument("--prompt")
    run_p.add_argument("--label", default=DEFAULT_LABEL)
    run_p.add_argument("--gpu-index", type=int, default=0)
    run_p.add_argument("--sample-interval", type=float, default=0.25)
    run_p.add_argument(
        "--baseline-of",
        help="baseline run: the head decode_bench.rs whose baseline-source rewrite must be the "
        "checkout's only (untracked) change",
    )
    run_p.set_defaults(func=run)

    table_p = sub.add_parser("table", help="render a Markdown table from sealed run directories")
    table_p.add_argument("runs", nargs="+", help="sealed run directories; the first is the baseline")
    table_p.add_argument("--output")
    table_p.set_defaults(func=table)

    base_p = sub.add_parser("baseline-source", help="rewrite decode_bench.rs for the pre-epic baseline")
    base_p.add_argument("--input", required=True)
    base_p.add_argument("--output", required=True)
    base_p.set_defaults(func=baseline_source)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
