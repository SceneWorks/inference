#!/usr/bin/env python3
"""Select immutable same-run Qwen artifacts and assemble the existing matrix seal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

try:
    from scripts.release import qwen38_bonsai_terminal as terminal
    from scripts.release import qwen38_bonsai_assets as assets
except ImportError:
    import qwen38_bonsai_terminal as terminal
    import qwen38_bonsai_assets as assets


CUDA_CELLS = (
    "candle-cuda-qwen38-parent", "candle-cuda-bonsai-gguf",
    "candle-cuda-qwen3vl-baseline", "functional-candle-bonsai-mlx",
    "functional-candle-pq2-bf16", "functional-candle-pq2-q8",
    "functional-candle-ptq1-bf16", "functional-candle-ptq1-q8",
)
ROLES = ("mlx", "cuda")


def qualified_publisher(root: Path, runtime_sha: str) -> tuple[dict, dict]:
    metadata = json.loads((root / "snapshot-metadata.json").read_text(encoding="utf-8"))
    provision = json.loads((root / "provision-report.json").read_text(encoding="utf-8"))
    closure = terminal.sha256(assets.DEFAULT_CLOSURE)
    if (
        metadata.get("all_metadata_qualified") is not True
        or metadata.get("publisher_verified_all") is not True
        or metadata.get("runtime_sha") != runtime_sha
        or metadata.get("publisher_closure_sha256") != closure
        or provision.get("complete") is not True
        or provision.get("runtime_sha") != runtime_sha
        or provision.get("publisher_closure_sha256") != closure
        or provision.get("model_execution_performed") is not False
    ):
        raise ValueError("publisher evidence is incomplete or belongs to another source")
    return metadata, provision


def stage_cuda(args: argparse.Namespace) -> int:
    source, output = args.source.resolve(), args.output.resolve()
    qualified_publisher(source, args.runtime_sha)
    if output.exists():
        raise ValueError("CUDA evidence output already exists")
    header_files = ("hardware-before.json", "snapshot-metadata.json", "provision-report.json", "gpu-reservation.json")
    for name in header_files:
        path = source / name
        if not path.is_file():
            raise ValueError(f"CUDA evidence lacks {name}")
    output.mkdir(parents=True)
    for name in header_files:
        shutil.copy2(source / name, output / name)
    for cell in CUDA_CELLS:
        preflight = source / f"{cell}-preflight.json"
        if preflight.is_file():
            shutil.copy2(preflight, output / preflight.name)
        row = source / cell
        if row.is_dir():
            shutil.copytree(row, output / cell)
    return 0


def list_run_artifacts(api_url: str, repository: str, run_id: str, token: str) -> list[dict]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GitHub repository")
    if not re.fullmatch(r"[0-9]+", run_id):
        raise ValueError("invalid GitHub run ID")
    result = []
    for page in range(1, 11):
        url = f"{api_url.rstrip('/')}/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        request = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.load(response)
        batch = body.get("artifacts")
        if not isinstance(batch, list):
            raise ValueError("GitHub artifact listing is malformed")
        result.extend(batch)
        if len(batch) < 100:
            return result
    raise ValueError("GitHub artifact listing exceeded bounded pagination")


def choose_artifacts(items: list[dict], *, roles: tuple[str, ...], runtime_sha: str, run_id: str) -> list[dict]:
    selected = []
    for role in roles:
        prefix = f"qwen38-bonsai-{role}-{runtime_sha}-{run_id}-"
        candidates = []
        for item in items:
            name = item.get("name")
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            attempt = name[len(prefix):]
            if not re.fullmatch(r"[1-9][0-9]*", attempt):
                raise ValueError(f"malformed attempt-scoped artifact name: {name}")
            source = item.get("workflow_run") or {}
            if (
                item.get("expired") is not False
                or type(item.get("id")) is not int
                or item["id"] <= 0
                or source.get("id") != int(run_id)
                or source.get("head_sha") != runtime_sha
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", item.get("digest") or "")
            ):
                raise ValueError(f"artifact lacks immutable current-run source identity: {name}")
            candidates.append((int(attempt), item))
        if not candidates:
            raise ValueError(f"missing current-run artifact for {role}")
        top_attempt = max(attempt for attempt, _ in candidates)
        highest = [item for attempt, item in candidates if attempt == top_attempt]
        if len(highest) != 1:
            raise ValueError(f"duplicate artifact for {role} attempt {top_attempt}")
        item = highest[0]
        selected.append({
            "role": role, "name": item["name"], "artifact_id": item["id"],
            "archive_digest": item["digest"], "run_attempt": top_attempt,
        })
    if len({entry["artifact_id"] for entry in selected}) != len(selected):
        raise ValueError("selected artifact IDs are duplicated")
    return selected


def select(args: argparse.Namespace) -> int:
    runtime_sha = terminal.checked_sha(args.runtime_sha, "runtime SHA")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is required to select immutable artifact IDs")
    items = list_run_artifacts(args.api_url, args.repository, args.run_id, token)
    chosen = choose_artifacts(items, roles=ROLES, runtime_sha=runtime_sha, run_id=args.run_id)
    terminal.write_new(args.output, {
        "schema_version": 1, "runtime_sha": runtime_sha, "run_id": args.run_id,
        "selected": chosen,
    })
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write("artifact_ids=" + ",".join(str(entry["artifact_id"]) for entry in chosen) + "\n")
    return 0


def check_partition(args: argparse.Namespace) -> int:
    spec = json.loads(args.matrix.read_text(encoding="utf-8"))
    terminal.validate_full_acceptance_contract(spec)
    qualified_publisher(args.root, args.runtime_sha)
    hardware = terminal.validate_hardware_record(args.root / "hardware-before.json")
    metadata = json.loads((args.root / "snapshot-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("hostname") != hardware.get("host", {}).get("hostname"):
        raise ValueError("row hardware and publisher verification used different hosts")
    if args.role == "mlx":
        ids = tuple(item["id"] for item in spec["cells"] if item["backend"] == "mlx")
    elif args.role == "cuda":
        ids = CUDA_CELLS
    else:
        raise ValueError("unknown accelerator campaign partition")
    cells = {item["id"]: item for item in spec["cells"]}
    for cell_id in ids:
        cell = cells[cell_id]
        model = terminal.load_model(args.manifest, cell["model_key"])
        preflight_path = args.root / f"{cell_id}-preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        terminal.validate_preflight_record(
            preflight, model=model, load_profile=cell["load_profile"],
            language_variant=cell.get("language_variant"),
            vision_variant=cell.get("vision_variant"),
        )
        if preflight.get("admitted") is not True:
            raise ValueError(f"{cell_id} was not admitted")
        receipt, provider = terminal.validate_receipt(
            args.root / cell_id, expected_backend=cell["backend"], expected_device=cell["device"]
        )
        receipt_model = receipt.get("model", {})
        expected_cases = spec["groups"][cell["group"]]["case_ids"] + cell.get("acceptance_case_ids", [])
        if (
            receipt.get("runtime", {}).get("head_sha") != args.runtime_sha
            or provider.get("runtime_sha") != args.runtime_sha
            or receipt_model.get("id") != cell_id
            or receipt_model.get("manifest_key") != cell["model_key"]
            or receipt_model.get("revision") != model["revision"]
            or receipt_model.get("language_variant") != cell.get("language_variant")
            or receipt_model.get("vision_variant") != cell.get("vision_variant")
            or receipt.get("command", {}).get("load_profile") != cell["load_profile"]
            or receipt.get("command", {}).get("candle_device") != terminal.LOAD_PROFILES[cell["load_profile"]]["command_device"]
            or receipt.get("command", {}).get("preflight_sha256") != terminal.sha256(preflight_path)
            or [row.get("case_id") for row in provider["cases"]] != expected_cases
        ):
            raise ValueError(f"{cell_id} source, model, preflight, or full case list differs from matrix")
        terminal.validate_artifact_size_evidence(
            receipt_model,
            terminal.pinned_admission_sizes(
                model, cell.get("language_variant"), cell.get("vision_variant")),
        )
        if cell["device"] == "cuda":
            reservation = json.loads((args.root / "gpu-reservation.json").read_text(encoding="utf-8"))
            recheck = receipt.get("gpu", {}).get("admission_recheck") or {}
            if (
                reservation.get("gpu_index") != terminal.CUDA_DEVICE_INDEX
                or reservation.get("gpu_uuid") != preflight.get("selected_gpu_uuid")
                or reservation.get("token_sha256") != preflight.get("reservation_token_sha256")
                or recheck.get("admitted") is not True
                or recheck.get("gpu_index") != terminal.CUDA_DEVICE_INDEX
                or recheck.get("selected_gpu_uuid") != preflight.get("selected_gpu_uuid")
                or recheck.get("selected_gpu_compute_processes") != []
            ):
                raise ValueError(f"{cell_id} reservation or selected-device recheck differs from preflight")
        elif receipt.get("gpu", {}).get("admission_recheck") is not None:
            raise ValueError(f"{cell_id} unexpectedly has a CUDA admission recheck")
        accepted = set(cell.get("acceptance_case_ids", [])) if cell.get("functional_acceptance") is True else set()
        if spec["groups"][cell["group"]].get("functional_acceptance") is True:
            accepted.update(spec["groups"][cell["group"]]["case_ids"])
        if cell["group"] != "matched" and any(case.get("status") == "resource_declined" for case in provider["cases"]):
            raise ValueError(f"{cell_id} format-functional case resource declined")
        if any(case.get("functional_acceptance_passed") is not True for case in provider["cases"] if case["case_id"] in accepted):
            raise ValueError(f"{cell_id} functional acceptance failed")
    return 0


def aggregate(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("runtime_sha") != args.runtime_sha or selection.get("run_id") != args.run_id:
        raise ValueError("artifact selection does not match aggregate source/run")
    entries = selection.get("selected")
    if not isinstance(entries, list) or [entry.get("role") for entry in entries] != list(ROLES):
        raise ValueError("artifact selection omits a required partition")
    roots = []
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unsafe selected artifact name")
        root = args.downloads / name
        if not root.is_dir():
            raise ValueError(f"selected artifact ID {entry.get('artifact_id')} was not downloaded")
        roots.append(root)
    args.output.mkdir(parents=True, exist_ok=False)
    sealed_selection = args.output / "selected-artifacts.json"
    shutil.copy2(args.selection, sealed_selection)
    common = dict(root=roots, matrix=args.matrix, manifest=args.manifest,
                  runtime_sha=args.runtime_sha, output=args.output / "matrix-report.json",
                  markdown=args.output / "matrix-report.md", seal=args.output / "matrix-seal.json",
                  artifact_selection=sealed_selection)
    status = terminal.matrix_status(argparse.Namespace(**common))
    if status != 0:
        raise ValueError("full 16-cell accelerator matrix did not pass; inspect aggregate report")
    terminal.verify_matrix_seal(argparse.Namespace(**common))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-cuda")
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--runtime-sha", required=True)
    stage.set_defaults(func=stage_cuda)
    selector = sub.add_parser("select")
    selector.add_argument("--role", choices=("matrix",), required=True)
    selector.add_argument("--runtime-sha", required=True)
    selector.add_argument("--run-id", required=True)
    selector.add_argument("--repository", required=True)
    selector.add_argument("--api-url", required=True)
    selector.add_argument("--output", type=Path, required=True)
    selector.add_argument("--github-output", type=Path)
    selector.set_defaults(func=select)
    check = sub.add_parser("check-partition")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--role", choices=("mlx", "cuda"), required=True)
    check.add_argument("--cell")
    check.add_argument("--runtime-sha", required=True)
    check.add_argument("--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json"))
    check.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    check.set_defaults(func=check_partition)
    fanin = sub.add_parser("aggregate")
    fanin.add_argument("--downloads", type=Path, required=True)
    fanin.add_argument("--selection", type=Path, required=True)
    fanin.add_argument("--output", type=Path, required=True)
    fanin.add_argument("--runtime-sha", required=True)
    fanin.add_argument("--run-id", required=True)
    fanin.add_argument("--matrix", type=Path, default=Path("release/qwen38-bonsai-matrix.json"))
    fanin.add_argument("--manifest", type=Path, default=Path("release/real-weight-models.toml"))
    fanin.set_defaults(func=aggregate)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"qwen38-bonsai-artifacts: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
