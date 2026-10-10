#!/usr/bin/env python3
"""Verify the authorized RC7 accelerator evidence reuse without relabeling receipts."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile

try:
    from scripts.release import qwen38_bonsai_artifacts as artifacts
    from scripts.release import qwen38_bonsai_terminal as terminal
except ImportError:
    import qwen38_bonsai_artifacts as artifacts
    import qwen38_bonsai_terminal as terminal


ROOT = Path(__file__).resolve().parents[2]
OBSERVED_SHA = "182d30f782824ee6f54f7fd60bbfda76099d2b57"
SCOPED_SHA = "60c83b9e6e63693fc53a2ffd90031a9c74b32f6e"
RUN_ID = 35644054098
REPOSITORY = "SceneWorks/inference"
ARTIFACTS = {
    "mlx": {
        "id": 10660536559,
        "attempt": 1,
        "digest": "sha256:176736900115b71571232458722d25151197e5e3eb1239841fbda881d8ab097b",
    },
    "cuda": {
        "id": 10669544635,
        "attempt": 2,
        "digest": "sha256:1774a5248017e1cc5723082a11fb70bb40356663dc7a7d250a8d453fa976fed9",
    },
}
SCOPED_CHANGED_PATHS = {
    ".github/workflows/real-weights.yml",
    "crates/llm/candle-llm/src/provider.rs",
    "docs/reference/qwen38-bonsai2-source-inventory.md",
    "docs/reference/qwen38/native-memory-admission.md",
    "release/qwen38-bonsai-matrix.json",
    "release/real-weight-models.toml",
    "scripts/release/qwen38_bonsai_artifacts.py",
    "scripts/release/qwen38_bonsai_terminal.py",
    "scripts/tests/test_ci_workflow_policy.py",
    "scripts/tests/test_qwen38_bonsai_artifacts.py",
    "scripts/tests/test_qwen38_bonsai_terminal.py",
}
BRIDGE_PATHS = {
    "release/VERSION",
    "release/README.md",
    "scripts/release/qwen38_bonsai_rc7_bridge.py",
    "scripts/tests/test_qwen38_bonsai_rc7_bridge.py",
}
CPU_CELLS = {
    "candle-cpu-qwen38-parent",
    "candle-cpu-bonsai-gguf",
    "candle-cpu-qwen3vl-baseline",
}
SUPPORTED_PROFILES = {
    "bonsai-qwen38-parent": ["mlx-unified", "candle-dense-cuda"],
    "bonsai-mlx-2bit": ["mlx-unified", "candle-packed-cuda"],
    "bonsai-gguf": ["mlx-unified", "candle-packed-cuda"],
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def git_file(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def changed_paths(base: str, head: str) -> set[str]:
    return set(git("diff", "--name-only", base, head).splitlines())


def validate_source_scope(*, head: str, clean: bool, version: str) -> None:
    if not clean:
        raise ValueError("candidate checkout must be clean")
    if git("rev-parse", "HEAD") != head or head == OBSERVED_SHA:
        raise ValueError("candidate SHA must be the clean current HEAD, distinct from RC7")
    if git("merge-base", OBSERVED_SHA, SCOPED_SHA) != OBSERVED_SHA:
        raise ValueError("scoped source does not descend from RC7")
    if git("merge-base", SCOPED_SHA, head) != SCOPED_SHA:
        raise ValueError("candidate does not descend from the reviewed accelerator-only source")
    if changed_paths(OBSERVED_SHA, SCOPED_SHA) != SCOPED_CHANGED_PATHS:
        raise ValueError("reviewed accelerator-only source changed unexpectedly")
    if not changed_paths(SCOPED_SHA, head) <= BRIDGE_PATHS:
        raise ValueError("candidate changes supported runtime or another unreviewed path")
    if version != "runtime-2026.09.0-rc.8\n":
        raise ValueError("candidate is not the reviewed RC8 version-only successor")
    # The scope commit pins every runtime and harness byte, including the CPU-only guard.
    # No subsequent provider/decode/kernel/oracle change can inherit these observations.
    if git("rev-parse", f"{head}:crates/llm/candle-llm/src/provider.rs") != "e3db6e703fc7408975eaa149199fb01f42621e4c":
        raise ValueError("reviewed CPU guard changed")


def validate_matrix(old: dict, new: dict) -> None:
    old_ids = {cell["id"] for cell in old["cells"]}
    if {cell["id"] for cell in new["cells"]} != old_ids - CPU_CELLS:
        raise ValueError("accelerator matrix cell set changed")
    retained = copy.deepcopy(old)
    retained["cells"] = [cell for cell in old["cells"] if cell["id"] not in CPU_CELLS]
    if new != retained:
        raise ValueError("retained cases, groups, acceptance or oracles changed")
    terminal.validate_full_acceptance_contract(new)


def validate_manifest(old_models: dict, new_models: dict) -> None:
    actual = {item["key"]: item for item in new_models["models"]}
    if any(actual[key].get("supported_execution_profiles") != profiles for key, profiles in SUPPORTED_PROFILES.items()):
        raise ValueError("supported model execution profiles changed")
    scrubbed = copy.deepcopy(new_models)
    for item in scrubbed["models"]:
        item.pop("supported_execution_profiles", None)
    if scrubbed != old_models:
        raise ValueError("model revisions, files, pinned sizes or other manifest data changed")


def validate_retained_contract() -> None:
    old = json.loads(git_file(OBSERVED_SHA, "release/qwen38-bonsai-matrix.json"))
    new = json.loads((ROOT / "release/qwen38-bonsai-matrix.json").read_text(encoding="utf-8"))
    validate_matrix(old, new)
    old_models = tomllib.loads(git_file(OBSERVED_SHA, "release/real-weight-models.toml").decode("utf-8"))
    new_models = tomllib.loads((ROOT / "release/real-weight-models.toml").read_text(encoding="utf-8"))
    validate_manifest(old_models, new_models)


def metadata_url(artifact_id: int) -> str:
    return f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}"


def fetch_metadata(artifact_id: int) -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(metadata_url(artifact_id), headers=headers), timeout=20) as response:
        return json.load(response)


def validate_artifact_metadata(role: str, item: dict) -> None:
    expected = ARTIFACTS[role]
    if (
        item.get("id") != expected["id"]
        or item.get("name") != f"qwen38-bonsai-{role}-{OBSERVED_SHA}-{RUN_ID}-{expected['attempt']}"
        or item.get("digest") != expected["digest"]
        or item.get("expired") is not False
        or (item.get("workflow_run") or {}).get("id") != RUN_ID
        or (item.get("workflow_run") or {}).get("head_sha") != OBSERVED_SHA
    ):
        raise ValueError(f"{role} artifact identity, source, attempt or digest differs")


def extract_verified_archive(role: str, archive: Path, destination: Path) -> None:
    expected = ARTIFACTS[role]
    if "sha256:" + terminal.sha256(archive) != expected["digest"]:
        raise ValueError(f"{role} archive bytes differ from GitHub artifact digest")
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if sum(entry.file_size for entry in entries) > 100_000_000:
            raise ValueError(f"{role} archive exceeds evidence size limit")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts or entry.filename in seen:
                raise ValueError(f"{role} archive has an unsafe or duplicate member")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"{role} archive contains a symlink")
            seen.add(entry.filename)
        source.extractall(destination)


def verify(args: argparse.Namespace) -> int:
    head = git("rev-parse", "HEAD")
    validate_source_scope(
        head=head,
        clean=not bool(git("status", "--porcelain=v1", "--untracked-files=all")),
        version=(ROOT / "release/VERSION").read_text(encoding="utf-8"),
    )
    validate_retained_contract()
    if args.output.exists():
        raise ValueError("bridge output already exists")
    metadata = {}
    for role in artifacts.ROLES:
        item = fetch_metadata(ARTIFACTS[role]["id"])
        validate_artifact_metadata(role, item)
        metadata[role] = item
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        roots = []
        for role in artifacts.ROLES:
            archive = getattr(args, f"{role}_archive")
            root = temp / role
            root.mkdir()
            extract_verified_archive(role, archive, root)
            artifacts.check_partition(argparse.Namespace(
                role=role, root=root, runtime_sha=OBSERVED_SHA, cell=None,
                matrix=ROOT / "release/qwen38-bonsai-matrix.json",
                manifest=ROOT / "release/real-weight-models.toml",
            ))
            roots.append(root)
        args.output.mkdir(parents=True)
        selection = {
            "schema_version": 1, "runtime_sha": OBSERVED_SHA, "run_id": str(RUN_ID),
            "selected": [
                {"role": role, "name": metadata[role]["name"],
                 "artifact_id": ARTIFACTS[role]["id"], "archive_digest": ARTIFACTS[role]["digest"],
                 "run_attempt": ARTIFACTS[role]["attempt"]}
                for role in artifacts.ROLES
            ],
        }
        terminal.write_new(args.output / "selected-artifacts.json", selection)
        common = dict(root=roots, matrix=ROOT / "release/qwen38-bonsai-matrix.json",
                      manifest=ROOT / "release/real-weight-models.toml",
                      runtime_sha=OBSERVED_SHA, output=args.output / "matrix-report.json",
                      markdown=args.output / "matrix-report.md", seal=args.output / "matrix-seal.json",
                      artifact_selection=args.output / "selected-artifacts.json")
        if terminal.matrix_status(argparse.Namespace(**common)) != 0:
            raise ValueError("16 accelerator cells did not pass scoped functional acceptance")
        terminal.verify_matrix_seal(argparse.Namespace(**common))
    report = json.loads((args.output / "matrix-report.json").read_text(encoding="utf-8"))
    terminal.write_new(args.output / "reuse-bridge.json", {
        "schema_version": 1,
        "observed_runtime_sha": OBSERVED_SHA,
        "candidate_runtime_sha": head,
        "reviewed_scope_sha": SCOPED_SHA,
        "source_run_id": RUN_ID,
        "source_artifacts": selection["selected"],
        "observed_matrix_seal_sha256": terminal.sha256(args.output / "matrix-seal.json"),
        "functional_acceptance_passed": report["functional_acceptance"]["passed"],
        "source_full_19_cell_campaign_passed": False,
        "model_execution_performed_on_candidate": False,
        "quality_threshold_applied": False,
        "note": "Historical RC7 model observations; scoped accelerator reuse, not new-SHA model execution.",
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-archive", type=Path, required=True)
    parser.add_argument("--cuda-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return verify(args)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"qwen38-bonsai-rc7-bridge: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
