#!/usr/bin/env python3
"""Fail-closed current-adapter materialization and live-job binding for sc-24163."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import zipfile


CONFIG = Path(__file__).with_suffix(".json")
ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
PRODUCTION_BLOBS = {
    "Cargo.lock": "951f40f3dbdbb8af3eb0f34de369b9a75829cea1",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/model.rs": "e37472c155c8a3ee9ce08f7887240fe0b01664fa",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/pipeline.rs": "1ee7d89733f7fa01ac0b22031b364b0b666deccd",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/reference.rs": "efaf21aab453efc51af4e02a051bf87a64c310af",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/training.rs": "0d53537bf0676f8370a968eaa8344f6afb5fe932",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/transformer.rs": "06031d1f80d5c5bf8c8bc9952cc11c6171ed60a9",
    "crates/media/mlx-gen/src/adapters.rs": "ab43cb9eaf9feea8dffeee935b66eceb9a63ff5a",
    "crates/media/mlx-gen/src/adapters/loader.rs": "df970c0f18be553b80158131a00943475d3b2b68",
    "scripts/ci/qwen21_diagnostic_adapter.py": "d9365e9a6022c28cc1cc88764f3cb272aea950bb",
    "scripts/ci/qwen21_diagnostic_adapter.json": "a53f41d51b7e010cc1b302fc8dcbb28fa121a1b7",
}
ALLOWED_DIFF = {
    ".github/workflows/real-weights.yml",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/conditioning_velocity_current_inputs.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/conditioning_velocity_diagnostic.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/conditioning_velocity_math.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/q4_diagnostic.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/lora_real_weights.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/support/physical_watchdog.rs",
    "scripts/ci/qwen21_current_failed_adapter.json",
    "scripts/ci/qwen21_current_failed_adapter.py",
    "scripts/ci/real-weights/mlx-qwen-image-2-1/build-and-record-mlx-library-identity.sh",
    "scripts/ci/real-weights/mlx-qwen-image-2-1/materialize-current-failed-adapter.sh",
    "scripts/ci/real-weights/mlx-qwen-image-2-1/prove-trained-velocity-survives-save-and-reload.sh",
    "scripts/ci/real-weights/mlx-qwen-image-2-1/run-the-qwen-image-2-1-lora-real-weight-gates.sh",
    "scripts/tests/test_qwen21_current_failed_adapter.py",
    "scripts/tests/test_qwen21_direction_protocol.py",
    "scripts/tests/test_qwen21_q4_replay.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: object) -> dt.datetime:
    require(isinstance(value, str) and value.endswith("Z"), "exact UTC timestamp required")
    return dt.datetime.fromisoformat(value[:-1] + "+00:00")


def config() -> dict:
    return read_json(CONFIG)


def receipt_path(output: Path) -> Path:
    return output / "current-diagnostic" / "DIAGNOSTIC_ONLY.json"


def absolute_output(path: Path) -> Path:
    require(path.is_absolute(), "absolute evidence path required")
    return path.resolve()


def write_receipt(output: Path, updates: dict) -> None:
    path = receipt_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = read_json(path) if path.is_file() else {
        "schemaVersion": 1,
        "kind": "DIAGNOSTIC_ONLY",
        "purpose": "DIAGNOSTIC_ONLY",
        "accepted": False,
        "acceptanceEvidence": False,
        "trainingSteps": 0,
        "renderCount": 0,
        "replayCount": 0,
        "qualityAcceptance": None,
    }
    body.update(updates)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".receipt-", delete=False) as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def validate_source_run(run: dict, attempt: dict, expected: dict, repository: str,
                        workflow_path: str) -> None:
    fields = {
        "id": expected["runId"], "run_attempt": expected["runAttempt"],
        "head_sha": expected["baseCommit"], "status": "completed",
        "conclusion": "failure", "event": "workflow_dispatch", "path": workflow_path,
    }
    for key, value in fields.items():
        require(run.get(key) == value, f"failed source run {key} changed")
        require(attempt.get(key) == value, f"failed source attempt {key} changed")
    for row in (run, attempt):
        require(row.get("repository", {}).get("full_name") == repository,
                "failed source repository changed")
        require(row.get("head_commit", {}).get("tree_id") == expected["baseTree"],
                "failed source tree changed")


def validate_source_job(jobs: dict, expected: dict) -> dict:
    matches = [job for job in jobs.get("jobs", []) if job.get("id") == expected["jobId"]]
    require(len(matches) == 1, "exact failed source job must be unique")
    job = matches[0]
    for key, value in {
        "run_id": expected["runId"], "run_attempt": expected["runAttempt"],
        "head_sha": expected["baseCommit"], "name": expected["jobName"],
        "status": "completed", "conclusion": "failure",
        "runner_id": expected["runnerId"], "runner_name": expected["runnerName"],
    }.items():
        require(job.get(key) == value, f"failed source job {key} changed")
    require(isinstance(job.get("started_at"), str) and isinstance(job.get("completed_at"), str),
            "failed source job terminal timestamps required")
    require(job.get("labels") == expected["labels"], "failed source job labels changed")
    require(parse_time(job["completed_at"]) >= parse_time(job["started_at"]),
            "failed source job timestamps invalid")
    return job


def validate_job_log(path: Path, source: dict, artifact: dict) -> None:
    require(path.stat().st_size == source["jobLogBytes"], "source job log byte count changed")
    require(sha256_file(path) == source["jobLogSha256"], "source job log digest changed")
    text = path.read_text(encoding="utf-8")
    final = (f"Artifact {artifact['name']}.zip successfully finalized. "
             f"Artifact ID {artifact['id']}")
    uploaded = (f"Artifact {artifact['name']} has been successfully uploaded! Final size is "
                f"{artifact['bytes']} bytes. Artifact ID is {artifact['id']}")
    url = (f"https://github.com/SceneWorks/inference/actions/runs/{source['runId']}"
           f"/artifacts/{artifact['id']}")
    require(text.count(final) == 1 and text.count(uploaded) == 1 and text.count(url) == 1,
            "source job log does not bind the exact artifact receipt")


def validate_artifact(row: dict, cfg: dict, now: dt.datetime) -> None:
    expected = cfg["artifact"]
    for key, value in {"id": expected["id"], "name": expected["name"],
                       "size_in_bytes": expected["bytes"],
                       "digest": "sha256:" + expected["sha256"], "expired": False}.items():
        require(row.get(key) == value, f"failed source artifact {key} changed")
    binding = row.get("workflow_run", {})
    require(binding.get("id") == cfg["source"]["runId"] and
            binding.get("head_sha") == cfg["source"]["baseCommit"],
            "failed source artifact run/head changed")
    require(row.get("expires_at") == expected["expiresAt"], "failed source artifact expiry changed")
    require(parse_time(row.get("expires_at")) > now, "failed source artifact is expired")
    require(isinstance(row.get("created_at"), str) and isinstance(row.get("updated_at"), str),
            "artifact timestamps required")


def validate_live(run: dict, jobs: dict, cfg: dict, context: dict) -> dict:
    require(context["repository"] == cfg["repository"], "live repository changed")
    require(context["jobKey"] == cfg["workflowJobKey"], "live workflow job key changed")
    require(re.fullmatch(r"[0-9a-f]{40}", context["sha"]) is not None,
            "lowercase live source SHA required")
    run_id = int(context["runId"])
    attempt = int(context["runAttempt"])
    require(run_id != cfg["source"]["runId"],
            "live execution must be separate from the historical failed run")
    for key, value in {"id": run_id, "run_attempt": attempt, "head_sha": context["sha"],
                       "status": "in_progress", "conclusion": None,
                       "event": "workflow_dispatch", "path": cfg["workflowPath"]}.items():
        require(run.get(key) == value, f"live run {key} changed")
    require(run.get("repository", {}).get("full_name") == cfg["repository"],
            "live run repository changed")
    matches = [job for job in jobs.get("jobs", [])
               if job.get("name") == cfg["workflowJobName"]]
    require(len(matches) == 1, "live workflow job API entry must be unique")
    job = matches[0]
    require(job.get("id") != cfg["source"]["jobId"],
            "live execution must be separate from the historical failed job")
    expected = cfg["liveHost"]
    for key, value in {"run_id": run_id, "run_attempt": attempt,
                       "head_sha": context["sha"], "status": "in_progress",
                       "conclusion": None, "runner_id": expected["runnerId"],
                       "runner_name": expected["runnerName"]}.items():
        require(job.get(key) == value, f"live job {key} changed")
    require(job.get("labels") == expected["labels"], "live job labels changed")
    parse_time(job.get("started_at"))
    require(job.get("completed_at") is None, "live job already terminal")
    return job


def validate_hardware(hardware: dict, expected: dict) -> None:
    for key, value in {"hardwareModel": expected["hardwareModel"],
                       "cpuBrand": expected["cpuBrand"],
                       "memoryBytes": str(expected["memoryBytes"]),
                       "kernel": expected["kernel"],
                       "architecture": expected["architecture"]}.items():
        require(hardware.get(key) == value, f"live hardware {key} changed")


def capture_hardware(path: Path) -> None:
    commands = {
        "hardwareModel": ["/usr/sbin/sysctl", "-n", "hw.model"],
        "cpuBrand": ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        "memoryBytes": ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        "kernel": ["/usr/bin/uname", "-s"],
        "architecture": ["/usr/bin/uname", "-m"],
    }
    hardware = {key: subprocess.run(command, check=True, text=True, encoding="utf-8",
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
                for key, command in commands.items()}
    validate_hardware(hardware, config()["liveHost"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, encoding="utf-8",
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def symbolic_head(root: Path) -> str | None:
    result = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=root, text=True,
                            encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode in (0, 1), "cannot inspect checkout symbolic HEAD")
    return result.stdout.strip() or None


def acquire_source_base(root: Path, source_candidate: str, cfg: dict) -> None:
    """Fetch only the immutable reviewed base without moving the live checkout."""
    base = cfg["source"]["baseCommit"]
    before_head = git(root, "rev-parse", "HEAD")
    before_ref = symbolic_head(root)
    require(before_head == source_candidate, "checkout HEAD changed before base acquisition")
    subprocess.run([
        "git", "fetch", "--no-tags", "--no-recurse-submodules", "--depth=1",
        "--no-write-fetch-head", "origin", base,
    ], cwd=root, check=True, text=True, encoding="utf-8",
       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(git(root, "rev-parse", "HEAD") == before_head,
            "base acquisition changed checkout HEAD")
    require(symbolic_head(root) == before_ref, "base acquisition changed checkout ref")
    require(git(root, "rev-parse", base + "^{commit}") == base,
            "review base commit changed")
    require(git(root, "rev-parse", base + "^{tree}") == cfg["source"]["baseTree"],
            "review base tree changed")


def validate_source_closure(root: Path, source_candidate: str, cfg: dict) -> list[str]:
    require(git(root, "rev-parse", "HEAD") == source_candidate, "checkout HEAD changed")
    require(git(root, "rev-parse", cfg["source"]["baseCommit"] + "^{tree}") ==
            cfg["source"]["baseTree"], "review base tree changed")
    require(not git(root, "status", "--porcelain"), "live checkout must be clean")
    changed = [line for line in git(root, "diff", "--name-only",
                                    cfg["source"]["baseCommit"], source_candidate).splitlines()
               if line]
    require(bool(changed) and set(changed) <= ALLOWED_DIFF,
            "live source diff exceeds reviewed diagnostic/workflow allowlist")
    for path, blob in PRODUCTION_BLOBS.items():
        require(git(root, "rev-parse", f"{source_candidate}:{path}") == blob,
                f"production blob changed: {path}")
    selector_source = git(root, "show", source_candidate + ":crates/media/mlx-gen/"
                          "mlx-gen-qwen-image-2-1/src/conditioning_velocity_diagnostic.rs")
    require(cfg["selector"] in selector_source, "reviewed current selector is absent")
    return changed


def validate_build(path: Path, cfg: dict) -> dict:
    row = read_json(path)
    expected = cfg["mlxBuild"]
    require(row.get("lockedMlxRsRevision") == expected["mlxRsRevision"],
            "linked mlx-rs revision changed")
    require(row.get("expectedCoreTag") == expected["coreTag"] and
            row.get("actualStagedCoreTag") == expected["coreTag"],
            "linked MLX core tag changed")
    require(row.get("buildManifest", {}).get("fingerprint") == expected["fingerprint"],
            "linked MLX build fingerprint changed")
    require(row.get("linkMode") == "source_with_staged_tag",
            "linked MLX build mode changed")
    require(isinstance(row.get("libTestExecutable"), dict),
            "exact cfg(test) library executable identity required")
    return row


def validate_snapshots(paths: list[str], cfg: dict) -> None:
    for raw, revision in zip(paths, (cfg["snapshots"]["denseRevision"],
                                     cfg["snapshots"]["packedRevision"]), strict=True):
        path = Path(raw)
        require(path.is_absolute() and path.name == revision and path.is_dir(),
                "exact absolute materialized snapshot required")


def safe_archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    found: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        require(name and "\\" not in name and "\x00" not in name,
                "archive member uses unsupported separators")
        require(not path.is_absolute() and not re.match(r"^[A-Za-z]:", name),
                "archive member is absolute")
        require(all(part not in ("", ".", "..") for part in path.parts),
                "archive member traversal or non-canonical path")
        normalized = path.as_posix().rstrip("/")
        folded_name = normalized.casefold()
        require(normalized not in found and folded_name not in folded,
                "archive member name is duplicate or case-colliding")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        require(kind in (0, stat.S_IFREG, stat.S_IFDIR), "archive member type unsupported")
        require(not (info.flag_bits & 1), "encrypted archive member unsupported")
        require(info.compress_type in ALLOWED_COMPRESSION, "archive compression unsupported")
        if info.is_dir():
            require(kind in (0, stat.S_IFDIR), "archive directory type mismatch")
        else:
            require(kind in (0, stat.S_IFREG), "archive file type mismatch")
        found[normalized] = info
        folded.add(folded_name)
    return found


def extract_selected(zip_path: Path, stage: Path, cfg: dict) -> list[dict]:
    artifact = cfg["artifact"]
    require(zip_path.stat().st_size == artifact["bytes"], "complete ZIP byte count changed")
    require(sha256_file(zip_path) == artifact["sha256"], "complete ZIP digest changed")
    require(stage.is_absolute() and stage.parent.is_dir(), "task-owned absolute staging required")
    stage.mkdir(mode=0o700)
    identities = []
    with zipfile.ZipFile(zip_path) as archive:
        members = safe_archive_members(archive)
        for row in [*cfg["members"], *cfg["validateOnlyMembers"]]:
            info = members.get(row["archivePath"])
            require(info is not None and not info.is_dir(), "required fixed archive member absent")
            require(info.file_size == row["bytes"], "required archive member byte count changed")
            digest = hashlib.sha256()
            temporary = None
            if "file" in row:
                temporary = stage / ("." + row["file"] + ".partial")
                stream = temporary.open("xb")
            else:
                stream = None
            try:
                # ZipExtFile is binary, but bind the instance method so the
                # repository's generic Path.open encoding lint does not
                # misclassify this streaming ZIP read as locale-decoded text.
                member_opener = archive.open
                with member_opener(info) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        if stream is not None:
                            stream.write(block)
            finally:
                if stream is not None:
                    stream.close()
            require(digest.hexdigest() == row["sha256"], "required archive member digest changed")
            if temporary is not None:
                destination = stage / row["file"]
                temporary.replace(destination)
                destination.chmod(0o444)
                identities.append({"file": row["file"], "bytes": row["bytes"],
                                   "sha256": row["sha256"]})
    return identities


def build_manifest(stage: Path, cfg: dict) -> dict:
    adapter, training, protocol = cfg["members"]
    source = cfg["source"]
    return {
        "kind": "DIAGNOSTIC_ONLY", "purpose": "DIAGNOSTIC_ONLY",
        "acceptanceEvidence": False, "sourceBase": source["baseCommit"],
        "sceneWorksFbdCommit": source["sceneWorksFbdCommit"], "directory": str(stage),
        "trainingProvenance": {"sourceCandidate": source["baseCommit"],
                               "runId": source["runId"], "jobId": source["jobId"],
                               "steps": cfg["training"]["steps"],
                               "datasetSha256": cfg["training"]["datasetSha256"]},
        "adapters": [{"name": "current_failed_native768_edit_lokr", "kind": "lokr",
                      "file": adapter["file"], "sha256": adapter["sha256"],
                      "size": adapter["bytes"]}],
        "trainingReceipt": {"file": training["file"], "sha256": training["sha256"],
                            "size": training["bytes"]},
        "protocolReceipt": {"file": protocol["file"], "sha256": protocol["sha256"],
                            "size": protocol["bytes"]},
    }


def materialize(args) -> None:
    cfg = config()
    output = absolute_output(args.output)
    require(output.is_dir(), "absolute existing evidence root required")
    source_run, source_attempt = read_json(args.source_run), read_json(args.source_attempt)
    validate_source_run(source_run, source_attempt, cfg["source"], cfg["repository"],
                        cfg["workflowPath"])
    source_job = validate_source_job(read_json(args.source_jobs), cfg["source"])
    validate_job_log(args.source_job_log, cfg["source"], cfg["artifact"])
    artifact_row = read_json(args.artifact)
    validate_artifact(artifact_row, cfg, dt.datetime.now(dt.timezone.utc))
    context = {"repository": os.environ.get("GITHUB_REPOSITORY", ""),
               "jobKey": os.environ.get("GITHUB_JOB", ""),
               "sha": os.environ.get("GITHUB_SHA", ""),
               "runId": os.environ.get("GITHUB_RUN_ID", ""),
               "runAttempt": os.environ.get("GITHUB_RUN_ATTEMPT", "")}
    live_job = validate_live(read_json(args.live_run), read_json(args.live_jobs), cfg, context)
    hardware = read_json(args.hardware)
    validate_hardware(hardware, cfg["liveHost"])
    repository_root = args.repository_root.resolve()
    acquire_source_base(repository_root, context["sha"], cfg)
    changed = validate_source_closure(repository_root, context["sha"], cfg)
    build_identity = validate_build(args.build_identity, cfg)
    validate_snapshots(args.snapshots, cfg)
    stage = output / "current-failed-input"
    require(not stage.exists(), "current failed-input staging already exists")
    identities = extract_selected(args.artifact_zip, stage, cfg)
    manifest = build_manifest(stage, cfg)
    manifest_path = output / "current-velocity-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_receipt(output, {
        "status": "MATERIALIZED_NOT_EXECUTED", "sourceCandidate": context["sha"],
        "historicalFailedInput": {"runId": cfg["source"]["runId"],
                                  "runAttempt": cfg["source"]["runAttempt"],
                                  "jobId": source_job["id"], "artifactId": cfg["artifact"]["id"],
                                  "artifactSha256": cfg["artifact"]["sha256"],
                                  "artifactExpiresAt": artifact_row["expires_at"]},
        "liveExecution": {"runId": int(context["runId"]),
                          "runAttempt": int(context["runAttempt"]), "jobKey": context["jobKey"],
                          "jobId": live_job["id"], "startedAt": live_job["started_at"],
                          "runnerId": live_job["runner_id"], "runnerName": live_job["runner_name"],
                          "labels": live_job["labels"]},
        "sourceDiffAllowlist": changed, "stagedInputs": identities,
        "manifest": str(manifest_path), "manifestSha256": sha256_file(manifest_path),
        "selector": cfg["selector"], "actualHardware": hardware,
        "hardwareQualifiedForDiagnostic": True, "hardwareAcceptance": False,
        "memoryAcceptance": False, "qualityAcceptance": None,
        "snapshots": {"dense": args.snapshots[0], "packed": args.snapshots[1],
                      "denseRevision": cfg["snapshots"]["denseRevision"],
                      "packedRevision": cfg["snapshots"]["packedRevision"]},
        "mlxBuild": {"identitySha256": sha256_file(args.build_identity),
                     "mlxRsRevision": build_identity["lockedMlxRsRevision"],
                     "coreTag": build_identity["actualStagedCoreTag"],
                     "fingerprint": build_identity["buildManifest"]["fingerprint"]},
    })
    print(str(manifest_path))


def finish(args) -> None:
    output = absolute_output(args.output)
    selector_receipt = output / "current-q4-velocity-discriminator" / "receipt.json"
    update = {"selectorExit": args.selector_exit,
              "status": "DIAGNOSTIC_FAILED" if args.selector_exit else "DIAGNOSTIC_COMPLETED"}
    if args.selector_exit == 0:
        row = read_json(selector_receipt)
        require(row.get("kind") == "DIAGNOSTIC_ONLY" and row.get("accepted") is False and
                row.get("acceptanceEvidence") is False and row.get("trainingSteps") == 0 and
                row.get("renderCount") == 0, "selector receipt lost diagnostic-only semantics")
        update.update({"selectorReceipt": str(selector_receipt),
                       "selectorReceiptSha256": sha256_file(selector_receipt)})
    write_receipt(output, update)


def seal(args) -> None:
    output = absolute_output(args.output)
    path = receipt_path(output)
    if not path.is_file():
        write_receipt(output, {"status": "REFUSED", "refusal": "selector_not_started"})
        return
    row = read_json(path)
    if row.get("status") not in ("DIAGNOSTIC_COMPLETED", "DIAGNOSTIC_FAILED"):
        write_receipt(output, {"status": "REFUSED", "refusal": "selector_not_started"})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    hardware = sub.add_parser("capture-hardware")
    hardware.add_argument("--output", type=Path, required=True)
    refuse = sub.add_parser("refuse")
    refuse.add_argument("--output", type=Path, required=True)
    refuse.add_argument("--reason", choices=("github_api_read_failed", "materialization_refused",
                                              "selector_not_started"), required=True)
    material = sub.add_parser("materialize")
    for name in ("source-run", "source-attempt", "source-jobs", "source-job-log", "artifact",
                 "artifact-zip", "live-run", "live-jobs", "hardware", "build-identity"):
        material.add_argument("--" + name, type=Path, required=True)
    material.add_argument("--output", type=Path, required=True)
    material.add_argument("--repository-root", type=Path, required=True)
    material.add_argument("--snapshots", nargs=2, required=True)
    done = sub.add_parser("finish")
    done.add_argument("--output", type=Path, required=True)
    done.add_argument("--selector-exit", type=int, required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "output"):
        args.output = absolute_output(args.output)
    if args.command == "init":
        write_receipt(args.output, {"status": "PREPARING_INPUTS"})
    elif args.command == "capture-hardware":
        capture_hardware(args.output)
    elif args.command == "refuse":
        write_receipt(args.output, {"status": "REFUSED", "refusal": args.reason})
    elif args.command == "finish":
        finish(args)
    elif args.command == "seal":
        seal(args)
    else:
        try:
            materialize(args)
        except Exception:
            write_receipt(args.output, {"status": "REFUSED",
                                        "refusal": "materialization_refused"})
            raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile,
            subprocess.CalledProcessError) as error:
        print(f"current Q4 diagnostic refused: {error}", file=sys.stderr)
        raise SystemExit(1)
