#!/usr/bin/env python3
"""Narrow external control for the dispatch-only YuE2 precision real-weight test.

This does not grade app admission: it records the engine's actual one-test verdict,
process release, and raw external device/footprint observations for later audit.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REFERENCE_SHA256 = "c4073edb3c7abfbf5d50a9c5bfa67c73b9b9a8ba20115570ddb616d00d5b72d4"
CUDA_CHILD_TIMEOUT_SECONDS = 180 * 60  # Operational cap, not a measured runtime.
CUDA_POSTFLIGHT_SECONDS = 600
CUDA_JOB_TIMEOUT_SECONDS = 240 * 60
TEST_NAME = "explicit_stage_precision_real_weights"
CUDA_SMOKES = (
    ("quant", "yue2_stable_conv::cuda_tests::bf16_conv_and_transpose_match_same_input_prefix_across_lengths"),
    ("vae", "vae::tests::bf16_cuda_both_vae_variants_decode_full_tiled_and_encode"),
)
BUILD_TARGETS = {
    "precision": ("candle-audio-yue2", "precision_real_weights", "test", "crates/audio/candle-audio-yue2/Cargo.toml", "crates/audio/candle-audio-yue2/tests/precision_real_weights.rs"),
    "quant": ("candle-quant-kernels", "candle_quant_kernels", "lib", "crates/kernels/candle-quant-kernels/Cargo.toml", "crates/kernels/candle-quant-kernels/src/lib.rs"),
    "vae": ("candle-audio-yue2", "candle_audio_yue2", "lib", "crates/audio/candle-audio-yue2/Cargo.toml", "crates/audio/candle-audio-yue2/src/lib.rs"),
}
DECODER_SHA256 = {
    "standard": "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346",
    "legacy": "b6d283628913bb41145ba99e2314eef613905ee95f690eb70e8212d5f4965044",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_revisions(engine_sha: str, control_sha: str) -> None:
    require(re.fullmatch(r"[0-9a-f]{40}", engine_sha) is not None, "engine SHA must be full lowercase hex")
    require(re.fullmatch(r"[0-9a-f]{40}", control_sha) is not None, "control SHA must be full lowercase hex")
    require(os.environ.get("GITHUB_SHA") == control_sha, "workflow control SHA differs from dispatch input")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, encoding="utf-8").stdout.strip()
    require(head == engine_sha, "engine checkout differs from dispatch input")


def resolve_binary(args: argparse.Namespace) -> None:
    package_name, target_name, target_kind, manifest, source = BUILD_TARGETS[args.target]
    root = Path.cwd().resolve()
    metadata = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
                              capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
    packages = [package for package in json.loads(metadata.stdout)["packages"]
                if package.get("name") == package_name and
                Path(package.get("manifest_path", "")).resolve() == root / manifest]
    require(len(packages) == 1, f"expected one exact workspace package {package_name}: {packages}")
    package_id = packages[0]["id"]
    candidates = []
    for line in args.build_json.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = row.get("target") or {}
        if (row.get("reason") == "compiler-artifact" and row.get("package_id") == package_id and
                target.get("name") == target_name and target.get("kind") == [target_kind] and
                Path(target.get("src_path", "")).resolve() == root / source and
                row.get("profile", {}).get("test") is True):
            executable = row.get("executable")
            if executable:
                candidates.append(Path(executable))
    require(len(candidates) == 1, f"expected exactly one {args.target} test executable; got {candidates}")
    require(candidates[0].is_file(), f"missing compiled test binary: {candidates[0]}")
    args.output.write_text(str(candidates[0].resolve()) + "\n", encoding="utf-8")
    write_json(Path(str(args.output) + ".identity.json"), {
        "target": args.target, "package_id": package_id, "target_name": target_name,
        "target_kind": target_kind, "manifest": str((root / manifest).resolve()),
        "source": str((root / source).resolve()),
        "build_json_sha256": sha256(args.build_json), "build_json": str(args.build_json.resolve()),
        "binary": str(candidates[0].resolve()), "binary_sha256": sha256(candidates[0]),
    })


def verify_binary_identity(binary: Path, label: str, evidence: Path) -> dict:
    stem = "binary.txt" if label == "precision" else f"{label}-binary.txt"
    identity = evidence / f"{stem}.identity.json"
    row = json.loads(identity.read_text(encoding="utf-8"))
    package, target, kind, manifest, source = BUILD_TARGETS[label]
    require(row.get("target") == label and row.get("target_name") == target and
            row.get("target_kind") == kind and row.get("manifest") == str((Path.cwd() / manifest).resolve()) and
            row.get("source") == str((Path.cwd() / source).resolve()) and
            row.get("binary") == str(binary.resolve()) and row.get("binary_sha256") == sha256(binary) and
            Path(row.get("build_json", "")).resolve().parent == evidence.resolve() and
            sha256(Path(row["build_json"])) == row.get("build_json_sha256") and
            package in row.get("package_id", ""), f"{label} build/binary identity changed after resolution")
    return row


def verify_reference(args: argparse.Namespace) -> None:
    from yue2_precision_reference_transfer import (  # type: ignore[import-not-found]
        LICENSE_SHA256, SOURCE_ARTIFACT_ID, SOURCE_ENGINE_SHA, SOURCE_METADATA_SHA256,
        SOURCE_RUN_ID, SOURCE_ZIP_SHA256, RELAY_ARTIFACT_ID, RELAY_CONTROL_SHA,
        RELAY_ENGINE_SHA, RELAY_METADATA_SHA256, RELAY_RUN_ID, RELAY_ZIP_SHA256,
    )
    metadata = json.loads((args.directory / "reference-provenance.json").read_text(encoding="utf-8"))
    source = args.directory / "vae_real_reference.safetensors"
    require(metadata.get("engine_sha") == args.engine_sha, "reference-stage engine SHA differs from this checkout")
    require(metadata.get("control_sha") == args.control_sha and
            metadata.get("transfer_run_id") == args.run_id and
            metadata.get("transfer_run_attempt") == "1" and
            metadata.get("runner") == "hosted-cpu-transfer" and
            metadata.get("source_run_id") == SOURCE_RUN_ID and
            metadata.get("source_run_attempt") == 1 and
            metadata.get("source_engine_sha") == SOURCE_ENGINE_SHA and
            metadata.get("source_artifact_id") == SOURCE_ARTIFACT_ID and
            metadata.get("source_artifact_zip_sha256") == SOURCE_ZIP_SHA256 and
            metadata.get("source_provenance_sha256") == SOURCE_METADATA_SHA256 and
            metadata.get("noncommercial_sha256") == LICENSE_SHA256,
            "reference transfer provenance differs from reviewed source/run")
    require(source.is_file(), "reference artifact is absent")
    relay_keys = {"reference_source_mode", "relay_run_id", "relay_artifact_id",
                  "relay_engine_sha", "relay_control_sha", "relay_artifact_zip_sha256",
                  "relay_provenance_sha256"}
    if relay_keys & metadata.keys():
        require({key: metadata.get(key) for key in relay_keys} == {
            "reference_source_mode": "relay", "relay_run_id": RELAY_RUN_ID,
            "relay_artifact_id": RELAY_ARTIFACT_ID,
            "relay_engine_sha": RELAY_ENGINE_SHA,
            "relay_control_sha": RELAY_CONTROL_SHA,
            "relay_artifact_zip_sha256": RELAY_ZIP_SHA256,
            "relay_provenance_sha256": RELAY_METADATA_SHA256,
        }, "reference relay identity differs from reviewed artifact")
    require(sha256(args.directory / "NONCOMMERCIAL.txt") == LICENSE_SHA256,
            "reference noncommercial notice differs from reviewed source")
    digest = sha256(source)
    require(digest == REFERENCE_SHA256 == metadata.get("sha256"), "reference digest differs from committed fixture")
    print(f"pinned external reference verified: {digest}, {source.stat().st_size} bytes")


def typed_compute_rows(output: str) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    columns: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if line.startswith("#"):
            names = fields[1:]
            if "pid" in names and "type" in names:
                columns = {name: index for index, name in enumerate(names)}
            continue
        require(columns, "nvidia-smi pmon output lacks typed process columns")
        pid_index, type_index = columns["pid"], columns["type"]
        require(len(fields) > max(pid_index, type_index), f"ambiguous nvidia-smi pmon row: {line}")
        gpu_index = columns.get("gpu")
        require(gpu_index is not None and len(fields) > gpu_index and fields[gpu_index] == "0",
                f"nvidia-smi pmon reported an unexpected physical GPU: {line}")
        if fields[pid_index] == "-" and fields[type_index] == "-":
            require(all(field == "-" or (index == gpu_index and field.isdigit())
                        for index, field in enumerate(fields)), f"ambiguous empty nvidia-smi pmon row: {line}")
            continue
        require(fields[pid_index].isdigit(), f"ambiguous nvidia-smi pmon PID: {line}")
        kind = fields[type_index]
        require(kind in {"C", "C+G", "G"}, f"unrecognized nvidia-smi pmon process type: {line}")
        if kind in {"C", "C+G"}:
            rows.append((line, int(fields[pid_index]), kind))
    require(columns, "nvidia-smi pmon output lacks typed process columns")
    return rows


def compute_capable_rows(output: str) -> list[str]:
    return [line for line, _, _ in typed_compute_rows(output)]


def query_compute_apps_rows(output: str) -> list[str]:
    """Fallback has no process type: every reported PID is conservatively busy."""
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = next(csv.reader([line]))
        require(len(fields) == 2 and fields[0].strip().isdigit() and fields[1].strip(),
                f"ambiguous nvidia-smi query-compute-apps row: {line}")
        rows.append(line)
    return rows


def cuda_census(*, admission: bool = True) -> tuple[str, list[str]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "1":
        from yue2_cuda_idle_context import census_shared_gpu1  # type: ignore[import-not-found]
        raw, verified = census_shared_gpu1(admission=admission)
        return raw, [] if verified else ["physical GPU1 shared-device proof refused"]
    command = ["nvidia-smi", "pmon", "-i", "0", "-c", "1", "-s", "um"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, encoding="utf-8")
    if result.returncode == 0:
        typed = typed_compute_rows(result.stdout)
        busy = [line for line, _, _ in typed]
        if not typed:
            try:
                from yue2_cuda_idle_context import census_empty_device  # type: ignore[import-not-found]
                raw, verified = census_empty_device(result.stdout)
                return raw, [] if verified else ["empty GPU0 physical probe refused"]
            except Exception as error:
                return (f"{result.stdout}\nempty GPU0 guard refused: {error}",
                        ["empty GPU0 physical probe refused"])
        # The normal typed guard still refuses every compute context. The only
        # exception is a currently reverified, receipt-bound WDDM C+G context;
        # pure C, multiple mixed rows, missing evidence, and faults stay busy.
        if len(typed) == 1 and typed[0][2] == "C+G" and os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID"):
            try:
                from yue2_cuda_idle_context import census_mixed_context  # type: ignore[import-not-found]
                raw, verified = census_mixed_context(typed[0][1], result.stdout)
                if verified:
                    return raw, []
                return raw, busy
            except Exception as error:
                return f"{result.stdout}\nreviewed C+G guard refused: {error}", busy
        return result.stdout, busy
    # Some Windows drivers do not expose pmon. The supported apps query has no
    # C/G type, so conservatively refuse every process it reports.
    fallback = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20, encoding="utf-8"
    )
    require(fallback.returncode == 0,
            f"CUDA census unavailable: pmon: {result.stderr.strip()}; query-compute-apps: {fallback.stderr.strip()}")
    return (f"pmon unavailable: {result.stderr.strip()}\nquery-compute-apps:\n{fallback.stdout}",
            query_compute_apps_rows(fallback.stdout))


def physical_busy_message(raw: str, busy: list[str], context: str) -> str:
    """Display a saved guard refusal without changing the busy decision."""
    message = f"{context}: {busy}"
    try:
        probe = json.loads(raw)
    except (ValueError, TypeError):
        return message
    if isinstance(probe, dict):
        refusal = probe.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            message += f"; physical guard refused: {refusal}"
    return message


def cuda_physical_census(*, admission: bool = True) -> tuple[str, list[str]]:
    """Require complete selected-GPU seven-family evidence before model work.

    A reviewed C+G receipt selects one signed process; an actually process-free
    GPU0 requires a fresh unfiltered 29-file probe. The owner-approved shared
    GPU1 route keeps foreign actors and allocations in raw evidence, admits
    only the selected card before a child, and checks owned release afterward.
    """
    raw, busy = cuda_census(admission=admission)
    if busy:
        return raw, busy
    try:
        probe = json.loads(raw)
        require(isinstance(probe, dict) and isinstance(probe.get("diagnosticFiles"), dict) and
                isinstance(probe.get("diagnosticFileBytesB64"), dict) and
                len(probe["diagnosticFiles"]) == len(probe["diagnosticFileBytesB64"]) == 29 and
                set(probe["diagnosticFiles"]) == set(probe["diagnosticFileBytesB64"]) and
                probe.get("commandExit") == 0 and
                probe.get("physicalMode") in (None, "empty-gpu0", "shared-gpu1") and
                (probe.get("physicalMode") not in ("empty-gpu0", "shared-gpu1") or
                 probe.get("validatedDevice", {}).get("physicalMode") == probe["physicalMode"]) and
                (probe.get("physicalMode") != "shared-gpu1" or
                 (probe.get("admission") is admission and
                  probe.get("validatedDevice", {}).get("admission") is admission and
                  probe.get("validatedDevice", {}).get("physicalIndex") == 1 and
                  probe.get("validatedDevice", {}).get("cudaOrdinal") == 0)) and
                "refusal" not in probe, "complete selected-device physical probe absent")
    except (ValueError, TypeError, RuntimeError):
        return raw, ["complete selected-device seven-family physical proof absent"]
    return raw, []


def retain_cuda_physical_evidence(evidence: Path, label: str, raw: str) -> list[dict]:
    probe = json.loads(raw)
    files = probe["diagnosticFiles"]
    raw_files = probe["diagnosticFileBytesB64"]
    require(isinstance(files, dict) and isinstance(raw_files, dict) and
            len(files) == len(raw_files) == 29 and set(files) == set(raw_files),
            "fresh physical probe inventory incomplete")
    directory = evidence / f"physical-{label}"
    require(not directory.exists(), "refuse to replace a physical proof")
    directory.mkdir()
    inventory = []
    for name, contents in sorted(files.items()):
        require(re.fullmatch(r"[a-z0-9.-]+\.json", name) is not None and
                isinstance(contents, str), "unsafe physical probe member")
        data = base64.b64decode(raw_files[name], validate=True)
        require(data.decode("utf-8-sig") == contents, "fresh physical probe raw bytes disagree")
        path = directory / name
        path.write_bytes(data)
        inventory.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return inventory


def retain_reviewed_baseline(evidence: Path, directory: Path) -> list[dict]:
    from yue2_cuda_idle_context import BASELINE_DIGEST, artifact_digest  # type: ignore[import-not-found]
    target = evidence / "reviewed-idle-context"
    require(not target.exists(), "refuse to replace reviewed owner receipt")
    source_files = sorted(directory.iterdir())
    require(len(source_files) == 28 and all(file.is_file() and not file.is_symlink() and
                                           file.suffix == ".json" for file in source_files),
            "reviewed owner receipt inventory changed")
    target.mkdir()
    inventory = []
    for file in source_files:
        copied = target / file.name
        shutil.copyfile(file, copied)
        require(sha256(file) == sha256(copied), "reviewed owner receipt changed during copy")
        inventory.append({"name": file.name, "bytes": copied.stat().st_size, "sha256": sha256(copied)})
    require(artifact_digest(target) == BASELINE_DIGEST,
            "copied reviewed owner receipt differs from pinned baseline")
    return inventory


def metal_worker_executable(name: str) -> bool:
    # Inspect only the executable, never argv: the GitHub controller may quote
    # a worker name in its own command line without running that worker.
    return bool(
        re.fullmatch(r"[a-z0-9_]*real_weights-[0-9a-f]+", name)
        or re.fullmatch(r"real_weight_tiling(?:-[0-9a-f]+)?", name)
        or name.startswith(("mlx-gen-", "mlx_gen_", "candle-gen-", "candle_gen_"))
        or name in {
            "mlx-gen", "mlx_gen", "candle-gen", "candle_gen",
            "sceneworks-worker", "sceneworks-rust-api", "sceneworks-api",
            "memory-mlx-adapter", "memory-candle-adapter", "candle_audio_yue2",
            "candle-audio-yue2",
        }
    )


def metal_census() -> tuple[str, list[str]]:
    result = subprocess.run(["/bin/ps", "-axo", "pid=,comm="], capture_output=True, text=True, timeout=20, encoding="utf-8")
    require(result.returncode == 0, f"ps executable census failed: {result.stderr.strip()}")
    busy = []
    # Executable names only: argv/controller text must not cause a false positive.
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        name = Path(fields[1]).name
        if metal_worker_executable(name):
            busy.append(line)
    return result.stdout, busy


def sample_cuda() -> dict:
    started = time.time_ns()
    index = "1" if os.environ.get("CUDA_VISIBLE_DEVICES") == "1" else "0"
    command = ["nvidia-smi", "-i", index, "--query-gpu=timestamp,index,memory.used,memory.free",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, encoding="utf-8")
    require(result.returncode == 0, result.stderr.strip() or "nvidia-smi sample failed")
    rows = [next(csv.reader([line], skipinitialspace=True)) for line in result.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 4 and rows[0][1].strip() == index and
            all(rows[0][column].strip().isdigit() for column in (2, 3)),
            "selected physical CUDA sample missing or ambiguous")
    return {"started_utc_ns": started, "ended_utc_ns": time.time_ns(),
            "method": "nvidia-smi query-gpu", "physical_gpu_index": int(index),
            "raw": result.stdout.strip()}


def sample_metal(pid: int) -> dict:
    started = time.time_ns()
    with tempfile.TemporaryDirectory(prefix="yue2-footprint-") as temp:
        output = Path(temp) / "footprint.json"
        command = ["/usr/bin/footprint", "--noCategories", "-j", str(output), "-p", str(pid)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, encoding="utf-8")
        require(result.returncode == 0, result.stderr.strip() or "footprint sample failed")
        payload = json.loads(output.read_text(encoding="utf-8"))
        matches = [p for p in payload.get("processes", []) if p.get("pid") == pid]
        require(len(matches) == 1, "footprint omitted or duplicated the owned test PID")
        value = matches[0].get("auxiliary", {}).get("phys_footprint")
        require(isinstance(value, int) and value >= 0, "footprint omitted phys_footprint")
        return {"started_utc_ns": started, "ended_utc_ns": time.time_ns(),
                "method": "Darwin phys_footprint", "pid": pid,
                "phys_footprint_bytes": value}


def wait_owned_child(child: subprocess.Popen, backend: str, timeout: float | None = None,
                     *, owned_tree: bool = False) -> tuple[int | None, bool, str | None]:
    """Bound only the Popen-owned precision test, leaving other processes untouched."""
    try:
        return child.wait(timeout=(CUDA_CHILD_TIMEOUT_SECONDS if timeout is None else timeout)
                          if backend == "cuda" else None), False, None
    except subprocess.TimeoutExpired:
        code, cleanup_error = reap_owned_child(child, owned_tree=owned_tree)
        return code, True, cleanup_error
    except Exception as error:
        code, cleanup_error = reap_owned_child(child, owned_tree=owned_tree)
        return code, False, f"{error}; cleanup: {cleanup_error}" if cleanup_error else str(error)


def reap_owned_child(child: subprocess.Popen, *, owned_tree: bool = False) -> tuple[int | None, str | None]:
    try:
        if child.poll() is None:
            if owned_tree:
                # The root is still our live Popen process. Windows taskkill /T
                # follows only this root's process tree, never a name/PID search.
                result = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                        capture_output=True, text=True, encoding="utf-8", timeout=15)
                require(result.returncode == 0 or child.poll() is not None,
                        "owned CUDA process-tree termination failed")
            else:
                child.kill()
            return child.wait(timeout=30), None
        return child.poll(), None
    except Exception as error:
        return child.poll(), str(error)


def one_test_executed(output: str) -> bool:
    return exact_one_test_executed(output, TEST_NAME)


def exact_one_test_executed(output: str, name: str) -> bool:
    headers = list(re.finditer(r"^test (?!result:)(\S+) \.\.\.(.*)$", output, re.MULTILINE))
    verdicts = re.findall(r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", output, re.MULTILINE)
    if len(headers) != 1 or headers[0].group(1) != name or verdicts != [("1", "0", "0")]:
        return False
    if re.findall(r"^running (\d+) test(?:s)?\s*$", output, re.MULTILINE) != ["1"]:
        return False
    tail = headers[0].group(2).strip()
    summary = output.find("test result:", headers[0].end())
    return tail == "ok" or (summary >= 0 and re.search(r"(?m)^ok\s*$", output[headers[0].end():summary]) is not None)


def remaining_cuda_budget(total_deadline: float, job_start_ns: int) -> float:
    remaining = total_deadline - time.monotonic()
    require(remaining > 0, "combined CUDA child deadline expired before next exact test")
    if os.environ.get("YUE2_IDLE_CONTEXT_RUN_ID"):
        from yue2_cuda_idle_context import require_remaining_window  # type: ignore[import-not-found]
        require_remaining_window(remaining + CUDA_POSTFLIGHT_SECONDS)
    require(time.time_ns() + (remaining + CUDA_POSTFLIGHT_SECONDS) * 1_000_000_000 <=
            job_start_ns + CUDA_JOB_TIMEOUT_SECONDS * 1_000_000_000,
            "combined CUDA child cannot finish before workflow upload tail")
    return remaining


def child_stage_succeeded(row: dict) -> bool:
    return (row["exit_code"] == 0 and not row["timed_out"] and row["wait_error"] is None and
            row["released"] and row["exact_one_test_passed"] and row["sample_count"] > 0 and
            row["binary_unchanged_after_child"] and
            not row["sampler_faults"] and row.get("owned_gpu_pid_released", True) and
            row.get("owned_descendants_released", True) and
            not row.get("post_census_error") and
            not row.get("post_census_busy"))


def descendant_generations(parents: dict[int, int], births: dict[int, int],
                           root_pid: int, root_birth: int) -> dict[int, int]:
    """Bind descendants to the Popen root's process generation, not reused PIDs."""
    require(root_pid not in births or births[root_pid] == root_birth,
            "owned CUDA root PID was reused")
    known = {root_pid: root_birth}
    for _ in range(len(parents)):
        children = {pid: parent for pid, parent in parents.items()
                    if parent in known and pid not in known}
        if not children:
            return {pid: birth for pid, birth in known.items() if pid != root_pid}
        for pid, parent in children.items():
            require(pid in births and births[pid] >= known[parent],
                    "owned CUDA descendant creation identity unavailable or older than parent")
            known[pid] = births[pid]
    raise RuntimeError("owned CUDA process lineage is cyclic or ambiguous")


def windows_process_birth(handle: int) -> int:
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    times = [wintypes.FILETIME() for _ in range(4)]
    require(kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)) != 0,
            "owned CUDA process creation time unavailable")
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def windows_owned_descendants(root_pid: int, root_birth: int) -> dict[int, int]:
    """Read only PID, parent PID, and creation for the root's Windows tree."""
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    require(snapshot not in (None, ctypes.c_void_p(-1).value),
            "owned CUDA process snapshot unavailable")
    parents = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        require(kernel.Process32FirstW(snapshot, ctypes.byref(entry)) != 0,
                "owned CUDA process snapshot empty")
        while True:
            pid = int(entry.th32ProcessID)
            require(pid not in parents, "owned CUDA process snapshot duplicated a PID")
            parents[pid] = int(entry.th32ParentProcessID)
            entry.dwSize = ctypes.sizeof(ProcessEntry)
            if kernel.Process32NextW(snapshot, ctypes.byref(entry)) == 0:
                break
    finally:
        kernel.CloseHandle(snapshot)
    candidates = {root_pid}
    for _ in range(len(parents)):
        added = {pid for pid, parent in parents.items() if parent in candidates}
        if added <= candidates:
            break
        candidates |= added
    births = {}
    for pid in candidates & parents.keys():
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        require(handle, "owned CUDA process creation handle unavailable")
        try:
            births[pid] = windows_process_birth(handle)
        finally:
            kernel.CloseHandle(handle)
    return descendant_generations(parents, births, root_pid, root_birth)


def owned_descendants_released(root_pid: int, root_birth: int,
                               observed: dict[int, int]) -> bool:
    current = windows_owned_descendants(root_pid, root_birth)
    # A child spawned in the last instant before root exit must also be seen.
    if current:
        return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    for pid, birth in observed.items():
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                if windows_process_birth(handle) == birth:
                    return False
            finally:
                kernel.CloseHandle(handle)
        else:
            require(ctypes.get_last_error() == 87, "observed owned CUDA descendant is inaccessible")
    return True


def owned_gpu_pid_released(raw: str, pid: int) -> bool:
    probe = json.loads(raw)
    if probe.get("physicalMode") != "shared-gpu1":
        return True
    actors = probe.get("validatedDevice", {}).get("observedActors")
    require(isinstance(actors, list) and len(actors) == 4 and
            all(isinstance(epoch, list) for epoch in actors),
            "selected GPU1 postflight actor inventory unavailable")
    return all(actor.get("pid") != pid for epoch in actors for actor in epoch)


def same_selected_cuda_device(initial: str, current: str) -> bool:
    first, later = json.loads(initial), json.loads(current)
    if first.get("physicalMode") != "shared-gpu1":
        return True
    require(later.get("physicalMode") == "shared-gpu1", "selected GPU1 proof mode changed")
    keys = ("uuid", "pci", "luid", "physicalIndex", "cudaOrdinal")
    before = first.get("validatedDevice", {})
    after = later.get("validatedDevice", {})
    return all(before.get(key) == after.get(key) for key in keys)


def run_test_child(binary: Path, name: str, label: str, backend: str, env: dict,
                   evidence: Path, total_deadline: float | None, owner_guard,
                   identity: dict) -> tuple[dict, list[dict], list[str]]:
    """Run and reap one owned test before another child may enter the device."""
    command = [str(binary), "--ignored", "--exact", name, "--nocapture", "--test-threads", "1"]
    samples: list[dict] = []
    faults: list[str] = []
    stop = threading.Event()
    track_tree = backend == "cuda" and env.get("CUDA_VISIBLE_DEVICES") == "1" and os.name == "nt"
    root_birth = None
    descendants: dict[int, int] = {}
    descendants_released = not track_tree
    started = time.time_ns()
    log_path = evidence / ("test.log" if label == "precision" else f"{label}-smoke.log")
    with log_path.open("w", encoding="utf-8") as log:
        if owner_guard is not None:
            owner_guard.arm()
            if owner_guard.failed.is_set():
                owner_guard.finish()  # Restores signal handlers even before Popen.
                raise RuntimeError("owner canceled before test child creation")
        if total_deadline is not None and time.monotonic() >= total_deadline:
            if owner_guard is not None:
                owner_guard.finish()
            raise RuntimeError("combined CUDA child deadline expired before Popen")
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        except BaseException:
            if owner_guard is not None:
                owner_guard.finish()
            raise
        if track_tree:
            try:
                root_birth = windows_process_birth(int(child._handle))
                descendants.update(windows_owned_descendants(child.pid, root_birth))
            except BaseException:
                reap_owned_child(child, owned_tree=True)
                raise
        def loop() -> None:
            while not stop.is_set() and child.poll() is None:
                try:
                    if track_tree:
                        descendants.update(windows_owned_descendants(child.pid, root_birth))
                    samples.append(sample_cuda() if backend == "cuda" else sample_metal(child.pid))
                except Exception as error:
                    if child.poll() is None:
                        faults.append(f"{time.time_ns()}: {error}")
                stop.wait(0.25 if backend == "cuda" else 1.0)
        thread = threading.Thread(target=loop, daemon=True)
        try:
            if owner_guard is not None:
                owner_guard.start(child)
            # Sample synchronously while the owned PID still exists. A fast
            # smoke that exits before a valid sample is a refusal, not a zero.
            if child.poll() is None:
                try:
                    first = sample_cuda() if backend == "cuda" else sample_metal(child.pid)
                    if child.poll() is None:
                        samples.append(first)
                except Exception as error:
                    if child.poll() is None:
                        faults.append(f"{time.time_ns()}: {error}")
            thread.start()
            wait_budget = max(0.0, total_deadline - time.monotonic()) if total_deadline is not None else None
            if owner_guard is not None:
                from yue2_gpu0_owner_guard import wait
                code, timed_out, wait_error = wait(child, owner_guard, wait_budget or 0)
            else:
                code, timed_out, wait_error = wait_owned_child(
                    child, backend, wait_budget, owned_tree=track_tree)
        except BaseException as error:
            if owner_guard is None and not isinstance(error, Exception):
                raise
            if owner_guard is not None:
                from yue2_gpu0_owner_guard import reap_tree
                code, cleanup_error = reap_tree(child)
            else:
                code, cleanup_error = reap_owned_child(child, owned_tree=track_tree)
            timed_out = False
            wait_error = f"sampler startup: {error}; cleanup: {cleanup_error}"
        finally:
            stop.set()
            if thread.is_alive():
                thread.join(timeout=25)
                if thread.is_alive():
                    faults.append("external sampler thread did not release")
            if track_tree and root_birth is not None and not thread.is_alive():
                try:
                    descendants_released = owned_descendants_released(child.pid, root_birth, descendants)
                except Exception as error:
                    faults.append(f"owned CUDA descendant release proof: {error}")
    if owner_guard is not None:
        try:
            owner_guard.finish()
        except BaseException as error:
            wait_error = f"{wait_error}; final holder: {error}"
    ended = time.time_ns()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    binary_digest = sha256(binary)
    result = {"label": label, "name": name, "binary_sha256": binary_digest, "build_identity": identity,
              "binary_unchanged_after_child": binary_digest == identity.get("binary_sha256"),
              "command": command, "log": log_path.name, "pid": child.pid,
              "started_utc_ns": started, "ended_utc_ns": ended, "exit_code": code,
              "timed_out": timed_out, "wait_error": wait_error,
              "released": child.poll() is not None,
              "owned_descendants": [{"pid": pid, "creation_filetime": birth}
                                    for pid, birth in sorted(descendants.items())],
              "owned_descendants_released": descendants_released,
              "exact_one_test_passed": exact_one_test_executed(output, name),
              "sample_count": len(samples), "sampler_faults": faults,
              "scheduling": owner_guard.summary() if owner_guard is not None else
              {"mode": env.get("YUE2_CUDA_SCHEDULING_MODE", "shared-host")}}
    return result, samples, faults


def stage_markers(output: str) -> list[dict]:
    prefix = "YUE2_PRECISION_STAGE "
    markers = []
    for line in output.splitlines():
        if prefix in line:
            markers.append(json.loads(line.split(prefix, 1)[1]))
    return markers


def missing_stage_markers(markers: list[dict]) -> list[str]:
    observed = {(row.get("stage"), row.get("event")) for row in markers}
    required = []
    for policy in ("Bf16", "Auto", "Fp32", "Legacy"):
        for stage in ("registered_load", "registered_generation", "cached_legacy_decode"):
            required.append(f"{policy}:{stage}")
    for policy in ("Bf16", "Auto", "Fp32"):
        for variant in ("standard", "legacy"):
            for stage in ("vae_load", "long:full_decode", "long:tiled_decode",
                          "production:full_decode", "production:tiled_decode", "encoder"):
                required.append(f"{policy}:{variant}:{stage}")
    required.extend(("Bf16:cross_policy_load", "Bf16:cross_policy_cached_decode"))
    return [f"{stage}:{event}" for stage in required for event in ("start", "end")
            if (stage, event) not in observed]


def stage_sample_coverage(markers: list[dict], samples: list[dict]) -> dict:
    starts = {m['stage']: m['unixMs'] * 1_000_000 for m in markers if m.get('event') == 'start'}
    coverage = {}
    for marker in markers:
        if marker.get('event') != 'end' or marker['stage'] not in starts:
            continue
        lo, hi = starts[marker['stage']], marker['unixMs'] * 1_000_000
        coverage[marker['stage']] = {
            'fully_contained_samples': sum(lo <= s['started_utc_ns'] and s['ended_utc_ns'] <= hi for s in samples),
            'overlapping_samples': sum(s['started_utc_ns'] <= hi and s['ended_utc_ns'] >= lo for s in samples),
        }
    return coverage


def validate_receipt(receipt: Path, backend: str, work_dir: Path | None = None) -> None:
    value = json.loads(receipt.read_text(encoding="utf-8"))
    require(value.get("schemaVersion") == 1 and value.get("backend") == backend,
            "precision receipt schema/backend mismatch")
    require(value.get("referenceSha256") == REFERENCE_SHA256,
            "precision receipt lost the independently pinned reference")
    if work_dir is not None:
        require(value.get("listeningDir") == str(work_dir),
                "precision receipt lost the retained runner-local listening path")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 3 and
            {case.get("requestedPolicy") for case in cases} == {"Bf16", "Auto", "Fp32"},
            "precision receipt must contain all three policies")
    run_ids, cache_ids = set(), set()
    for case in cases:
        policy = case["requestedPolicy"]
        want_vae = "bfloat16" if policy == "Bf16" else "float32"
        want_model = "float32" if policy == "Fp32" else "bfloat16"
        dtypes = case.get("effectiveDtypes", {})
        require(dtypes == {"ar": want_model, "nar": want_model,
                           "vaeDecoder": want_vae, "vaeEncoder": want_vae},
                f"{policy} effective dtype report is incomplete or substituted")
        generation = case.get("generation", {})
        config = generation.get("config", {})
        require(config.get("compute_policy") == policy.lower() and config.get("vae_dtype") == want_vae,
                f"{policy} generation config disagrees with requested policy")
        run_id, cache_id = generation.get("runIdentity"), generation.get("legacyCacheIdentity")
        require(isinstance(run_id, str) and run_id and isinstance(cache_id, str) and cache_id,
                f"{policy} run/cache identity missing")
        run_ids.add(run_id); cache_ids.add(cache_id)
        decoders = case.get("decoders")
        require(isinstance(decoders, list) and len(decoders) == 2 and
                {decoder.get("variant") for decoder in decoders} == set(DECODER_SHA256),
                f"{policy} did not run standard and legacy VAEs")
        for decoder in decoders:
            variant = decoder["variant"]
            dtype = "BF16" if policy == "Bf16" else "F32"
            require(decoder.get("weightsSha256") == DECODER_SHA256[variant] and
                    decoder.get("parameterDtype") == dtype and decoder.get("activationDtype") == dtype,
                    f"{policy}/{variant} did not report pinned resident/activation dtype")
            rows = decoder.get("decodeCases")
            require(isinstance(rows, list) and len(rows) == 2 and
                    {row.get("name") for row in rows} == {"long", "production"} and
                    decoder.get("encoderMean") is not None and decoder.get("encoderScale") is not None,
                    f"{policy}/{variant} decode/encoder coverage absent")
    require(len(run_ids) == len(cache_ids) == 3, "cross-policy waveform identities collided")
    cross = value.get("legacyToBf16CachedDecode", {})
    source, target = cross.get("sourceGeneration", {}), cross.get("targetConfig", {})
    require(source.get("identity") == cross.get("sourceIdentity") and
            isinstance(source.get("config"), dict) and "compute_policy" not in source["config"] and
            target.get("compute_policy") == "bf16" and target.get("vae_dtype") == "bfloat16" and
            target.get("decoder_release") == "legacy" and
            target.get("cached_decode", {}).get("source_identity") == cross.get("sourceIdentity") and
            cross.get("targetIdentity") != cross.get("sourceIdentity"),
            "historical Legacy latents were not freshly decoded with strict BF16 provenance")


def execute(args: argparse.Namespace) -> None:
    evidence = args.evidence
    evidence.mkdir(parents=True, exist_ok=True)
    reference = args.reference / "vae_real_reference.safetensors"
    require(reference.is_file() and sha256(reference) == REFERENCE_SHA256, "pinned external reference not verified")
    require(args.binary.is_file(), f"test binary missing: {args.binary}")
    smoke_binaries = (getattr(args, "quant_smoke_binary", None), getattr(args, "vae_smoke_binary", None))
    if args.backend == "cuda":
        require(all(path is not None and path.is_file() for path in smoke_binaries),
                "both exact CUDA smoke binaries are required before the real-weight test")
    require(not args.work_dir.exists(), "refuse to reuse an earlier precision listening directory")
    require(args.work_dir.parent.is_dir(), "persistent listening parent is unavailable")
    verify_revisions(args.engine_sha, args.control_sha)
    require(not args.app_sha or re.fullmatch(r"[0-9a-f]{40}", args.app_sha) is not None,
            "optional caller app SHA must be full lowercase hex")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                           capture_output=True, text=True, check=True, encoding="utf-8").stdout
    require(not dirty.strip(), "engine source became dirty before hardware execution")
    runner = os.environ.get("RUNNER_NAME", "")
    baseline_files = None
    owner_guard = None
    scheduling = getattr(args, "cuda_scheduling_mode", "shared-host")
    require(scheduling == "shared-host" or args.backend == "cuda", "GPU0 owner mode is CUDA only")
    if args.backend == "metal":
        require(runner == "nax-macos-2", f"Metal proof assigned to wrong runner: {runner}")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
                "CUDA proof must bind logical CUDA0 to the owner's physical GPU1")
        from yue2_cuda_idle_context import check_shared_gpu1_dispatch, require_remaining_window  # type: ignore[import-not-found]
        job_start = os.environ.get("YUE2_PRECISION_JOB_STARTED_UTC_NS", "")
        require(job_start.isdigit() and int(job_start) <= time.time_ns() and
                time.time_ns() + (CUDA_CHILD_TIMEOUT_SECONDS + CUDA_POSTFLIGHT_SECONDS) * 1_000_000_000 <=
                int(job_start) + CUDA_JOB_TIMEOUT_SECONDS * 1_000_000_000,
                "bounded CUDA child cannot finish before workflow upload tail")
        baseline = None
        check_shared_gpu1_dispatch()
    total_deadline = time.monotonic() + CUDA_CHILD_TIMEOUT_SECONDS if args.backend == "cuda" else None
    before_raw, before_busy = cuda_physical_census() if args.backend == "cuda" else metal_census()
    (evidence / "census-before.txt").write_text(before_raw, encoding="utf-8")
    require(not before_busy, physical_busy_message(before_raw, before_busy, "foreign/lingering accelerator executables before test"))
    before_files = retain_cuda_physical_evidence(evidence, "before", before_raw) if args.backend == "cuda" else None
    env = {k: v for k, v in os.environ.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    env["YUE2_VAE_REFERENCE_DIR"] = str(args.reference)
    env["YUE2_PRECISION_RECEIPT"] = str(evidence / "precision-receipt.json")
    env["YUE2_PRECISION_WORK_DIR"] = str(args.work_dir)
    started = time.time_ns()
    samples: list[dict] = []
    faults: list[str] = []
    child_results = []
    handoff_files = []
    stage_refusals = []
    stages = ([(label, name, path) for (label, name), path in zip(CUDA_SMOKES, smoke_binaries)]
              if args.backend == "cuda" else []) + [("precision", TEST_NAME, args.binary)]
    for label, name, binary in stages:
        try:
            if total_deadline is not None:
                remaining_cuda_budget(total_deadline, int(job_start))
            identity = verify_binary_identity(binary, label, evidence)
            if scheduling in {"owner-gpu0", "owner-gpu0-mac-anchor", "gpu0-with-reviewed-gpu1"}:
                from yue2_gpu0_owner_guard import OwnerGuard
                owner_guard = OwnerGuard(evidence, args.engine_sha, args.control_sha, mode=scheduling)
                owner_guard.preflight()
            if args.backend == "cuda":
                # The preceding child's tree and watchdog have released; the
                # fresh physical census is the last action before Popen.
                handoff_raw, handoff_busy = cuda_physical_census()
                (evidence / f"census-pre-{label}.txt").write_text(handoff_raw, encoding="utf-8")
                require(same_selected_cuda_device(before_raw, handoff_raw),
                        "selected GPU1 adapter/LUID changed before owned child")
                require(not handoff_busy, physical_busy_message(
                    handoff_raw, handoff_busy, f"foreign process at {label} handoff"))
                handoff_files.append(retain_cuda_physical_evidence(evidence, f"pre-{label}", handoff_raw))
            result, stage_samples, stage_faults = run_test_child(
                binary, name, label, args.backend, env, evidence, total_deadline, owner_guard, identity)
        except Exception as error:
            stage_refusals.append({"label": label, "name": name, "error": str(error),
                                   "observed_utc_ns": time.time_ns()})
            break
        child_results.append(result)
        samples.extend(stage_samples)
        faults.extend(stage_faults)
        if args.backend == "cuda":
            try:
                release_raw, release_busy = cuda_physical_census(admission=False)
                (evidence / f"census-post-{label}.txt").write_text(release_raw, encoding="utf-8")
                require(same_selected_cuda_device(before_raw, release_raw),
                        "selected GPU1 adapter/LUID changed after owned child")
                result["post_census_files"] = retain_cuda_physical_evidence(
                    evidence, f"post-{label}", release_raw) if not release_busy else None
                result["owned_gpu_pid_released"] = (owned_gpu_pid_released(release_raw, result["pid"])
                                                    if not release_busy else False)
                result["post_census_busy"] = release_busy
                result["post_census_error"] = None
            except Exception as error:
                result["post_census_error"] = str(error)
                result["post_census_busy"] = []
        # A failed or ambiguous smoke never advances to the next child.
        if not child_stage_succeeded(result):
            break
    ended = time.time_ns()
    last_child = child_results[-1] if child_results else None
    code = last_child["exit_code"] if last_child else None
    timed_out = bool(last_child and last_child["timed_out"])
    wait_error = last_child["wait_error"] if last_child else "no owned test child launched"
    post_census_error = None
    try:
        after_raw, after_busy = cuda_physical_census(admission=False) if args.backend == "cuda" else metal_census()
        if args.backend == "cuda":
            require(same_selected_cuda_device(before_raw, after_raw),
                    "selected GPU1 adapter/LUID changed at final postflight")
    except Exception as error:
        after_raw, after_busy = "", []
        post_census_error = str(error)
    (evidence / "census-after.txt").write_text(after_raw, encoding="utf-8")
    after_files = None
    if args.backend == "cuda" and post_census_error is None and not after_busy:
        try:
            after_files = retain_cuda_physical_evidence(evidence, "after", after_raw)
        except Exception as error:
            post_census_error = str(error)
    write_json(evidence / "external-samples.json", {"backend": args.backend, "samples": samples, "faults": faults})
    output = (evidence / "test.log").read_text(encoding="utf-8", errors="replace") if (evidence / "test.log").is_file() else ""
    markers = stage_markers(output)
    missing_markers = missing_stage_markers(markers)
    coverage = stage_sample_coverage(markers, samples)
    (evidence / "stage-markers.jsonl").write_text(
        "".join(json.dumps(marker, sort_keys=True) + "\n" for marker in markers), encoding="utf-8"
    )
    peak = None
    if args.backend == "cuda":
        for sample in samples:
            for row in sample["raw"].splitlines():
                fields = [value.strip() for value in row.split(",")]
                if len(fields) >= 4 and fields[1] == ("1" if os.environ.get("CUDA_VISIBLE_DEVICES") == "1" else "0") and fields[2].isdigit():
                    peak = max(peak or 0, int(fields[2]))
    else:
        peak = max((sample["phys_footprint_bytes"] for sample in samples), default=None)
    receipt = evidence / "precision-receipt.json"
    receipt_schema_error = None
    if receipt.is_file():
        try:
            validate_receipt(receipt, args.backend, args.work_dir)
        except Exception as error:
            receipt_schema_error = str(error)
    local_audio = []
    if args.work_dir.is_dir():
        for wav in sorted(args.work_dir.rglob("*.wav")):
            local_audio.append({"path": str(wav), "sha256": sha256(wav), "bytes": wav.stat().st_size})
    report = {"schema": "yue2-precision-control-v1", "backend": args.backend,
              "engine_sha": args.engine_sha, "control_sha": args.control_sha, "caller_app_sha": args.app_sha,
              "runner_name": runner, "binary_sha256": sha256(args.binary),
              "reference_sha256": sha256(reference), "started_utc_ns": started,
              "persistent_listening_dir": str(args.work_dir), "local_audio": local_audio,
              "ended_utc_ns": ended, "test_exit_code": code, "sample_count": len(samples),
              "stage_marker_count": len(markers), "external_peak": peak,
              "missing_stage_markers": missing_markers,
              "stage_sample_coverage": coverage,
              "external_peak_unit": "MiB selected physical device used" if args.backend == "cuda" else "bytes owned phys_footprint",
              "external_peak_scope": "selected-device-global-capacity" if args.backend == "cuda" else "owned-process",
              "owned_cuda_peak_unavailable": args.backend == "cuda",
              "sampler_faults": faults, "owned_test_pid": last_child["pid"] if last_child else None,
              "owned_test_released": bool(last_child and last_child["released"]),
              "owned_test_timed_out": timed_out, "owned_test_wait_error": wait_error,
              "cuda_child_timeout_seconds": CUDA_CHILD_TIMEOUT_SECONDS if args.backend == "cuda" else None,
              "reviewed_baseline": baseline if args.backend == "cuda" else None,
              "reviewed_baseline_files": baseline_files,
              "fresh_physical_before_files": before_files, "fresh_physical_after_files": after_files,
              "handoff_files": handoff_files, "owned_children": child_results,
              "stage_refusals": stage_refusals,
              "combined_cuda_child_timeout_seconds": CUDA_CHILD_TIMEOUT_SECONDS if args.backend == "cuda" else None,
              "post_census_busy": after_busy, "post_census_error": post_census_error,
              "receipt_sha256": sha256(receipt) if receipt.is_file() else None,
              "receipt_schema_error": receipt_schema_error,
              "scheduling": owner_guard.summary() if owner_guard is not None else {"mode": scheduling}}
    write_json(evidence / "control.json", report)
    print(json.dumps(report, indent=2), flush=True)
    require(len(child_results) == len(stages) and all(child_stage_succeeded(row) for row in child_results),
            f"owned exact-test sequence failed, timed out, or incomplete: {child_results}; "
            f"stage refusals: {stage_refusals}; see per-child logs")
    require(not timed_out and wait_error is None and code == 0,
            f"precision test timed out or exited {code}; wait error: {wait_error}; see test.log")
    require(one_test_executed(output), "one exact ignored test did not execute")
    require(receipt.is_file(), "precision test did not produce its receipt")
    require(receipt_schema_error is None, f"precision receipt contract mismatch: {receipt_schema_error}")
    require(local_audio, "precision test completed without retained runner-local WAVs")
    require(markers, "precision test emitted no stage boundaries for sampler attribution")
    require(not missing_markers, f"precision stage markers incomplete: {missing_markers}")
    require(samples and not faults, "external sampler had no valid sample or suffered a fault")
    require(post_census_error is None, f"post-test release census failed: {post_census_error}")
    require(last_child is not None and last_child["released"] and not after_busy,
            physical_busy_message(after_raw, after_busy, "owned test/process cleanup uncertain"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("resolve-binary")
    p.add_argument("--build-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", choices=tuple(BUILD_TARGETS), default="precision")
    p = sub.add_parser("verify-reference")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--engine-sha", required=True)
    p.add_argument("--control-sha", required=True)
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("run")
    p.add_argument("--backend", choices=("cuda", "metal"), required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--quant-smoke-binary", type=Path)
    p.add_argument("--vae-smoke-binary", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--engine-sha", required=True)
    p.add_argument("--control-sha", required=True)
    p.add_argument("--app-sha", default="")
    p.add_argument("--cuda-scheduling-mode", choices=("shared-host", "shared-gpu1", "owner-gpu0", "owner-gpu0-mac-anchor",
                                                      "gpu0-with-reviewed-gpu1"), default="shared-host")
    args = parser.parse_args()
    {"resolve-binary": resolve_binary, "verify-reference": verify_reference, "run": execute}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"::error::{error}", file=sys.stderr)
        sys.exit(1)
