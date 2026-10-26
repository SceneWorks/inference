#!/usr/bin/env python3
"""Manual, one-route CPU prequantization on the primary real-weights Mac.

This controller parses safetensors headers and streams source/output bytes only
for SHA-256; it never loads tensors. The existing Rust examples own conversion
and the Metal trap. Outputs stay on the runner's model volume; Actions uploads
only the receipt.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

_guard_spec = importlib.util.spec_from_file_location(
    "ios_pricing_compile_only", Path(__file__).with_name("ios_pricing_compile_only.py"))
_guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(_guard)
descendants, process_rows = _guard.descendants, _guard.process_rows

GIB = 1024**3
MIB = 1024**2
RSS_LIMIT = 96 * GIB
BUILD_SECONDS = 7200
CONVERT_SECONDS = 6 * 3600
PINS = {
    "sana-1600m": ("SceneWorks", "Sana_1600M_1024px_mlx", "ac421696dd6eb0b41f446f4c45a53ccc057d82a1", "bf16"),
    "sana-sprint": ("SceneWorks", "Sana_Sprint_1.6B_1024px_mlx", "0b0d18484cac2fb515e76d25a09a5911ae4ab58e", "bf16"),
    "iris-generation": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", ""),
    "iris-restoration": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", "upscaler"),
    "iris-depth": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", "depth"),
    "iris-text-encoder": ("Qwen", "Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", ""),
}
ROUTES = frozenset(f"{name}-q{bits}" for name in PINS for bits in (4, 8))


def parse_binding(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"source_sha", "route"}:
        raise ValueError("binding requires only source_sha and route")
    if not isinstance(value["source_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["source_sha"]):
        raise ValueError("source_sha must be an exact lower-case commit")
    if value["route"] not in ROUTES:
        raise ValueError("route must be one supported model/task and q4/q8 tier")
    return value


def source_for(route, cache_root):
    name = route.rsplit("-q", 1)[0]
    owner, repo, revision, suffix = PINS[name]
    source = cache_root / f"models--{owner}--{repo}" / "snapshots" / revision
    if suffix:
        source /= suffix
    return name, f"{owner}/{repo}@{revision}", source


def header_rows(path):
    """Bound each header read; payload checksums are streamed only after admission."""
    files = sorted(path.rglob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"missing safetensors at {path}")
    rows = []
    for file in files:
        with file.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size <= 100 * MIB:
                raise ValueError(f"invalid safetensors header size at {file}")
            header = json.loads(stream.read(size))
        for key, value in header.items():
            if key != "__metadata__":
                start, end = value["data_offsets"]
                if not 0 <= start < end <= file.stat().st_size - size - 8:
                    raise ValueError(f"invalid tensor offset at {file}: {key}")
                rows.append({"file": str(file), "tensor": key, "bytes": end - start,
                             "dtype": value["dtype"], "shape": value["shape"]})
    if not rows:
        raise ValueError("no tensor headers")
    return rows


def input_headers(name, source):
    if name.startswith("sana-"):
        paths = [source / "transformer", source / "text_encoder"]
    elif name == "iris-text-encoder":
        paths = [source]
    else:
        paths = [source / "model.safetensors"]
    return [row for path in paths for row in header_rows(path)]


def tensor_budget_mib(name, headers):
    # Deliberately conservative. The Rust CLI recomputes its exact envelope and
    # rejects before load if this bound is insufficient. SANA releases components
    # separately; using their aggregate here costs no allocation.
    total = sum(row["bytes"] for row in headers)
    largest = max(row["bytes"] for row in headers)
    bound = 2 * total + 3 * largest
    if bound > RSS_LIMIT:
        raise ValueError(f"header-derived tensor bound {bound} exceeds RSS ceiling")
    return (bound + MIB - 1) // MIB, bound


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipts(root):
    return [{"path": str(file.relative_to(root)), "bytes": file.stat().st_size, "sha256": sha(file)}
            for file in sorted(root.rglob("*")) if file.is_file()]


def converter_arguments(name, source, destination, bits, budget):
    common = [str(source), str(destination), str(bits), str(budget)]
    return [name.removeprefix("iris-"), *common] if name.startswith("iris-") else common


def output_paths(cache, run_id, attempt, binding):
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("invalid Actions run identity")
    volume = cache.resolve().parent.parent
    output_root = volume / "sceneworks-ios-prequant" / f"{run_id}-{attempt}"
    target = volume / "sceneworks-ios-prequant-build" / binding["source_sha"]
    return output_root, output_root / binding["route"], target


def run_guarded(argv, env, log, seconds):
    """Only signal this command's process group, never another runner job."""
    started = time.monotonic()
    peak = 0
    samples = 0
    reason = None
    with log.open("wb") as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        while child.poll() is None:
            try:
                rows = process_rows()
                rss = sum(rows[pid][1] for pid in descendants(rows, child.pid) if pid in rows)
                peak = max(peak, rss)
                samples += 1
                if rss > RSS_LIMIT:
                    reason = "RSS ceiling exceeded"
            except Exception as error:
                reason = f"RSS monitoring failed: {error!r}"
            if time.monotonic() - started > seconds:
                reason = "wall limit exceeded"
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        exit_code = child.wait()
    result = {"argv": argv, "exit": exit_code, "seconds": round(time.monotonic() - started, 2),
              "max_sampled_tree_rss_bytes": peak, "rss_samples": samples, "guard_stop": reason,
              "log_sha256": sha(log)}
    if reason or exit_code != 0 or samples == 0:
        raise RuntimeError(f"command failed: {result}; tail={log.read_text(errors='replace')[-2000:]}")
    return result


def main():
    receipt_path = Path(os.environ["RUNNER_TEMP"]) / "ios-prequant-receipt.json"
    receipt = {"state": "STARTING", "complete": False, "device": "CPU", "rss_limit_bytes": RSS_LIMIT,
               "metal_trap_required": True, "outputs_uploaded": False}
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    save()
    try:
        binding = parse_binding(os.environ["IOS_PREQUANT_BINDING"])
        receipt["binding"] = binding
        if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("RUNNER_NAME") != "nax-macos":
            raise ValueError("requires manual dispatch on primary nax-macos")
        if os.environ.get("GITHUB_SHA") != binding["source_sha"]:
            raise ValueError("dispatch ref differs from frozen source")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("requires native Apple silicon")
        repo = Path(__file__).resolve().parents[2]
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != binding["source_sha"]:
            raise ValueError("checkout differs from frozen source")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True).strip():
            raise ValueError("checkout is dirty")
        cache = Path.home() / ".cache/huggingface/hub"
        if not cache.is_dir():
            raise ValueError("provisioned Hugging Face cache missing")
        name, source_revision, source = source_for(binding["route"], cache)
        if not source.is_dir():
            raise ValueError(f"pinned source missing: {source}")
        receipt.update(source_revision=source_revision, source=str(source),
                       cache_realpath=str(cache.resolve()))
        headers = input_headers(name, source)
        budget, bound = tensor_budget_mib(name, headers)
        receipt.update(source_tensor_count=len(headers), source_payload_bytes=sum(x["bytes"] for x in headers),
                       source_max_tensor_bytes=max(x["bytes"] for x in headers),
                       header_tensor_bound_bytes=bound, tensor_budget_mib=budget)
        selected_files = {Path(row["file"]) for row in headers}
        if name.startswith("sana-"):
            selected_files.update(path for path in source.rglob("*") if path.is_file())
        selected_files.update(path for path in source.iterdir() if path.is_file()
                              and path.suffix != ".safetensors")
        selected_files.update(path for path in (source / "empty_prompt.safetensors",)
                              if path.is_file())
        selected_files = sorted(selected_files)
        receipt["source_files"] = [{"path": str(path), "bytes": path.stat().st_size,
                                    "sha256": sha(path)} for path in selected_files]
        # One route per dispatch. No existing snapshot or output can be overwritten.
        run_id = os.environ["GITHUB_RUN_ID"]
        attempt = os.environ["GITHUB_RUN_ATTEMPT"]
        output_root, destination, target = output_paths(cache, run_id, attempt, binding)
        if output_root.exists():
            raise ValueError(f"run output already exists: {output_root}")
        free = shutil.disk_usage(cache.resolve().parent.parent).free
        reserve = 2 * sum(x["bytes"] for x in headers) + 48 * GIB
        receipt.update(output=str(destination), free_bytes=free, reserved_free_bytes=reserve)
        if free < reserve:
            raise ValueError("insufficient disk for source and new output")
        output_root.mkdir(parents=True)
        cargo = Path.home() / ".cargo/bin/cargo"
        if not cargo.is_file():
            raise ValueError("preinstalled Cargo missing")
        native = Path(os.environ["PMETAL_MLX_PREBUILT_DIR"])
        if not native.is_absolute() or not (native / "pmetal-mlx-prebuilt.txt").is_file():
            raise ValueError("matching prebuilt MLX native artifact required")
        if Path(os.environ["PMETAL_METALLIB_PATH"]) != native / "mlx.metallib" or not (native / "mlx.metallib").is_file():
            raise ValueError("matching prebuilt metallib required")
        sdk = Path(os.environ["SDKROOT"])
        if sdk.name != "MacOSX26.5.sdk" or not sdk.is_dir():
            raise ValueError("SDK26.5 required")
        linker = Path.home() / ".rustup/toolchains/1.96.0-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/rust-lld"
        if not linker.is_file():
            raise ValueError("Rust1.96 native linker required")
        # The workflow's physical-host concurrency serializes dispatches; a source-keyed
        # target reuses the expensive native build across selected route waves.
        target.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", CARGO_INCREMENTAL="0",
                   CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
                   CARGO_PROFILE_DEV_STRIP="none", CARGO_PROFILE_TEST_STRIP="none",
                   CARGO_NET_OFFLINE="true", RUSTC_WRAPPER="", MACOSX_DEPLOYMENT_TARGET="26.2",
                   RUSTFLAGS=f"-C linker-flavor=ld64.lld -C linker={linker}")
        package, example = (("mlx-gen-sana", "sana_prequant_cpu") if name.startswith("sana-")
                            else ("mlx-gen-iris", "iris_prequant_cpu"))
        receipt["cli"] = {"package": package, "example": example, "rust": "1.96.0",
                          "jobs": 4, "incremental": False, "debug": 0, "prebuilt": str(native),
                          "sdk": str(sdk), "linker": str(linker)}
        save()
        receipt["build"] = run_guarded([str(cargo), "+1.96.0", "build", "--locked", "--offline",
                                        "-p", package, "--example", example], env,
                                       output_root / "build.log", BUILD_SECONDS)
        binary = target / "debug/examples" / example
        if not binary.is_file():
            raise ValueError("compiled converter missing")
        startup = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        if startup.returncode == 0 or "usage:" not in startup.stderr:
            raise ValueError("converter startup did not reject missing arguments")
        receipt["startup"] = {"exit": startup.returncode, "expected_usage_refusal": True,
                              "binary_sha256": sha(binary)}
        save()
        bits = int(binding["route"].rsplit("-q", 1)[1])
        arguments = converter_arguments(name, source, destination, bits, budget)
        receipt["conversion"] = run_guarded([str(binary), *arguments], env,
                                             output_root / "convert.log", CONVERT_SECONDS)
        with (output_root / "convert.log").open("rb") as log:
            log_start = log.read(4000).decode(errors="replace")
        if "Metal submission trap armed: true" not in log_start:
            raise ValueError("converter did not confirm armed Metal trap")
        if not destination.is_dir():
            raise ValueError("converter produced no output")
        receipt["output_headers"] = header_rows(destination)
        receipt["output_files"] = file_receipts(destination)
        if receipt["source_files"] != [{"path": str(path), "bytes": path.stat().st_size,
                                        "sha256": sha(path)} for path in selected_files]:
            raise ValueError("pinned source bytes changed during conversion")
        receipt["metal_trap_armed"] = True
        receipt["complete"] = True
        receipt["state"] = "CONVERTED_LOCAL_OUTPUT_PENDING_REVIEW"
        save()
        return 0
    except Exception as error:
        receipt["state"] = "FAILED"
        receipt["error"] = repr(error)
        save()
        print(f"::error::{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
