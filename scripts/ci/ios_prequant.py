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
import tomllib

_guard_spec = importlib.util.spec_from_file_location(
    "ios_pricing_compile_only", Path(__file__).with_name("ios_pricing_compile_only.py"))
_guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(_guard)
descendants, process_rows = _guard.descendants, _guard.process_rows
_hub_spec = importlib.util.spec_from_file_location(
    "pinned_hub_source", Path(__file__).with_name("pinned_hub_source.py"))
_hub = importlib.util.module_from_spec(_hub_spec)
_hub_spec.loader.exec_module(_hub)

GIB = 1024**3
MIB = 1024**2
BUILD_SECONDS = 7200
CONVERT_SECONDS = 6 * 3600
MAX_LOG_BYTES = 64 * MIB
DISK_FLOOR_BYTES = 8 * GIB
MLX_REVISION = "355e2ae4e81c13032233a4244e956bc3547f49e4"
MLX_FINGERPRINT = "076ae47b575ed5c1c36a8cd67e5e23fc423b710157d725ef6d3e8d347a3b7aff"
PINS = {
    "anima-turbo": ("circlestone-labs", "Anima", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b", "split_files"),
    "sana-1600m": ("SceneWorks", "Sana_1600M_1024px_mlx", "ac421696dd6eb0b41f446f4c45a53ccc057d82a1", "bf16"),
    "sana-sprint": ("SceneWorks", "Sana_Sprint_1.6B_1024px_mlx", "0b0d18484cac2fb515e76d25a09a5911ae4ab58e", "bf16"),
    "iris-generation": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", ""),
    "iris-restoration": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", "upscaler"),
    "iris-depth": ("speridlabs", "iris-3b", "7445443349bc9abe3c96f01ff793e2098ca012b3", "depth"),
    "iris-text-encoder": ("Qwen", "Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", ""),
}
ROUTES = frozenset(f"{name}-q{bits}" for name in PINS if name != "anima-turbo" for bits in (4, 8)) | {"anima-turbo"}


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
    name = route if route == "anima-turbo" else route.rsplit("-q", 1)[0]
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
    if name == "anima-turbo":
        paths = [source / "diffusion_models/anima-turbo-v1.0.safetensors",
                 source / "text_encoders/qwen_3_06b_base.safetensors",
                 source / "vae/qwen_image_vae.safetensors"]
    elif name.startswith("sana-"):
        paths = [source / "transformer", source / "text_encoder"]
    elif name == "iris-text-encoder":
        paths = [source]
    else:
        paths = [source / "model.safetensors"]
    return [row for path in paths for row in header_rows(path)]


def tensor_budget_mib(name, headers):
    # SANA releases transformer before loading text_encoder; the Rust exporter
    # stages two copies of one component plus its largest tensor. Iris stages a
    # packed output plus three copies of its largest source tensor. Bound its
    # packed output by twice all source payload bytes, including shard padding.
    total = sum(row["bytes"] for row in headers)
    largest = max(row["bytes"] for row in headers)
    if name == "anima-turbo":
        dit = [row["bytes"] for row in headers if "anima-turbo-v1.0.safetensors" in row["file"]]
        if not dit:
            raise ValueError("Anima Turbo DiT headers missing")
        # The existing quantize_anima_dit loads a dense map, builds a packed map,
        # then serializes it; allow three DiT copies and its largest live tensor.
        staging = 3 * sum(dit) + max(dit)
    elif name.startswith("sana-"):
        components = {}
        for row in headers:
            parts = Path(row["file"]).parts
            component = "transformer" if "transformer" in parts else "text_encoder"
            components.setdefault(component, []).append(row["bytes"])
        staging = max(2 * sum(values) + max(values) for values in components.values())
    else:
        staging = 2 * total + 3 * largest
    # Account for resident/mapped input while the exporter stages its output,
    # and reserve eight GiB for unpriced runtime allocations and file buffers.
    work_bound = total + staging + 8 * GIB
    return (staging + MIB - 1) // MIB, staging, work_bound


def parse_host_memory(pressure, stat):
    total = re.search(r"The system has (\d+) ", pressure)
    free_pct = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
    page_size = re.search(r"page size of (\d+) bytes", stat)
    counts = dict(re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.", stat, re.M))
    if not total or not free_pct or not page_size or set(counts) != {"free", "inactive", "speculative"}:
        raise ValueError("Darwin available-memory probe incomplete")
    total_bytes = int(total.group(1))
    pct = int(free_pct.group(1))
    if not 0 <= pct <= 100:
        raise ValueError("invalid memory_pressure percentage")
    pressure_free = total_bytes * pct // 100
    reclaimable = int(page_size.group(1)) * sum(int(value) for value in counts.values())
    return {"physical_bytes": total_bytes, "pressure_free_bytes": pressure_free,
            "vm_reclaimable_bytes": reclaimable, "available_bytes": min(pressure_free, reclaimable)}


def host_memory():
    pressure = subprocess.check_output(["/usr/bin/memory_pressure", "-Q"], text=True, timeout=10)
    stat = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=10)
    return parse_host_memory(pressure, stat)


def admit_memory(work_bound, minimum_cap):
    observation = host_memory()
    reserve = max(16 * GIB, observation["physical_bytes"] // 8)
    cap = min(work_bound, max(0, observation["available_bytes"] - reserve))
    if cap < minimum_cap:
        raise ValueError(f"memory admission refused: available={observation['available_bytes']} "
                         f"reserve={reserve} required={minimum_cap}")
    return cap, reserve, observation


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipts(root):
    return [{"path": str(file.relative_to(root)), "bytes": file.stat().st_size, "sha256": sha(file)}
            for file in sorted(root.rglob("*")) if file.is_file()]


def verify_hub_source(name, source, source_files):
    owner, repository, revision, suffix = PINS[name]
    snapshot = source.parent if suffix else source
    relative_files = tuple(file.relative_to(snapshot).as_posix() for file in source_files)
    return _hub.verify_pinned_hub_files(f"{owner}/{repository}", revision, snapshot, relative_files)


def converter_arguments(name, source, destination, bits, budget):
    if name == "anima-turbo":
        return [str(source), str(destination)]
    common = [str(source), str(destination), str(bits), str(budget)]
    return [name.removeprefix("iris-"), *common] if name.startswith("iris-") else common


def output_paths(cache, run_id, attempt, binding, native_mode):
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("invalid Actions run identity")
    if native_mode not in {"prebuilt", "source"}:
        raise ValueError("native mode must come from the verified fetch step")
    volume = cache.resolve().parent.parent
    output_root = volume / "sceneworks-ios-prequant" / f"{run_id}-{attempt}"
    target = volume / "sceneworks-ios-prequant-build" / binding["source_sha"] / native_mode
    return output_root, output_root / binding["route"], target


def verify_native_artifact(repo, target, mode, env):
    with (repo / "Cargo.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    packages = [item for item in lock["package"] if item["name"] == "pmetal-mlx-sys"]
    if len(packages) != 1 or f"rev={MLX_REVISION}#{MLX_REVISION}" not in packages[0]["source"]:
        raise ValueError("mlx-rs source revision differs from the frozen native key")
    candidates = []
    for build_output in (target / "debug/build").glob("pmetal-mlx-sys-*/output"):
        text = build_output.read_text(errors="replace", encoding='utf-8')
        link = re.search(r"^cargo:rustc-link-search=native=(.+)$", text, re.M)
        metal = re.search(r"^cargo:metallib=(.+)$", text, re.M)
        if link and metal and Path(metal.group(1)) == Path(link.group(1)) / "mlx.metallib":
            candidates.append(Path(link.group(1)))
    if len(candidates) != 1:
        raise ValueError(f"expected one selected pmetal build artifact, found {len(candidates)}")
    native = candidates[0]
    if mode == "prebuilt":
        if native != Path(env["PMETAL_MLX_PREBUILT_DIR"]):
            raise ValueError("Cargo linked a different prebuilt from the fetched artifact")
    elif not native.resolve().is_relative_to(target.resolve()):
        raise ValueError("source build did not link its owned Cargo target")
    manifest_path = native / "pmetal-mlx-prebuilt.txt"
    manifest = dict(line.split("=", 1) for line in manifest_path.read_text(encoding='utf-8').splitlines() if "=" in line)
    expected = {"schema": "1", "target": "aarch64-apple-darwin", "build_type": "Debug",
                "deployment_target": "26.2", "features": "accelerate,metal",
                "fingerprint": MLX_FINGERPRINT}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("linked native manifest does not match the frozen MLX source/build cell")
    files = [manifest_path, *(native / name for name in ("libmlx.a", "libmlxc.a", "mlx.metallib"))]
    if not all(path.is_file() for path in files):
        raise ValueError("linked native artifact lacks archives or metallib")
    return {"mode": mode, "mlx_revision": MLX_REVISION, "directory": str(native),
            "manifest": expected, "files": [{"name": path.name, "bytes": path.stat().st_size,
                                              "sha256": sha(path)} for path in files]}


def run_guarded(argv, env, log, seconds, rss_cap, host_reserve):
    """Only signal this command's process group, never another runner job."""
    started = time.monotonic()
    peak = 0
    samples = 0
    reason = None
    minimum_available = None
    minimum_disk_free = None
    next_host_check = started
    with log.open("wb") as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        while child.poll() is None:
            try:
                rows = process_rows()
                rss = sum(rows[pid][1] for pid in descendants(rows, child.pid) if pid in rows)
                peak = max(peak, rss)
                samples += 1
                if rss > rss_cap:
                    reason = "RSS ceiling exceeded"
                if log.stat().st_size > MAX_LOG_BYTES:
                    reason = "log byte ceiling exceeded"
            except Exception as error:
                reason = f"RSS monitoring failed: {error!r}"
            if time.monotonic() >= next_host_check:
                try:
                    available = host_memory()["available_bytes"]
                    minimum_available = available if minimum_available is None else min(minimum_available, available)
                    if available < host_reserve:
                        reason = "host memory headroom lost"
                    disk_free = shutil.disk_usage(log.parent).free
                    minimum_disk_free = disk_free if minimum_disk_free is None else min(minimum_disk_free, disk_free)
                    if disk_free < DISK_FLOOR_BYTES:
                        reason = "host disk headroom lost"
                except Exception as error:
                    reason = f"host memory monitoring failed: {error!r}"
                next_host_check = time.monotonic() + 5
            if time.monotonic() - started > seconds:
                reason = "wall limit exceeded"
            if reason:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(0.25)
        exit_code = child.wait()
    if log.stat().st_size > MAX_LOG_BYTES:
        reason = reason or "log byte ceiling exceeded"
    result = {"argv": argv, "exit": exit_code, "seconds": round(time.monotonic() - started, 2),
              "rss_cap_bytes": rss_cap, "host_reserve_bytes": host_reserve,
              "minimum_observed_available_bytes": minimum_available,
              "minimum_observed_disk_free_bytes": minimum_disk_free,
              "max_sampled_tree_rss_bytes": peak, "rss_samples": samples, "guard_stop": reason,
              "log_bytes": log.stat().st_size, "log_limit_bytes": MAX_LOG_BYTES,
              "disk_floor_bytes": DISK_FLOOR_BYTES, "log_sha256": sha(log)}
    if reason or exit_code != 0 or samples == 0:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4096))
            tail = stream.read(4096).decode(errors="replace")[-2000:]
        raise RuntimeError(f"command failed: {result}; tail={tail}")
    return result


def main():
    receipt_path = Path(os.environ["RUNNER_TEMP"]) / "ios-prequant-receipt.json"
    receipt = {"state": "STARTING", "complete": False, "device": "CPU",
               "metal_trap_required": True, "outputs_uploaded": False}
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding='utf-8')
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
        budget, staging, work_bound = tensor_budget_mib(name, headers)
        receipt.update(source_tensor_count=len(headers), source_payload_bytes=sum(x["bytes"] for x in headers),
                       source_max_tensor_bytes=max(x["bytes"] for x in headers),
                       header_tensor_bound_bytes=staging, conversion_work_bound_bytes=work_bound,
                       tensor_budget_mib=budget)
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
        receipt["source_identity"] = verify_hub_source(name, source, selected_files)
        # One route per dispatch. No existing snapshot or output can be overwritten.
        run_id = os.environ["GITHUB_RUN_ID"]
        attempt = os.environ["GITHUB_RUN_ATTEMPT"]
        native_mode = os.environ["IOS_PREQUANT_NATIVE_MODE"]
        output_root, destination, target = output_paths(cache, run_id, attempt, binding, native_mode)
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
        if native_mode == "prebuilt":
            native = Path(os.environ["PMETAL_MLX_PREBUILT_DIR"])
            if not native.is_absolute() or not (native / "pmetal-mlx-prebuilt.txt").is_file():
                raise ValueError("fetched MLX prebuilt artifact missing")
            if Path(os.environ["PMETAL_METALLIB_PATH"]) != native / "mlx.metallib" or not (native / "mlx.metallib").is_file():
                raise ValueError("fetched MLX metallib missing")
        else:
            native = None
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
        if native_mode == "source":
            env.pop("PMETAL_MLX_PREBUILT_DIR", None)
            env.pop("PMETAL_METALLIB_PATH", None)
        package, example = (("mlx-gen-anima", "anima_prequant_cpu") if name == "anima-turbo" else
                            (("mlx-gen-sana", "sana_prequant_cpu") if name.startswith("sana-")
                             else ("mlx-gen-iris", "iris_prequant_cpu")))
        receipt["cli"] = {"package": package, "example": example, "rust": "1.96.0",
                          "jobs": 4, "incremental": False, "debug": 0, "native_mode": native_mode,
                          "sdk": str(sdk), "linker": str(linker)}
        build_cap, build_reserve, build_host = admit_memory(32 * GIB, 8 * GIB)
        receipt["build_admission"] = {"rss_cap_bytes": build_cap, "host_reserve_bytes": build_reserve,
                                      "host": build_host}
        save()
        receipt["build"] = run_guarded([str(cargo), "+1.96.0", "build", "--locked", "--offline",
                                        "-p", package, "--example", example], env,
                                       output_root / "build.log", BUILD_SECONDS, build_cap, build_reserve)
        receipt["native_artifact"] = verify_native_artifact(repo, target, native_mode, env)
        env["PMETAL_METALLIB_PATH"] = str(Path(receipt["native_artifact"]["directory"]) / "mlx.metallib")
        binary = target / "debug/examples" / example
        if not binary.is_file():
            raise ValueError("compiled converter missing")
        startup = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10, encoding="utf-8")
        if startup.returncode == 0 or "usage:" not in startup.stderr:
            raise ValueError("converter startup did not reject missing arguments")
        receipt["startup"] = {"exit": startup.returncode, "expected_usage_refusal": True,
                              "binary_sha256": sha(binary)}
        save()
        bits = None if name == "anima-turbo" else int(binding["route"].rsplit("-q", 1)[1])
        arguments = converter_arguments(name, source, destination, bits, budget)
        conversion_cap, conversion_reserve, conversion_host = admit_memory(work_bound, work_bound)
        receipt["conversion_admission"] = {"rss_cap_bytes": conversion_cap,
                                           "host_reserve_bytes": conversion_reserve, "host": conversion_host}
        save()
        receipt["conversion"] = run_guarded([str(binary), *arguments], env,
                                             output_root / "convert.log", CONVERT_SECONDS,
                                             conversion_cap, conversion_reserve)
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
