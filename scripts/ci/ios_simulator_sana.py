#!/usr/bin/env python3
"""Dispatch-only SANA simulator image witness; never a physical-iPhone fit claim.

simctl reparents GUI apps to launchd_sim, so a process-tree supervisor around `simctl launch`
does not own the model process. This controller gates the app until it verifies the returned PID,
executable path and kernel start identity, then samples that exact PID and host reserve until a
fresh nonce-bound report and PNG exist. Its cleanup terminates only the still-owned app.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.ci.ios_simulator_sana_inventory import content_digest

UDID = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
PNG = b"\x89PNG\r\n\x1a\n"


def require_dispatch(env: dict[str, str]) -> None:
    for key, expected in (("GITHUB_EVENT_NAME", "workflow_dispatch"),
                          ("RUNNER_OS", "macOS"),
                          ("RUNNER_ARCH", "ARM64")):
        if env.get(key) != expected:
            raise ValueError(f"{key} must be {expected}; no local simulator model execution")
    if env.get("RUNNER_NAME") not in ("nax-macos", "nax-macos-2"):
        raise ValueError("RUNNER_NAME must be a provisioned Mac runner")


def owned_booted_device(simulator: str, receipt_path: Path,
                        env: dict[str, str]) -> tuple[str, dict]:
    """Bind the booted UUID to this workflow attempt's disposable iPhone simulator."""
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_size > 4096:
        raise ValueError("owned simulator receipt missing or invalid")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    run_id, attempt = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "")
    name = f"SceneWorksSana-{run_id}-{attempt}"
    if (not run_id.isdecimal() or not attempt.isdecimal()
            or receipt.get("owned_udid") != simulator
            or receipt.get("owned_name") != name
            or receipt.get("run_id") != run_id or receipt.get("attempt") != attempt
            or receipt.get("runner") != env.get("RUNNER_NAME")
            or receipt.get("template_name") != env.get("SIMULATOR_SELECTOR")
            or not isinstance(receipt.get("template_name"), str)
            or not receipt["template_name"].startswith("iPhone ")
            or not isinstance(receipt.get("template_udid"), str)
            or not UDID.fullmatch(receipt["template_udid"])
            or receipt["template_udid"].casefold() == simulator.casefold()
            or not isinstance(receipt.get("device_type"), str)
            or not receipt["device_type"].startswith("com.apple.CoreSimulator.SimDeviceType.iPhone-")
            or not isinstance(receipt.get("runtime"), str)
            or not re.fullmatch(r"com\.apple\.CoreSimulator\.SimRuntime\.iOS-\d+-\d+", receipt["runtime"])):
        raise ValueError("owned simulator receipt does not match this workflow attempt")
    devices = json.loads(command("xcrun", "simctl", "list", "devices", "booted", "--json"))["devices"]
    matches = [(runtime, device) for runtime, rows in devices.items() for device in rows
               if device.get("udid", "").casefold() == simulator.casefold()
               and device.get("state") == "Booted" and device.get("isAvailable")
               and device.get("name") == name
               and device.get("deviceTypeIdentifier") == receipt.get("device_type")
               and runtime == receipt.get("runtime")]
    if len(matches) != 1:
        raise ValueError("owned iPhone simulator is not uniquely booted with its receipt identity")
    return matches[0]


def policy_for_simulator(path: Path) -> safety.SafetyPolicy:
    policy = safety.load_policy(path)
    if (policy.backend != "darwin-mlx" or policy.deadline_seconds > 7200
            or policy.child_footprint_cap_bytes > 12 * 1024**3
            or policy.host_free_reserve_bytes < 16 * 1024**3
            or not 100 <= policy.poll_millis <= 1000
            or policy.stdout_cap_bytes > 1024**2
            or policy.stderr_cap_bytes > 1024**2
            or policy.event_cap_bytes > 16 * 1024**2):
        raise ValueError("simulator policy exceeds the reviewed deadline/cap/output bounds or weakens host reserve")
    return policy


def command(*args: str, timeout: float = 20, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=timeout, check=True)
    if len(result.stdout) > 1024**2 or len(result.stderr) > 1024**2:
        raise ValueError("simctl output exceeded 1 MiB")
    return result.stdout.decode("utf-8", "strict").strip()


def app_identity(pid: int) -> tuple[str, int] | None:
    library = safety._load_libproc()
    if library is None:
        raise ValueError("Darwin libproc unavailable")
    library.proc_pidpath.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    library.proc_pidpath.restype = ctypes.c_int
    path = ctypes.create_string_buffer(4096)
    count = library.proc_pidpath(pid, path, len(path))
    if count <= 0:
        return None
    info = safety._RusageInfoV4()
    if library.proc_pid_rusage(pid, safety._RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return os.path.realpath(os.fsdecode(path.value)), info.ri_proc_start_abstime


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_pixels(path: Path) -> tuple[int, int, int, int, int]:
    """Decode the fixed RGB8 PNG format the Rust image crate writes; verify real pixel spread."""
    data = path.read_bytes()
    if not data.startswith(PNG):
        raise ValueError("PNG signature missing")
    at, compressed, size = len(PNG), bytearray(), None
    while at + 12 <= len(data):
        length = int.from_bytes(data[at:at + 4], "big")
        kind = data[at + 4:at + 8]
        end = at + 12 + length
        if end > len(data) or zlib.crc32(data[at + 4:at + 8 + length]) & 0xffffffff != int.from_bytes(data[at + 8 + length:end], "big"):
            raise ValueError("PNG chunk length or CRC invalid")
        payload = data[at + 8:at + 8 + length]
        if kind == b"IHDR":
            width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (width, height, bits, color, compression, filtering, interlace) != (256, 256, 8, 2, 0, 0, 0):
                raise ValueError("PNG is not the required 256x256 RGB8 image")
            size = width * 3
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
        at = end
    if size is None or not compressed:
        raise ValueError("PNG lacks image payload")
    decoder = zlib.decompressobj()
    expected = 256 * (size + 1)
    raw = decoder.decompress(compressed, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError("PNG decoded length is invalid")
    before = bytearray(size)
    low, high = 255, 0
    channel_low, channel_high = [255] * 3, [0] * 3
    for row in range(256):
        start = row * (size + 1)
        filt = raw[start]
        scan = bytearray(raw[start + 1:start + 1 + size])
        if filt > 4:
            raise ValueError("PNG filter unsupported")
        for i, value in enumerate(scan):
            left = scan[i - 3] if i >= 3 else 0
            above = before[i]
            upper_left = before[i - 3] if i >= 3 else 0
            if filt == 1: value += left
            elif filt == 2: value += above
            elif filt == 3: value += (left + above) // 2
            elif filt == 4:
                p = left + above - upper_left
                distances = (abs(p - left), abs(p - above), abs(p - upper_left))
                value += (left, above, upper_left)[distances.index(min(distances))]
            scan[i] = value & 255
        low, high = min(low, min(scan)), max(high, max(scan))
        for channel in range(3):
            values = scan[channel::3]
            channel_low[channel] = min(channel_low[channel], min(values))
            channel_high[channel] = max(channel_high[channel], max(values))
        before = scan
    spatial_span = max(hi - lo for hi, lo in zip(channel_high, channel_low))
    if spatial_span < 32:
        raise ValueError("simulator PNG is spatially near-constant")
    return 256, 256, low, high, spatial_span


def retain_failure_diagnostics(docs: Path, output: Path, policy: safety.SafetyPolicy,
                               nonce: str, source_sha: str) -> None:
    """Retain bounded app breadcrumbs before owned-simulator cleanup; never certify them."""
    files = {}
    for name, destination, cap in (("smoke-report.txt", "failure-report.txt", policy.stdout_cap_bytes),
                                   ("sana-progress.txt", "failure-progress.txt", policy.stderr_cap_bytes),
                                   ("launch-memory.txt", "failure-launch-memory.txt", 4096),
                                   ("simulator-stdout.txt", "failure-stdout.txt", policy.stdout_cap_bytes),
                                   ("simulator-stderr.txt", "failure-stderr.txt", policy.stderr_cap_bytes)):
        path = docs / name
        try:
            meta = path.lstat()
            if not stat.S_ISREG(meta.st_mode) or meta.st_size > cap:
                files[name] = "absent, nonregular, or over cap"
                continue
            with path.open("rb") as stream:
                data = stream.read(cap + 1)
            if len(data) > cap:
                files[name] = "grew beyond cap during read"
                continue
            (output / destination).write_bytes(data)
            files[name] = {"path": destination, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}
            if name == "smoke-report.txt":
                files[name]["nonce_seen"] = f"SIMULATOR_NONCE {nonce}\n".encode() in data
                files[name]["source_seen"] = f"SOURCE_SHA {source_sha}\n".encode() in data
            elif name == "sana-progress.txt":
                files[name]["nonce_and_source_seen"] = (
                    f"SIMULATOR_NONCE {nonce} SOURCE_SHA {source_sha} stage=".encode() in data)
        except OSError as error:
            files[name] = f"unavailable: {type(error).__name__}"
    (output / "failure-diagnostics.json").write_text(json.dumps({
        "accepted": False, "claim": "diagnostic-only app breadcrumbs; no functional or phone-fit evidence",
        "source_sha": source_sha, "nonce": nonce, "files": files,
    }, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    require_dispatch(os.environ)
    if not UDID.fullmatch(args.simulator) or not SHA.fullmatch(args.source_sha):
        raise ValueError("exact simulator UDID and 40-hex source SHA required")
    actual = command("git", "rev-parse", "HEAD")
    if actual != args.source_sha:
        raise ValueError("checked-out source SHA differs from requested source")
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("fresh absolute output directory required")
    policy = policy_for_simulator(args.policy)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("model") != "sana-1600m-mlx" or not re.fullmatch("[0-9a-f]{64}", inventory.get("inventory_sha256", "")):
        raise ValueError("verified SANA content inventory required")
    app_dir = Path(command("xcrun", "simctl", "get_app_container", args.simulator, args.bundle_id, "app")).resolve()
    data_dir = Path(command("xcrun", "simctl", "get_app_container", args.simulator, args.bundle_id, "data")).resolve()
    app_info = plistlib.loads((app_dir / "Info.plist").read_bytes())
    if app_info.get("CFBundleIdentifier") != args.bundle_id:
        raise ValueError("installed simulator bundle identity mismatch")
    runtime, device = owned_booted_device(args.simulator, args.owned_receipt, os.environ)
    executable = (app_dir / app_info["CFBundleExecutable"]).resolve()
    metallib = app_dir / "mlx.metallib"
    if not executable.is_file() or not metallib.is_file():
        raise ValueError("installed app executable or simulator metallib absent")
    docs = data_dir / "Documents"
    snapshot = docs / "sana"
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.resolve() != snapshot:
        raise ValueError("the exact Documents/sana snapshot must be a real directory")
    if args.copied_inventory.is_symlink():
        raise ValueError("copied snapshot inventory must be a real file")
    copied = json.loads(args.copied_inventory.read_text(encoding="utf-8"))
    expected_tier = sorted(({"path": row["path"][5:], "size": row["size"],
                             "sha256": row["sha256"]} for row in inventory["files"]
                            if row["path"].startswith("bf16/")), key=lambda row: row["path"])
    if (copied.get("source_inventory_sha256") != inventory["inventory_sha256"]
            or copied.get("revision") != inventory.get("revision")
            or copied.get("tier") != "bf16"
            or copied.get("files") != expected_tier
            or copied.get("tier_inventory_sha256") != content_digest(expected_tier)):
        raise ValueError("installed snapshot content differs from verified source")
    for name in ("smoke-report.txt", "renders.txt", "sana-progress.txt",
                 "simulator-stdout.txt", "simulator-stderr.txt"):
        if (docs / name).exists():
            raise ValueError("fresh simulator app container required; stale report or render manifest")
    if list(docs.glob("sana-*.png")):
        raise ValueError("fresh simulator app container required; stale PNG")
    probe = safety.SystemProbe(policy)
    free, host = safety.read_host(probe)
    if free < policy.host_free_reserve_bytes + policy.child_footprint_cap_bytes:
        raise ValueError("host free memory below app footprint cap plus reserve")
    args.output.mkdir(parents=True)
    (args.output / "admission.json").write_text(json.dumps({"source_sha": actual, "inventory": inventory,
        "policy_sha256": policy.sha256, "host_preflight": host}, sort_keys=True) + "\n", encoding="utf-8")
    nonce = os.urandom(16).hex()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("IOS_SMOKE_", "IOS_CAMPAIGN_", "SIMCTL_CHILD_"))}
    env.update(SIMCTL_CHILD_IOS_SIMULATOR_SANA="1",
               SIMCTL_CHILD_IOS_SIMULATOR_SANA_NONCE=nonce,
               SIMCTL_CHILD_IOS_CAMPAIGN_BUILD_SHA=actual)
    pid, claimed, peak, minimum_free = None, None, 0, free
    started = time.monotonic()
    def interrupted(_signal, _frame):
        raise InterruptedError("simulator validation interrupted")
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        stdout_file, stderr_file = docs / "simulator-stdout.txt", docs / "simulator-stderr.txt"
        launched = command("xcrun", "simctl", "launch", f"--stdout={stdout_file}",
                           f"--stderr={stderr_file}", args.simulator, args.bundle_id,
                           timeout=30, env=env)
        match = re.fullmatch(re.escape(args.bundle_id) + r":\s*(\d+)", launched)
        if not match:
            raise ValueError("simctl did not return the exact bundle PID")
        pid = int(match.group(1))
        claimed = app_identity(pid)
        if claimed is None or claimed[0] != str(executable) or claimed[1] <= 0:
            raise ValueError("launched PID path/start did not match installed app")
        (args.output / "launch-attempt.json").write_text(json.dumps({
            "accepted": False, "claim": "diagnostic-only claimed app launch; no image or phone-fit evidence",
            "source_sha": actual, "nonce": nonce, "bundle_id": args.bundle_id,
            "simulator_udid": args.simulator, "runner_name": os.environ["RUNNER_NAME"],
            "pid": pid, "executable_path": claimed[0], "process_start_abstime": claimed[1],
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        (docs / f"simulator-go-{nonce}").write_text(actual + "\n", encoding="utf-8")
        samples = []
        report_file = docs / "smoke-report.txt"
        png_file = docs / "sana-256-tile192-SIMULATOR-FUNCTIONAL-ONLY.png"
        while time.monotonic() - started < policy.deadline_seconds:
            if app_identity(pid) != claimed:
                raise ValueError("owned simulator app PID exited or changed identity before verified report")
            footprint, lifetime = safety.proc_pid_rusage_footprint(pid)
            free_now, _ = safety.read_host(probe)
            if footprint > policy.child_footprint_cap_bytes or free_now < policy.host_free_reserve_bytes:
                raise ValueError("simulator footprint cap or host reserve breached")
            peak, minimum_free = max(peak, footprint, lifetime), min(minimum_free, free_now)
            for path, cap in ((report_file, policy.stdout_cap_bytes),
                              (docs / "sana-progress.txt", policy.stderr_cap_bytes),
                              (docs / "launch-memory.txt", 4096),
                              (stdout_file, policy.stdout_cap_bytes),
                              (stderr_file, policy.stderr_cap_bytes),
                              (png_file, policy.event_cap_bytes)):
                if path.exists() and (path.is_symlink() or path.stat().st_size > cap):
                    raise ValueError(f"simulator output cap or file type breached: {path.name}")
            samples.append({"elapsed_ms": round((time.monotonic() - started) * 1000),
                            "footprint_bytes": footprint, "host_available_bytes": free_now})
            if report_file.is_file():
                report = report_file.read_text(encoding="utf-8")
                expected = ("SMOKE: PASS\n", f"SIMULATOR_NONCE {nonce}\n", f"SOURCE_SHA {actual}\n",
                            "SIMULATOR FUNCTIONAL ONLY; no phone admission, memory fit, or performance claim")
                if not all(token in report for token in expected) or "[ok] SANA image generation (E5)" not in report:
                    raise ValueError("fresh simulator SANA report failed or lacked source/nonce/body witness")
                if not png_file.is_file():
                    raise ValueError("SANA report has no exact PNG")
                width, height, low, high, spatial_span = png_pixels(png_file)
                png = args.output / png_file.name
                png.write_bytes(png_file.read_bytes())
                (args.output / "report.txt").write_text(report, encoding="utf-8")
                (args.output / "result.json").write_text(json.dumps({"accepted": True,
                    "claim": "simulator SANA functional image only; no phone fit/admission/performance",
                    "source_sha": actual, "snapshot_inventory_sha256": inventory["inventory_sha256"],
                    "tier": "bf16", "tier_inventory_sha256": copied["tier_inventory_sha256"],
                    "bundle_id": args.bundle_id, "simulator_udid": args.simulator,
                    "runner_name": os.environ["RUNNER_NAME"],
                    "simulator_name": device["name"], "simulator_runtime": runtime,
                    "pid": pid, "process_start_abstime": claimed[1], "executable_sha256": sha256_file(executable),
                    "metallib_sha256": sha256_file(metallib),
                    "png": png.name, "png_sha256": sha256_file(png),
                    "width": width, "height": height, "pixel_min": low, "pixel_max": high,
                    "max_channel_spatial_span": spatial_span,
                    "peak_phys_footprint_bytes": peak, "minimum_host_available_bytes": minimum_free,
                    "duration_seconds": time.monotonic() - started, "samples": samples}, sort_keys=True) + "\n", encoding="utf-8")
                return 0
            time.sleep(policy.poll_millis / 1000)
        raise ValueError("simulator app exceeded hard validation deadline")
    finally:
        if args.output.is_dir() and not (args.output / "result.json").exists():
            try:
                retain_failure_diagnostics(docs, args.output, policy, nonce, actual)
            except OSError:
                pass  # Do not hide the primary refusal or delay owned-process cleanup.
        gate = docs / f"simulator-go-{nonce}"
        gate.unlink(missing_ok=True)
        if pid is not None and claimed is not None and app_identity(pid) == claimed:
            try:
                command("xcrun", "simctl", "terminate", args.simulator, args.bundle_id, timeout=15)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("simulator", "bundle-id", "source-sha", "inventory", "copied-inventory", "owned-receipt", "policy", "output"):
        parser.add_argument("--" + flag, required=True, type=Path if flag in ("inventory", "copied-inventory", "owned-receipt", "policy", "output") else str)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, subprocess.SubprocessError, safety.SupervisionError, InterruptedError) as error:
        if args.output.is_absolute() and args.output.is_dir():
            (args.output / "failure.json").write_text(json.dumps({"accepted": False,
                "reason": str(error), "claim": "no simulator functional acceptance or phone-fit claim"},
                sort_keys=True) + "\n", encoding="utf-8")
        print(f"simulator SANA validation refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
