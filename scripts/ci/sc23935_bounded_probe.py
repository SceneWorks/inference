"""NEVER MERGE: bound one Windows native comparison and prove its process exited."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def native_pids(row: Path) -> set[int]:
    found: set[int] = set()
    receipt = row / "receipt.json"
    if receipt.is_file():
        process = json.loads(receipt.read_text(encoding="utf-8")).get("process", {})
        pid = process.get("process_id")
        if isinstance(pid, int) and pid > 0:
            found.add(pid)
    for name in ("progress-rss.jsonl", "stderr.log"):
        path = row / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            opening = line.find("{")
            if opening < 0:
                continue
            try:
                entry = json.loads(line[opening:])
            except json.JSONDecodeError:
                continue
            pid = entry.get("process_id")
            if isinstance(pid, int) and pid > 0:
                found.add(pid)
    return found


def process_exited(pid: int) -> bool:
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, 0, pid)
    if not handle:
        # Access denied is not evidence of process absence. Only the invalid-PID error is.
        return kernel.GetLastError() == 87
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0
    finally:
        kernel.CloseHandle(handle)


def load_json(path: Path) -> dict:
    # Windows PowerShell 5 Set-Content -Encoding utf8 prepends a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stage_markers(row: Path) -> list[dict]:
    result = []
    for line in (row / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("kind") == "comparison_stage_v1":
            result.append(item)
    return result


def validate_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-sha", required=True)
    args = parser.parse_args(sys.argv[2:])
    root = args.root
    if not root.is_dir():
        print(f"diagnostic evidence root is missing: {root}", file=sys.stderr)
        return 1
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def read(name: str) -> dict:
        path = root / name
        if not path.is_file():
            errors.append(f"missing {name}")
            return {}
        try:
            return load_json(path)
        except (OSError, ValueError) as error:
            errors.append(f"invalid {name}: {error}")
            return {}

    def verify_row_seal(row_name: str) -> None:
        row = root / row_name
        receipt_path = row / "receipt.json"
        manifest_path = row / "artifact-manifest.json"
        seal = read(f"{row_name}/seal.json")
        manifest = read(f"{row_name}/artifact-manifest.json")
        for name, path, expected in (
            ("receipt", receipt_path, seal.get("receipt_sha256")),
            ("artifact manifest", manifest_path, seal.get("artifact_manifest_sha256")),
        ):
            require(
                path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"{row_name} {name} seal mismatch",
            )
        for entry in manifest.get("files", []) or []:
            path = row / entry.get("path", "")
            require(
                path.is_file()
                and path.stat().st_size == entry.get("bytes")
                and hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256"),
                f"{row_name} artifact {entry.get('path')} mismatch",
            )

    provenance = read("provenance.json")
    metadata = read("snapshot-metadata.json")
    reservation = read("gpu-reservation.json")
    cpu_launch = read("cpu-launch.json")
    cuda_launch = read("cuda-launch.json")
    cpu = read("cpu-parent/receipt.json")
    cuda = read("cuda-bonsai/receipt.json")
    cpu_provider = read("cpu-parent/provider.json") if cpu.get("status") == "completed" else {}
    cuda_provider = read("cuda-bonsai/provider.json")
    expected_cases = [
        "reasoning_low", "reasoning_medium", "reasoning_xhigh", "preserve_thinking",
        "tool_roundtrip", "json_thinking", "tool", "code",
    ]
    required = expected_cases[:6]

    require(provenance.get("checked_out_sha") == args.runtime_sha, "source SHA mismatch")
    require(provenance.get("clean_tree") is True, "dirty diagnostic source")
    require(
        provenance.get("rc4_base_sha") == "7b7730a1a06a231ce337352dbc5b525eb3f8cc78",
        "wrong RC4 source base",
    )
    require(metadata.get("all_metadata_qualified") is True, "frozen snapshots not qualified")
    require(reservation.get("gpu_index") == 0, "wrong GPU index")
    require(
        reservation.get("gpu_uuid") == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
        "wrong GPU UUID",
    )
    for label, launch, receipt, revision, model_id in (
        (
            "CPU", cpu_launch, cpu, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "candle-cpu-qwen38-parent-stage-probe",
        ),
        (
            "CUDA", cuda_launch, cuda, "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
            "candle-cuda-bonsai-gguf-eight-case-probe",
        ),
    ):
        require(launch.get("safe_to_continue") is True, f"{label} process cleanup unproven")
        require(bool(launch.get("native_pids")), f"{label} native PID missing")
        require(receipt.get("runtime", {}).get("head_sha") == args.runtime_sha, f"{label} runtime SHA mismatch")
        require(receipt.get("model", {}).get("revision") == revision, f"{label} model revision mismatch")
        require(receipt.get("model", {}).get("id") == model_id, f"{label} model ID mismatch")
        require(
            bool(receipt.get("model", {}).get("inventory_before", {}).get("files")),
            f"{label} frozen publisher-closure inventory is absent",
        )
        require(receipt.get("process", {}).get("process_id") in (launch.get("native_pids") or []), f"{label} native PID mismatch")
        verify_row_seal("cpu-parent" if label == "CPU" else "cuda-bonsai")
        if receipt.get("status") == "completed":
            model = receipt.get("model", {})
            require(
                model.get("inventory_before", {}).get("inventory_sha256")
                == model.get("inventory_after", {}).get("inventory_sha256"),
                f"{label} frozen snapshot changed",
            )

    cpu_status = cpu.get("status")
    require(cpu_status in ("completed", "timed_out"), "CPU neither completed nor bounded-timeout classified")
    require(cpu_launch.get("returncode") == (0 if cpu_status == "completed" else 1), "CPU wrapper exit does not match receipt")
    cpu_process = cpu.get("process", {})
    require(cpu.get("command", {}).get("candle_device") == "cpu", "CPU device selector mismatch")
    require(cpu.get("command", {}).get("load_profile") == "candle-dense-cpu", "CPU load profile mismatch")
    require(cpu_process.get("diagnostic_timeout_seconds") == 600, "CPU 600-second watchdog absent")
    samples = root / "cpu-parent" / "progress-rss.jsonl"
    require(samples.is_file() and samples.stat().st_size > 0, "CPU progress RSS absent")
    markers = stage_markers(root / "cpu-parent") if (root / "cpu-parent/stderr.log").is_file() else []
    require(
        all(
            item.get("event") in ("start", "end")
            and isinstance(item.get("elapsed_seconds"), (int, float))
            for item in markers
        ),
        "CPU stage event or clock is invalid",
    )
    clocks = [item.get("elapsed_seconds") for item in markers]
    if all(isinstance(clock, (int, float)) for clock in clocks):
        require(clocks == sorted(clocks), "CPU stage clocks are out of order")
    require(
        all(
            item.get("run_id") == cpu_process.get("run_id")
            and item.get("process_id") == cpu_process.get("process_id")
            for item in markers
        ),
        "CPU stage marker run/PID mismatch",
    )
    cpu_stage = markers[-1].get("stage") if markers else "native_startup"
    cpu_event = markers[-1].get("event") if markers else "not_started"
    if cpu_status == "timed_out":
        cleanup = cpu_process.get("child_tree_cleanup") or {}
        require(cleanup.get("root_reaped") is True, "timed-out CPU native root not reaped")
        require(cleanup.get("tree_termination_requested") is True, "timed-out CPU tree termination absent")
        require(cpu_process.get("exit_code") != 0, "timed-out CPU process misleadingly exited zero")
    else:
        require(cpu_provider.get("status") == "completed", "CPU provider did not complete")
        require(cpu_provider.get("case_ids") == ["arithmetic"], "CPU ran more than arithmetic")
        cases = cpu_provider.get("cases", [])
        case = cases[0] if len(cases) == 1 else {}
        require(case.get("status") == "completed", "CPU arithmetic not completed")
        require(case.get("evidence_complete") is True, "CPU arithmetic evidence incomplete")
        require(case.get("functional_acceptance_passed") is True, "CPU arithmetic oracle failed")
        require(case.get("output", {}).get("text") == "391", "CPU arithmetic answer mismatch")

    require(cuda_launch.get("returncode") == 0, "CUDA wrapper did not exit zero")
    require(cuda.get("status") == "completed", "CUDA Bonsai wrapper did not complete")
    require(cuda.get("command", {}).get("candle_device") == "auto", "CUDA selector mismatch")
    require(cuda.get("command", {}).get("load_profile") == "candle-packed-cuda", "CUDA profile mismatch")
    require(cuda.get("gpu", {}).get("admission_recheck", {}).get("admitted") is True, "CUDA reservation recheck absent")
    require(cuda_provider.get("status") == "completed", "CUDA Bonsai provider did not complete")
    require(cuda_provider.get("case_ids") == expected_cases, "CUDA Bonsai case list changed")
    by_id = {case.get("case_id"): case for case in (cuda_provider.get("cases") or [])}
    for case_id in required:
        case = by_id.get(case_id, {})
        require(case.get("status") == "completed", f"{case_id} did not complete")
        require(case.get("evidence_complete") is True, f"{case_id} evidence incomplete")
        require(case.get("functional_acceptance_passed") is True, f"{case_id} acceptance failed")
        require(case.get("stream_contract_passed") is True, f"{case_id} stream contract failed")
    for case_id in ("tool", "code"):
        case = by_id.get(case_id, {})
        require(case.get("status") == "completed", f"diagnostic {case_id} did not complete")
        require(case.get("evidence_complete") is True, f"diagnostic {case_id} evidence incomplete")
        require(isinstance(case.get("output", {}).get("tool_calls"), list), f"diagnostic {case_id} output is unparsed")
    reservation_path = os.environ.get("SC23935_GPU_RESERVATION")
    require(bool(reservation_path), "owned reservation path is unknown")
    if reservation_path:
        require(not Path(reservation_path).exists(), "owned GPU reservation remains")

    verification = {
        "schema_version": 1,
        "passed": not errors,
        "runtime_sha": args.runtime_sha,
        "rc4_base_sha": provenance.get("rc4_base_sha"),
        "cpu_status": cpu_status,
        "cpu_last_stage": cpu_stage,
        "cpu_last_event": cpu_event,
        "cuda_required_six": required,
        "cuda_diagnostic_only": ["tool", "code"],
        "cuda_diagnostic_results": {
            case_id: {
                "quality_passed": by_id.get(case_id, {}).get("quality_passed"),
                "tool_calls": by_id.get(case_id, {}).get("output", {}).get("tool_calls"),
                "text": by_id.get(case_id, {}).get("output", {}).get("text"),
            }
            for case_id in ("tool", "code")
        },
        "full_campaign_required": True,
        "full_campaign_accepted": False,
        "publisher_verification_scope": "wrapper inventory checked against pinned manifest; no provision-assets publisher-closure claim",
        "errors": errors,
    }
    (root / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "probe-manifest.json":
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (root / "probe-manifest.json").write_text(
        json.dumps({"schema_version": 1, "runtime_sha": args.runtime_sha, "files": entries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2), flush=True)
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "win32" or not 1 <= args.timeout_seconds <= 7200:
        parser.error("Windows and a bounded 1..7200 second timeout are required")
    if not args.command or args.command[0] != "--" or len(args.command) < 2:
        parser.error("pass the reviewed wrapper command after --")
    command = args.command[1:]
    if args.record.exists() or args.row.exists():
        parser.error("probe output already exists")
    started = time.monotonic()
    timed_out = False
    taskkill = None
    with (args.record.parent / f"{args.row.name}-wrapper-stdout.log").open("xb") as stdout:
        with (args.record.parent / f"{args.row.name}-wrapper-stderr.log").open("xb") as stderr:
            wrapper = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                returncode = wrapper.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    taskkill = subprocess.run(
                        ["taskkill", "/PID", str(wrapper.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    pass
                try:
                    wrapper.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                returncode = 124
    pids = sorted(native_pids(args.row))
    wrapper_exited = wrapper.poll() is not None and process_exited(wrapper.pid)
    native_exited = bool(pids) and all(process_exited(pid) for pid in pids)
    record = {
        "schema_version": 1,
        "scope": "SC-23935 bounded diagnostic; never full-matrix acceptance",
        "row": args.row.name,
        "wrapper_pid": wrapper.pid,
        "native_pids": pids,
        "timeout_seconds": args.timeout_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "returncode": returncode,
        "taskkill_returncode": taskkill.returncode if taskkill is not None else None,
        "wrapper_exited": wrapper_exited,
        "native_exited": native_exited,
        "safe_to_continue": wrapper_exited and native_exited,
    }
    args.record.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)
    if not record["safe_to_continue"]:
        return 125
    return returncode


if __name__ == "__main__":
    raise SystemExit(validate_main() if len(sys.argv) > 1 and sys.argv[1] == "validate" else main())
