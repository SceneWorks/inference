"""Bounded weights-free native S3 CUDA build/lint/self-test; manual dispatch only.

The exact S0 physical guard is shared unchanged. No download, random benchmark,
real weights, production setting change, or alternative sparse kernel is used.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release import veda_cuda_guard as guard

TEST = "veda::tests::actual_cuda_kernel_gate"
KERNEL = "cuda-veda-sage-int8-tiled-v2"
# v2 (sc-25777): one scoring pass per same-shape head batch (`predictor_score_batches`, was
# per-head `predictor_score_calls`), plus the exact scorer/mask and tiled-vs-pointer-v1 checks.
SCHEMA = "sc-25777-actual-cuda-kernel-v2"


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def batch_command(phase: str) -> list[str]:
    if phase not in {"build", "lint"}:
        raise ValueError("unknown S3 phase")
    return ["cmd", "/d", "/c", f"scripts\\ci\\veda-s3-cuda\\{phase}.cmd"]


def compiled_binary(out: Path, target: Path) -> Path:
    """Use Cargo's exact test artifact; never glob a stale test executable."""
    candidates = []
    finished = False
    for line in (out / "build.stdout.log").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # vcvars diagnostics precede Cargo's JSON stream.
        if row.get("reason") == "build-finished":
            finished = row.get("success") is True
        if (row.get("reason") == "compiler-artifact"
                and row.get("target", {}).get("name") == "candle_gen_minimax_h3"
                and row.get("profile", {}).get("test") is True
                and row.get("executable")):
            candidates.append(Path(row["executable"]))
    if not finished or len(candidates) != 1:
        raise RuntimeError("Cargo did not identify exactly one completed S3 library test binary")
    binary = candidates[0].resolve()
    if not binary.is_relative_to(target.resolve()) or not binary.is_file() or binary.suffix != ".exe":
        raise RuntimeError("S3 test binary is absent or outside the isolated Cargo target")
    return binary


def verify_test_output(out: Path, phase: str, result: dict, absent: bool) -> list[dict]:
    stdout = (out / f"{phase}.stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr = (out / f"{phase}.stderr.log").read_text(encoding="utf-8", errors="replace")
    if result["timeout"] or "running 1 test" not in stdout or TEST not in stdout:
        raise RuntimeError(f"{phase}: exact single S3 device gate did not execute")
    if absent:
        if result["returncode"] != 101 or "0 passed; 1 failed" not in stdout or "S3 requires an exclusively admitted CUDA device" not in stderr:
            raise RuntimeError("S3 missing-device gate did not fail explicitly; skip-success is forbidden")
        return []
    if result["returncode"] != 0 or "1 passed; 0 failed; 0 ignored" not in stdout:
        raise RuntimeError("S3 actual-device test failed or did not prove one nonignored test")
    rows = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line.partition("S3_DEVICE_RECEIPT ")[2])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA:
            rows.append(row)
    if len(rows) != 2 or {row.get("dtype") for row in rows} != {"BF16", "F16"}:
        raise RuntimeError("S3 must preserve both BF16 and F16 device receipts")
    for row in rows:
        if (row.get("kernel") != KERNEL or row.get("device_ordinal") != 0
                or row.get("self_test_dimensions") != [16, 128]
                or row.get("native_batch") != 2 or row.get("native_heads") != 2
                or row.get("sparse_calls") != 1 or row.get("predictor_score_batches") != 2
                or row.get("exact_scorer_and_masks") is not True
                or row.get("tiled_kernel_bits_equal_pointer_v1") is not True
                or row.get("forced_failure_dense") is not True
                or row.get("unsupported_dtype_dense") is not True
                or row.get("intentional_dense_identity") is not True):
            raise RuntimeError("S3 device receipt contract differs from the selected native gate")
    return rows


def append_result(receipt: dict, out: Path, phase: str, argv: list[str], deadline: float, job_type) -> dict:
    receipt["status"] = f"{phase}_starting"
    receipt["active_phase"] = phase
    receipt["active_argv"] = argv
    guard.persist_receipt(out, receipt)
    result = guard.execute_owned(phase, argv, out, deadline, job_type)
    for suffix in ("stdout", "stderr"):
        result[f"{suffix}_sha256"] = hash_file(out / f"{phase}.{suffix}.log")
    receipt["results"].append(result)
    receipt["status"] = f"{phase}_child_completed"
    guard.persist_receipt(out, receipt)
    return result


def main() -> int:
    out = Path(os.environ["S3_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + guard.MAX_SECONDS
    receipt = dict(schema="sc-25771-cuda-validation-v1", instrument="veda_s3_cuda",
        complete=False, status="starting", host=socket.gethostname(), runner=os.environ.get("RUNNER_NAME"),
        source_commit=guard.command(["git", "rev-parse", "HEAD"]), results=[],
        max_session_seconds=guard.MAX_SECONDS, started_unix=time.time(),
        owned_deadline_unix=time.time()+guard.MAX_SECONDS, kernel=KERNEL,
        methodology="Weights-free production kernel and learned native runtime fixtures; no benchmark or media-quality claim.")
    guard.persist_receipt(out, receipt)
    lock = None
    try:
        if os.name != "nt" or receipt["runner"] not in guard.ALLOWED_RUNNERS:
            raise RuntimeError("S3 requires an explicitly admitted existing Windows CUDA listener")
        if receipt["source_commit"] != os.environ["GITHUB_SHA"] or guard.command(["git", "status", "--porcelain"]):
            raise RuntimeError("checkout identity/cleanliness mismatch")
        receipt["kernel_source_sha256"] = hash_file(Path("crates/kernels/candle-quant-kernels/src/veda_int8.cu"))
        receipt["rustc"] = guard.command(["rustc", "--version"])
        device, lock, baseline = guard.claim_device(out)
        receipt["device"] = device
        receipt["graphics_baseline_policy"] = dict(max_baseline_mib=guard.MAX_GRAPHICS_BASELINE_MIB,
            max_utilization_percent=guard.MAX_BASELINE_UTILIZATION_PERCENT, sample_count=guard.BASELINE_SAMPLES,
            max_sampling_seconds=guard.MAX_SAMPLING_SECONDS, baseline=baseline,
            admitted_process_types=["G", "C+G"], refused_process_types=["C", "unknown"],
            type_authority="nvidia-smi -q -x driver XML; no executable-name inference")
        receipt["device_lock_directory"] = str(Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks")
        os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        receipt["cuda_visible_devices"] = device["uuid"]
        receipt["status"] = "device_admitted"
        guard.persist_receipt(out, receipt)
        from scripts.media_campaign_windows import WindowsJob
        for phase in ("build", "lint"):
            guard.verify_idle(device, out, phase, baseline)
            result = append_result(receipt, out, phase, batch_command(phase), deadline, WindowsJob)
            if result["timeout"] or result["returncode"] != 0:
                receipt["status"] = f"{phase}_failed"
                return result["returncode"] or 124
        binary = compiled_binary(out, Path(os.environ["CARGO_TARGET_DIR"]))
        receipt["binary_path"] = str(binary)
        receipt["binary_sha256"] = hash_file(binary)
        argv = [str(binary), TEST, "--exact", "--ignored", "--nocapture"]
        for phase, absent in (("absent_device", True), ("actual_device", False)):
            guard.verify_idle(device, out, phase, baseline)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if absent else device["uuid"]
            before = hash_file(binary)
            if before != receipt["binary_sha256"]:
                raise RuntimeError("S3 binary changed before test")
            result = append_result(receipt, out, phase, argv, deadline, WindowsJob)
            result["binary_sha256_before"] = before
            result["binary_sha256_after"] = hash_file(binary)
            result["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
            guard.persist_receipt(out, receipt)
            if result["binary_sha256_after"] != before:
                raise RuntimeError("S3 binary changed during test")
            rows = verify_test_output(out, phase, result, absent)
            receipt[f"{phase}_receipts"] = rows
            receipt["status"] = f"{phase}_verified"
            guard.persist_receipt(out, receipt)
        if guard.command(["git", "rev-parse", "HEAD"]) != receipt["source_commit"] or guard.command(["git", "status", "--porcelain"]):
            raise RuntimeError("checkout changed during S3 validation")
        receipt["status"] = "complete"
        receipt["complete"] = True
        return 0
    except Exception as exc:
        receipt["status"] = "refused_or_failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        print(receipt["error"], file=sys.stderr)
        return 2
    finally:
        receipt["artifact_sha256"] = {path.name: hash_file(path) for path in out.iterdir()
            if path.is_file() and not path.name.startswith("receipt.json")}
        receipt["finished_unix"] = time.time()
        guard.persist_receipt(out, receipt)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
