#!/usr/bin/env python3
"""Narrow external control for the dispatch-only YuE2 precision real-weight test.

This does not grade app admission: it records the engine's actual one-test verdict,
process release, and raw external device/footprint observations for later audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

REFERENCE_SHA256 = "c4073edb3c7abfbf5d50a9c5bfa67c73b9b9a8ba20115570ddb616d00d5b72d4"
TEST_NAME = "explicit_stage_precision_real_weights"
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


def resolve_binary(args: argparse.Namespace) -> None:
    candidates = []
    for line in args.build_json.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = row.get("target") or {}
        if row.get("reason") == "compiler-artifact" and target.get("name") == "precision_real_weights":
            executable = row.get("executable")
            if executable:
                candidates.append(Path(executable))
    require(len(candidates) == 1, f"expected exactly one precision test executable; got {candidates}")
    require(candidates[0].is_file(), f"missing compiled test binary: {candidates[0]}")
    args.output.write_text(str(candidates[0].resolve()) + "\n", encoding="utf-8")


def verify_reference(args: argparse.Namespace) -> None:
    metadata = json.loads((args.directory / "reference-provenance.json").read_text(encoding="utf-8"))
    source = args.directory / "vae_real_reference.safetensors"
    require(metadata.get("engine_sha") == args.engine_sha, "reference-stage engine SHA differs from this checkout")
    require(source.is_file(), "reference artifact is absent")
    digest = sha256(source)
    require(digest == REFERENCE_SHA256 == metadata.get("sha256"), "reference digest differs from committed fixture")
    print(f"pinned external reference verified: {digest}, {source.stat().st_size} bytes")


def compute_capable_rows(output: str) -> list[str]:
    rows = []
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
            rows.append(line)
    require(columns, "nvidia-smi pmon output lacks typed process columns")
    return rows


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


def cuda_census() -> tuple[str, list[str]]:
    command = ["nvidia-smi", "pmon", "-i", "0", "-c", "1", "-s", "um"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode == 0:
        return result.stdout, compute_capable_rows(result.stdout)
    # Some Windows drivers do not expose pmon. The supported apps query has no
    # C/G type, so conservatively refuse every process it reports.
    fallback = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    require(fallback.returncode == 0,
            f"CUDA census unavailable: pmon: {result.stderr.strip()}; query-compute-apps: {fallback.stderr.strip()}")
    return (f"pmon unavailable: {result.stderr.strip()}\nquery-compute-apps:\n{fallback.stdout}",
            query_compute_apps_rows(fallback.stdout))


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
    result = subprocess.run(["/bin/ps", "-axo", "pid=,comm="], capture_output=True, text=True, timeout=20)
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
    command = ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    require(result.returncode == 0, result.stderr.strip() or "nvidia-smi sample failed")
    return {"started_utc_ns": started, "ended_utc_ns": time.time_ns(),
            "method": "nvidia-smi query-gpu", "raw": result.stdout.strip()}


def sample_metal(pid: int) -> dict:
    started = time.time_ns()
    with tempfile.TemporaryDirectory(prefix="yue2-footprint-") as temp:
        output = Path(temp) / "footprint.json"
        command = ["/usr/bin/footprint", "--noCategories", "-j", str(output), "-p", str(pid)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        require(result.returncode == 0, result.stderr.strip() or "footprint sample failed")
        payload = json.loads(output.read_text(encoding="utf-8"))
        matches = [p for p in payload.get("processes", []) if p.get("pid") == pid]
        require(len(matches) == 1, "footprint omitted or duplicated the owned test PID")
        value = matches[0].get("auxiliary", {}).get("phys_footprint")
        require(isinstance(value, int) and value >= 0, "footprint omitted phys_footprint")
        return {"started_utc_ns": started, "ended_utc_ns": time.time_ns(),
                "method": "Darwin phys_footprint", "pid": pid,
                "phys_footprint_bytes": value}


def one_test_executed(output: str) -> bool:
    return (f"test {TEST_NAME} ..." in output and
            "test result: ok. 1 passed; 0 failed; 0 ignored" in output)


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
    require(not args.work_dir.exists(), "refuse to reuse an earlier precision listening directory")
    require(args.work_dir.parent.is_dir(), "persistent listening parent is unavailable")
    require(re.fullmatch(r"[0-9a-f]{40}", args.engine_sha) is not None, "engine SHA must be full lowercase hex")
    require(not args.app_sha or re.fullmatch(r"[0-9a-f]{40}", args.app_sha) is not None,
            "optional caller app SHA must be full lowercase hex")
    require(os.environ.get("GITHUB_SHA") == args.engine_sha, "checked-out engine SHA differs from dispatch input")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    require(head == args.engine_sha, "engine checkout moved after build")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                           capture_output=True, text=True, check=True).stdout
    require(not dirty.strip(), "engine source became dirty before hardware execution")
    runner = os.environ.get("RUNNER_NAME", "")
    if args.backend == "metal":
        require(runner == "nax-macos-2", f"Metal proof assigned to wrong runner: {runner}")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
                "CUDA proof must bind the same physical GPU 0 used by its process census")
    before_raw, before_busy = cuda_census() if args.backend == "cuda" else metal_census()
    (evidence / "census-before.txt").write_text(before_raw, encoding="utf-8")
    require(not before_busy, f"foreign/lingering accelerator executables before test: {before_busy}")
    env = os.environ.copy()
    env["YUE2_VAE_REFERENCE_DIR"] = str(args.reference)
    env["YUE2_PRECISION_RECEIPT"] = str(evidence / "precision-receipt.json")
    env["YUE2_PRECISION_WORK_DIR"] = str(args.work_dir)
    command = [str(args.binary), "--ignored", "--exact", TEST_NAME, "--nocapture", "--test-threads", "1"]
    started = time.time_ns()
    samples: list[dict] = []
    faults: list[str] = []
    stop = threading.Event()
    with (evidence / "test.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        def loop() -> None:
            while not stop.is_set() and child.poll() is None:
                try:
                    samples.append(sample_cuda() if args.backend == "cuda" else sample_metal(child.pid))
                except Exception as error:  # keep raw failure, never turn missing telemetry green
                    if child.poll() is None:
                        faults.append(f"{time.time_ns()}: {error}")
                stop.wait(0.25 if args.backend == "cuda" else 1.0)
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        code = child.wait()
        stop.set()
        thread.join(timeout=25)
    ended = time.time_ns()
    post_census_error = None
    try:
        after_raw, after_busy = cuda_census() if args.backend == "cuda" else metal_census()
    except Exception as error:
        after_raw, after_busy = "", []
        post_census_error = str(error)
    (evidence / "census-after.txt").write_text(after_raw, encoding="utf-8")
    write_json(evidence / "external-samples.json", {"backend": args.backend, "samples": samples, "faults": faults})
    output = (evidence / "test.log").read_text(encoding="utf-8", errors="replace")
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
                if len(fields) >= 4 and fields[2].isdigit():
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
              "engine_sha": args.engine_sha, "caller_app_sha": args.app_sha,
              "runner_name": runner, "binary_sha256": sha256(args.binary),
              "reference_sha256": sha256(reference), "started_utc_ns": started,
              "persistent_listening_dir": str(args.work_dir), "local_audio": local_audio,
              "ended_utc_ns": ended, "test_exit_code": code, "sample_count": len(samples),
              "stage_marker_count": len(markers), "external_peak": peak,
              "missing_stage_markers": missing_markers,
              "stage_sample_coverage": coverage,
              "external_peak_unit": "MiB global device used" if args.backend == "cuda" else "bytes owned phys_footprint",
              "sampler_faults": faults, "owned_test_pid": child.pid,
              "owned_test_released": child.poll() is not None,
              "post_census_busy": after_busy, "post_census_error": post_census_error,
              "receipt_sha256": sha256(receipt) if receipt.is_file() else None,
              "receipt_schema_error": receipt_schema_error}
    write_json(evidence / "control.json", report)
    print(json.dumps(report, indent=2), flush=True)
    require(code == 0, f"precision test exited {code}; see test.log")
    require(one_test_executed(output), "one exact ignored test did not execute")
    require(receipt.is_file(), "precision test did not produce its receipt")
    require(receipt_schema_error is None, f"precision receipt contract mismatch: {receipt_schema_error}")
    require(local_audio, "precision test completed without retained runner-local WAVs")
    require(markers, "precision test emitted no stage boundaries for sampler attribution")
    require(not missing_markers, f"precision stage markers incomplete: {missing_markers}")
    require(samples and not faults, "external sampler had no valid sample or suffered a fault")
    require(post_census_error is None, f"post-test release census failed: {post_census_error}")
    require(child.poll() is not None and not after_busy, f"owned test/process cleanup uncertain: {after_busy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("resolve-binary")
    p.add_argument("--build-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify-reference")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--engine-sha", required=True)
    p = sub.add_parser("run")
    p.add_argument("--backend", choices=("cuda", "metal"), required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--engine-sha", required=True)
    p.add_argument("--app-sha", default="")
    args = parser.parse_args()
    {"resolve-binary": resolve_binary, "verify-reference": verify_reference, "run": execute}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"::error::{error}", file=sys.stderr)
        sys.exit(1)
