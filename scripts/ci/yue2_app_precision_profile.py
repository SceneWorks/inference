#!/usr/bin/env python3
"""Bounded, exact-source app precision profile control; no GPU work in this module."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CASES = {
    "strict-bf16-standard": ("bf16", "strict-bf16-standard", "standard", "bf16", "bfloat16", "bfloat16"),
    "strict-bf16-legacy": ("bf16", "strict-bf16-legacy", "legacy", "bf16", "bfloat16", "bfloat16"),
    "strict-fp32-standard": ("bf16", "strict-fp32-standard", "standard", "fp32", "float32", "float32"),
    "strict-bf16-q8-standard": ("q8", "strict-bf16-standard", "standard", "bf16", "bfloat16", "bfloat16"),
    "strict-bf16-q4-standard": ("q4", "strict-bf16-standard", "standard", "bf16", "bfloat16", "bfloat16"),
    "strict-fp32-q8-standard": ("q8", "strict-fp32-standard", "standard", "fp32", "float32", "float32"),
    "strict-fp32-q4-standard": ("q4", "strict-fp32-standard", "standard", "fp32", "float32", "float32"),
    "experimental-fp8-auto": ("bf16", "experimental-fp8-auto", "standard", "auto", "bfloat16", "float32"),
}
NAMES = tuple(CASES)
CUDA_ONLY_CASES = frozenset({"experimental-fp8-auto"})
STAGES = ("load", "plan", "semantic", "acoustic", "decode")
CASE_SOURCE_SHA256 = {
    "strict-bf16-standard": "a43f2b1c3f8c288a4963885a924a91c6449d8d654db8a0c59619fba74c7f88bc",
    "strict-bf16-legacy": "077cef67bb570c0bbbddd6bff8caa6696363376b4194600af83336256f3ac5b2",
    "strict-fp32-standard": "0250c211b09b495608b316ec950d2de39b377c6cad6a04f236f3e37cd911d369",
    "strict-bf16-q8-standard": "42623e2c49445c090b8cb503c46c7942d1a5dbc22b6d7e73a42153678151392a",
    "strict-bf16-q4-standard": "a5cdad80a36c95db51eca85961701f6b2ee4208c4ebbd5b244ef013e6bde3492",
    "strict-fp32-q8-standard": "68534c2b050e8147b804b7053c5b9b7b71ba18fa1377716d60d69c253525c4ba",
    "strict-fp32-q4-standard": "3162d290206e8c16e24c1361f6f5abbe1ee738e6652a08bf3da47ee12aba1910",
    "experimental-fp8-auto": "56d5dee367c86ae4ff142dcd30c1f9a194201cddb4f0d4afad7440f15e4f36d7",
}
SUPPORTED_RUNTIME_POLICIES = {
    # The M4 CUDA BF16 VAE used a private cuBLAS handle in this mode. A changed
    # arithmetic policy must be bound to its exact engine revision before capture.
    "825341ff8d0110ea448213485891b39d57806fa4": "disallow_reduced_precision_reduction_v1",
}


def names_for_backend(backend: str) -> tuple[str, ...]:
    require(backend in ("cuda", "metal"), "unsupported backend")
    return tuple(name for name in NAMES if backend == "cuda" or name not in CUDA_ONLY_CASES)


def case_id(backend: str, name: str) -> str:
    tier, case_name, *_ = CASES[name]
    return f"yue2:{tier}:{backend}:{case_name}"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MIN_FREE_DISK = 4 * 7_261_441_640  # Pinned model checkpoint equivalents for fresh install + build.
REFERENCE_PEAK = 23_184_818_176  # Measured CPU F32 reference, a preflight floor, not a safe GPU bound.
RECEIPT_FILES = (
    "case.json", "admission.json", "stages.jsonl", "outcome.json", "record.json",
    "cuda-samples.jsonl", "watchdog.jsonl",
)
OFF_PLAN_CHECK = """
import { readFileSync } from 'node:fs';
import { readSources, validateRecord, externalCaseItem, caseIdentity, coverage } from './scripts/yue2-memory-profile.mjs';
const [recordPath, casePath] = process.argv.slice(1);
const sources = await readSources();
const record = validateRecord(JSON.parse(readFileSync(recordPath, 'utf8')));
const item = externalCaseItem(JSON.parse(readFileSync(casePath, 'utf8')), sources.plan);
const expected = caseIdentity(item, sources.manifest);
const declared = sources.closures.providers[record.lane];
if (record.caseId !== item.id || !declared ||
    record.identity.engine.closureDigest !== declared.digest ||
    record.identity.engine.digestVersion !== sources.closures.digestVersion ||
    record.identity.sceneworks.dirty ||
    ['model', 'decoder'].some(part => Object.entries(expected[part]).some(([key, value]) =>
      record.identity[part]?.[key] !== value))) {
  throw new Error('off-plan record source, closure, catalog, or case identity is stale');
}
const rows = coverage(record);
if (!rows.length || rows.some(row => !row.covered)) {
  throw new Error(`off-plan admission underpriced or unmeasured: ${JSON.stringify(rows)}`);
}
console.log(`${record.caseId}: current off-plan identity, completed ${record.outcome.status}`);
console.log(JSON.stringify(rows));
"""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def verify_sources(app: Path, engine: Path, control: Path, app_sha: str, engine_sha: str, control_sha: str) -> dict:
    require(all(HEX40.fullmatch(sha) for sha in (app_sha, engine_sha, control_sha)),
            "exact lowercase 40-hex SHAs required")
    require(os.environ.get("GITHUB_SHA") == control_sha, "workflow control SHA differs from dispatch input")
    for root, expected, label in ((app, app_sha, "app"), (engine, engine_sha, "engine"),
                                  (control, control_sha, "control")):
        require(git(root, "rev-parse", "HEAD") == expected, f"{label} checkout SHA mismatch")
        require(not git(root, "status", "--porcelain", "--untracked-files=normal"), f"{label} checkout is dirty")
    pins = re.findall(r'SceneWorks/inference",\s*rev\s*=\s*"([0-9a-f]{40})"',
                      (app / "Cargo.toml").read_text(encoding="utf-8"))
    require(bool(pins) and all(pin == engine_sha for pin in pins),
            "app Cargo inference pins are not exact engine SHA")
    return {"app_sha": app_sha, "engine_sha": engine_sha, "control_sha": control_sha, "app_pins": pins}


def verified_runtime_policy(app: Path, engine: Path, evidence: Path,
                            supported: dict[str, str] = SUPPORTED_RUNTIME_POLICIES) -> tuple[str, str]:
    source = json.loads((evidence / "sources.json").read_text(encoding="utf-8"))
    require(isinstance(source, dict) and
            all(isinstance(source.get(key), str) for key in ("app_sha", "engine_sha", "control_sha")),
            "source manifest lacks exact checkout identities")
    verified = verify_sources(app, engine, Path(__file__).resolve().parents[2],
                              source["app_sha"], source["engine_sha"], source["control_sha"])
    require(source == verified, "source manifest differs from verified clean checkouts and app pin")
    policy = supported.get(source["engine_sha"])
    require(type(policy) is str and bool(policy),
            f"unsupported engine revision for CUDA BF16 VAE math policy: {source['engine_sha']}")
    return source["engine_sha"], policy


def prepare_cases(template_dir: Path, destination: Path, backend: str) -> dict:
    names = names_for_backend(backend)
    require(not destination.exists(), "run-owned case directory already exists")
    destination.mkdir(parents=True)
    rows = []
    for name in names:
        source = template_dir / f"{name}.json"
        require(source.is_file(), f"missing fixed case {name}")
        require(sha256(source) == CASE_SOURCE_SHA256[name], f"fixed case {name} changed")
        body = json.loads(source.read_text(encoding="utf-8"))
        tier, _, decoder, policy, _, _ = CASES[name]
        require(body.get("id") == case_id("cuda", name), f"unexpected source ID in {name}")
        require(body.get("tier") == tier and body.get("decoder") == decoder and
                body.get("computePolicy") == policy, f"unexpected fixed case fields in {name}")
        expected_ar_mode = "experimentalFp8" if name in CUDA_ONLY_CASES else None
        require(body.get("arMode") == expected_ar_mode, f"unexpected AR mode in {name}")
        body["id"] = case_id(backend, name)
        target = destination / f"{name}.json"
        target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        rows.append({"name": name, "case_id": body["id"], "source_sha256": sha256(source),
                     "run_case_sha256": sha256(target)})
    manifest = {"backend": backend, "cases": rows}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def preflight(backend: str, evidence: Path, label: str) -> dict:
    # The merged engine control owns the typed process census. Refuse if absent;
    # a generic process-name guess would weaken the shared physical-host lock.
    from yue2_precision_proof import (  # type: ignore[import-not-found]
        cuda_physical_census, metal_census, physical_busy_message, retain_cuda_physical_evidence,
        retain_reviewed_baseline,
    )

    require(backend in ("cuda", "metal"), "unsupported backend")
    if backend == "metal":
        require(os.environ.get("RUNNER_NAME") == "nax-macos-2", "Metal must run on nax-macos-2")
    evidence.mkdir(parents=True, exist_ok=True)
    baseline_files = None
    if backend == "cuda" and label == "initial":
        from yue2_cuda_idle_context import require_remaining_window  # type: ignore[import-not-found]
        _, baseline_dir = require_remaining_window(480 * 60 + 600)
        baseline_files = retain_reviewed_baseline(evidence, baseline_dir)
    census, busy = (cuda_physical_census if backend == "cuda" else metal_census)()
    physical_files = (retain_cuda_physical_evidence(evidence, label, census)
                      if backend == "cuda" and not busy else None)
    disk = shutil.disk_usage(evidence.parent).free
    memory = None
    available = None
    if backend == "metal":
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, encoding="utf-8").strip())
        stat = subprocess.check_output(["vm_stat"], text=True, encoding="utf-8")
        page = re.search(r"page size of (\d+) bytes", stat)
        counts = re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.", stat, re.M)
        require(page is not None and {key for key, _ in counts} == {"free", "inactive", "speculative"},
                "Metal free-memory probe is incomplete")
        available = int(page.group(1)) * sum(int(value) for _, value in counts)
    errors = []
    if busy:
        errors.append(physical_busy_message(census, busy, "competing physical-device processes"))
    if disk < MIN_FREE_DISK:
        errors.append(f"disk free {disk} below fresh-install/build floor {MIN_FREE_DISK}")
    if backend == "metal" and (memory < REFERENCE_PEAK or available < REFERENCE_PEAK):
        errors.append(f"physical/available memory {memory}/{available} below observed F32 reference {REFERENCE_PEAK}")
    record = {"backend": backend, "label": label, "runner": os.environ.get("RUNNER_NAME"),
              "hostname": os.environ.get("COMPUTERNAME") or subprocess.check_output(["hostname"], text=True, encoding="utf-8").strip(),
              "disk_free_bytes": disk, "physical_memory_bytes": memory, "available_memory_bytes": available,
              "minimum_disk_bytes": MIN_FREE_DISK, "observed_cpu_reference_peak_bytes": REFERENCE_PEAK,
              "census": census, "physical_files": physical_files,
              "reviewed_baseline_files": baseline_files,
              "competing_processes": busy, "admitted": not errors, "errors": errors}
    (evidence / f"preflight-{label}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    require(not errors, "; ".join(errors))
    return record


def verify_record(record_path: Path, backend: str, name: str, cuda_bf16_math_policy: str) -> dict:
    require(name in names_for_backend(backend), "unknown case/backend")
    row = json.loads(record_path.read_text(encoding="utf-8"))
    _, case_name, decoder, policy, model_dtype, vae_dtype = CASES[name]
    require(row.get("caseId") == case_id(backend, name) and row.get("backend") == backend,
            "record case/backend mismatch")
    expected_repo = "m-a-p/YuE2-Vae" + ("-legacy" if decoder == "legacy" else "")
    require(row.get("identity", {}).get("decoder", {}).get("repo") == expected_repo,
            "record decoder identity mismatch")
    require(row.get("request", {}).get("name") == case_name and
            row.get("request", {}).get("computePolicy") == policy,
            "record request name/policy mismatch")
    expected_ar_mode = "experimentalFp8" if name in CUDA_ONLY_CASES else None
    require(row.get("request", {}).get("arMode") == expected_ar_mode,
            "record AR mode mismatch")
    require(row.get("admission", {}).get("outcome") == "admitted", "profile was not admitted")
    outcome = row.get("outcome", {})
    require(outcome.get("status") == "completed", "profile did not complete")
    for key, wanted in (("engineComputePolicy", policy), ("engineModelDtype", model_dtype),
                        ("engineVaeDtype", vae_dtype)):
        require(outcome.get(key) == wanted, f"effective {key} does not match {wanted}")
    math_policy = cuda_bf16_math_policy if backend == "cuda" and policy == "bf16" else None
    if math_policy is None:
        require("engineVaeCudaBf16MathPolicy" not in outcome,
                "effective CUDA BF16 VAE math policy is present on another backend or compute policy")
    else:
        require(outcome.get("engineVaeCudaBf16MathPolicy") == math_policy,
                f"effective CUDA BF16 VAE math policy does not match {math_policy}")
    if name in CUDA_ONLY_CASES:
        require(outcome.get("engineQuantization") == "fp8", "experimental FP8 AR did not execute")
        host_bytes = row.get("admission", {}).get("estimate", {}).get("weights", {}).get("hostBytes")
        require(type(host_bytes) is int and host_bytes >= 2 * 1024 ** 3,
                "FP8 admission omitted retained BF16 AR originals")
    measured = row.get("measured", {})
    require(measured.get("peakBytes", 0) > 0, "profile has no overall measured peak")
    stages = measured.get("stages", {})
    require(all(stages.get(stage, {}).get("samples", 0) > 0 and
                stages[stage].get("peakBytes", 0) > 0 for stage in STAGES),
            "profile lacks a sampled stage")
    result = {"case_id": row["caseId"], "backend": backend, "admission": "admitted",
              "effective_compute_policy": policy, "effective_model_dtype": model_dtype,
              "effective_vae_dtype": vae_dtype, "effective_vae_cuda_bf16_math_policy": math_policy,
              "peak_bytes": measured["peakBytes"],
              "stage_samples": {stage: stages[stage]["samples"] for stage in STAGES},
              "record_sha256": sha256(record_path)}
    if name in CUDA_ONLY_CASES:
        result["effective_ar_quantization"] = "fp8"
        result["host_original_bytes"] = host_bytes
    return result


def verify_audio(profile_dir: Path, backend: str, name: str) -> dict:
    require(profile_dir.is_absolute(), "profile root must be an absolute run-owned path")
    require(name in names_for_backend(backend), "unknown audio case/backend")
    root = profile_dir.resolve(strict=True)
    candidate = profile_dir / case_id(backend, name).replace(":", "__") / "run" / "audio.wav"
    audio = candidate.resolve(strict=True)
    require(audio.is_relative_to(root) and audio.is_file(), "audio escaped the run-owned profile root")
    size = audio.stat().st_size
    require(size > 44, "completed case has empty or truncated audio")
    with audio.open("rb") as source:
        header = source.read(12)
    require(header[:4] == b"RIFF" and header[8:12] == b"WAVE", "completed case has no WAV header")
    return {"case_id": case_id(backend, name), "path": str(audio), "size_bytes": size,
            "sha256": sha256_stream(audio)}


def collect(profile_dir: Path, evidence: Path, backend: str, app: Path, engine: Path) -> dict:
    engine_sha, cuda_bf16_math_policy = verified_runtime_policy(app, engine, evidence)
    require(not (evidence / "profile").exists(), "profile receipts were already collected")
    rows = []
    for name in names_for_backend(backend):
        source = profile_dir / case_id(backend, name).replace(":", "__")
        row = verify_record(source / "record.json", backend, name, cuda_bf16_math_policy)
        row["listening_audio"] = verify_audio(profile_dir, backend, name)
        rows.append(row)
        target = evidence / "profile" / name
        target.mkdir(parents=True)
        for filename in RECEIPT_FILES:
            file = source / filename
            if file.is_file():
                shutil.copy2(file, target / filename)
        boundary = source / "boundary"
        if boundary.is_dir():
            for file in boundary.glob("*.json"):
                (target / "boundary").mkdir(exist_ok=True)
                shutil.copy2(file, target / "boundary" / file.name)
    verdict = {"backend": backend, "cases": rows, "status": "completed",
               "engine_sha": engine_sha, "expected_cuda_bf16_vae_math_policy": cuda_bf16_math_policy,
               "listening_audio": [row["listening_audio"] for row in rows]}
    scheduling = os.environ.get("YUE2_CUDA_SCHEDULING_MODE", "shared-host")
    if scheduling in {"owner-gpu0", "owner-gpu0-mac-anchor"}:
        require(backend == "cuda", "GPU0 owner scheduling cannot grade Metal")
        verdict["scheduling_acceptance"] = ("provisional-mac-anchor-chronology"
                                             if scheduling == "owner-gpu0-mac-anchor"
                                             else "provisional-holder-chronology")
        verdict["holder_chronology_file"] = "gpu0-holder-chronology.jsonl"
    (evidence / "audio-inventory.json").write_text(
        json.dumps({"backend": backend, "status": "completed", "cases": verdict["listening_audio"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (evidence / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    return verdict


def copy_partial(profile_dir: Path, evidence: Path, backend: str) -> None:
    """Retain known JSON/log receipts on a failed case, without audio or tensors."""
    for name in names_for_backend(backend):
        source = profile_dir / case_id(backend, name).replace(":", "__")
        if not source.is_dir():
            continue
        target = evidence / "partial-profile" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in RECEIPT_FILES:
            file = source / filename
            if file.is_file():
                shutil.copy2(file, target / filename)
        boundary = source / "boundary"
        if boundary.is_dir():
            for file in boundary.glob("*.json"):
                (target / "boundary").mkdir(exist_ok=True)
                shutil.copy2(file, target / "boundary" / file.name)


def run_captures(app: Path, engine: Path, data: Path, output: Path, evidence: Path,
                 cases: Path, backend: str) -> dict:
    _, cuda_bf16_math_policy = verified_runtime_policy(app, engine, evidence)
    require(not output.exists(), "profile output already exists")
    output.mkdir(parents=True)
    evidence.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    scheduling = environment.get("YUE2_CUDA_SCHEDULING_MODE", "shared-host")
    require(scheduling in {"shared-host", "owner-gpu0", "owner-gpu0-mac-anchor"} and
            (scheduling == "shared-host" or backend == "cuda"), "invalid CUDA owner route")
    environment.pop("HF_HUB_CACHE", None)
    environment.pop("HUGGINGFACE_HUB_CACHE", None)
    environment.pop("GH_TOKEN", None)
    environment.pop("GITHUB_TOKEN", None)
    manifest = json.loads((cases / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("backend") == backend and
            [row.get("name") for row in manifest.get("cases", [])] == list(names_for_backend(backend)),
            "run-owned case manifest/backend mismatch")
    shutil.copy2(cases / "manifest.json", evidence / "cases-manifest.json")
    completed_audio = []
    try:
        for row in manifest["cases"]:
            name = row["name"]
            case = cases / f"{name}.json"
            require(case.is_file(), f"missing run-owned case {name}")
            require(sha256(case) == row.get("run_case_sha256"), f"run-owned case {name} changed")
            require(json.loads(case.read_text(encoding="utf-8")).get("id") == case_id(backend, name),
                    f"run-owned case {name} has wrong backend")
            preflight(backend, evidence, f"before-{name}")
            command = ["node", "scripts/yue2-memory-profile.mjs", "capture", "--case-file", str(case),
                       "--inference-repo", str(engine), "--data-dir", str(data), "--gpu-id", "0",
                       "--out", str(output)]
            if backend == "metal":
                command.extend(("--budget-minutes", "120"))
            for label, argv in (("dry-run", [*command, "--dry-run"]), ("capture", command)):
                with (evidence / f"{name}-{label}.log").open("w", encoding="utf-8") as log:
                    if scheduling in {"owner-gpu0", "owner-gpu0-mac-anchor"}:
                        from yue2_gpu0_owner_guard import guarded_command
                        status = guarded_command(argv, app, environment, log, evidence, f"{name}-{label}")
                    else:
                        status = subprocess.run(argv, cwd=app, env=environment,
                                                stdout=log, stderr=subprocess.STDOUT, check=False).returncode
                require(status == 0, f"{name} {label} exited {status}; see retained log")
            record = output / case_id(backend, name).replace(":", "__") / "record.json"
            verify_record(record, backend, name, cuda_bf16_math_policy)
            with (evidence / f"{name}-check.log").open("w", encoding="utf-8") as log:
                status = subprocess.run(["node", "--input-type=module", "-e", OFF_PLAN_CHECK,
                                         str(record), str(case)],
                                        cwd=app, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, check=False).returncode
            require(status == 0, f"{name} off-plan closure/currency check exited {status}")
            completed_audio.append(verify_audio(output, backend, name))
            (evidence / "audio-inventory.json").write_text(
                json.dumps({"backend": backend, "status": "partial", "cases": completed_audio}, indent=2) + "\n",
                encoding="utf-8",
            )
        preflight(backend, evidence, "after-cases")
        return collect(output, evidence, backend, app, engine)
    finally:
        copy_partial(output, evidence, backend)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-sources")
    for field in ("app", "engine", "control", "app-sha", "engine-sha", "control-sha", "output"):
        verify.add_argument(f"--{field}", required=True)
    prepare = sub.add_parser("prepare-cases")
    for field in ("templates", "destination", "backend"):
        prepare.add_argument(f"--{field}", required=True)
    flight = sub.add_parser("preflight")
    for field in ("backend", "evidence", "label"):
        flight.add_argument(f"--{field}", required=True)
    receipts = sub.add_parser("collect")
    for field in ("profile", "evidence", "backend", "app", "engine"):
        receipts.add_argument(f"--{field}", required=True)
    captures = sub.add_parser("run-captures")
    for field in ("app", "engine", "data", "output", "evidence", "cases", "backend"):
        captures.add_argument(f"--{field}", required=True)
    args = parser.parse_args()
    if args.command == "verify-sources":
        result = verify_sources(Path(args.app), Path(args.engine), Path(args.control),
                                args.app_sha, args.engine_sha, args.control_sha)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "prepare-cases":
        result = prepare_cases(Path(args.templates), Path(args.destination), args.backend)
    elif args.command == "preflight":
        result = preflight(args.backend, Path(args.evidence), args.label)
    elif args.command == "collect":
        result = collect(Path(args.profile), Path(args.evidence), args.backend,
                         Path(args.app), Path(args.engine))
    else:
        result = run_captures(Path(args.app), Path(args.engine), Path(args.data),
                              Path(args.output), Path(args.evidence), Path(args.cases), args.backend)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"yue2-app-precision-profile: {error}", file=sys.stderr)
        raise SystemExit(1)
