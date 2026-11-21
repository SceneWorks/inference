#!/usr/bin/env python3
"""Explicit same-job Tier 1 adapter comparison against the integrated ENG-4a baseline."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import flux1_comparison as paired
from scripts.ci import ltx_adapter_tier1_fixture as ltx
from scripts.ci import model_inference_proofs as proof
from scripts.ci import wan_adapter_tier1_fixture as wan
from scripts.ci.flux1_pair.fixture import sha
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import load_policy

BASE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294"
PROFILES = {("mlx", wan.PROFILE): wan, ("mlx", ltx.PROFILE): ltx, ("candle-cuda", ltx.PROFILE): ltx}
ARMS = ("base", "head")
GIB = 2**30
MIB = 2**20
POLICY = {"schemaVersion": 1, "backend": "darwin-mlx", "deadlineSeconds": 300,
          "pollMillis": 250, "termGraceMillis": 5000, "hostFreeReserveBytes": 4 * GIB,
          "childFootprintCapBytes": 16 * GIB, "stdoutCapBytes": MIB, "stderrCapBytes": MIB,
          "eventCapBytes": 4 * MIB}
BUILD_TIMEOUT_SECONDS = 2400
RESULT_LIMIT = 64 * 1024
PROOF_BIN = "sw-sga-adapter-pair"


def request_value(backend, value):
    keys = {"selection", "profile", "download_missing"}
    route_key = "runner" if backend == "mlx" else "cuda_device_uuid"
    if (not isinstance(value, dict) or set(value) != keys | {route_key}
            or value["selection"] != "sga-adapter-tier1" or value["download_missing"] is not True
            or not isinstance(value["profile"], str) or (backend, value["profile"]) not in PROFILES):
        raise ValueError("select one registered backend/adapter profile with download_missing=true")
    if backend == "mlx" and value["runner"] != "nax":
        raise ValueError("adapter comparison requires its secondary MLX runner label")
    if backend == "candle-cuda" and (not isinstance(value["cuda_device_uuid"], str) or not re.fullmatch(
            r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["cuda_device_uuid"]
    )):
        raise ValueError("adapter comparison requires the exact physical CUDA device UUID")
    return PROFILES[backend, value["profile"]]


def policy_value(backend, request):
    if backend == "mlx":
        return POLICY
    return dict(POLICY, backend="windows-cuda", cudaDeviceUuid=request["cuda_device_uuid"],
                gpuFreeReserveBytes=2 * GIB, childGpuCapBytes=12 * GIB)


def plan(backend, value, head):
    profile = request_value(backend, value)
    source = proof.repository_identity(head)
    if head == BASE or subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head],
                                      cwd=ROOT, timeout=30).returncode != 0:
        raise ValueError("candidate must descend from the exact ENG-4a baseline")
    with TemporaryDirectory(prefix="sga-adapter-policy-") as directory:
        policy_path = Path(directory) / "policy.json"
        policy = policy_value(backend, value)
        proof.write_json(policy_path, policy)
        load_policy(policy_path)
    baseline_tree = paired.git(ROOT, "rev-parse", BASE + "^{tree}")
    runner = proof.NAX_ROUTES["nax"]["labels"] if backend == "mlx" else proof.RUNNERS[backend]
    probes = {}
    for arm, revision in zip(ARMS, (BASE, head), strict=True):
        overlay = profile.overlay(lambda path: subprocess.check_output(["git", "show", f"{revision}:{path}"],
                                  cwd=ROOT, timeout=30), arm, backend, PROOF_BIN)
        probes[arm] = overlay_identity(overlay)
    return {"schema": 1, "protocol": "sga-adapter-tier1-v1", "runner": runner,
            "asset_setup": "sga-adapter-comparison", "backend": backend, "request": value,
            "base": {"revision": BASE, "tree": baseline_tree}, "head": source,
            "policy": policy, "build_timeout_seconds_per_arm": BUILD_TIMEOUT_SECONDS,
            "overlay_sha256": probes, **profile.contract(backend)}


def overlay_identity(overlay):
    return {name: hashlib.sha256(data).hexdigest() for name, data in sorted(overlay.items())}


def build(work, origin, arm, contract, profile, env):
    root = work / arm
    expected = contract[arm]
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(origin), str(root)],
                   check=True, timeout=90)
    subprocess.run(["git", "checkout", "--quiet", "--detach", expected["revision"]], cwd=root,
                   check=True, timeout=90)
    observed = {"revision": paired.git(root, "rev-parse", "HEAD"),
                "tree": paired.git(root, "rev-parse", "HEAD^{tree}")}
    if observed != expected or paired.git(root, "status", "--porcelain"):
        raise ValueError("comparison build checkout does not match its exact clean Git tree")
    backend = contract["backend"]
    overlay = profile.overlay(lambda path: subprocess.check_output(["git", "show", f"HEAD:{path}"],
                              cwd=root, timeout=30), arm, backend, PROOF_BIN)
    if overlay_identity(overlay) != contract["overlay_sha256"][arm]:
        raise ValueError("adapter probe overlay differs from the hosted plan")
    for name, data in overlay.items():
        destination = root / name
        proof.unlinked_path(destination, root)
        destination.parent.mkdir(exist_ok=True)
        mode = "xb" if destination.name == PROOF_BIN + ".rs" else "wb"
        with destination.open(mode) as output:
            output.write(data)
    toolchain = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--locked", "-p", profile.package(backend), "--bin", PROOF_BIN, "--no-default-features"]
    if backend == "candle-cuda":
        command += ["--features", "cuda"]
    subprocess.run(command, env=env, cwd=root, check=True, timeout=BUILD_TIMEOUT_SECONDS)
    binary = target / "debug" / (PROOF_BIN + (".exe" if backend == "candle-cuda" else ""))
    return binary, {**observed, "overlay_sha256": overlay_identity(overlay), "toolchain": toolchain,
                    "build_command": command, "binary_sha256": sha(binary)}


def compiler_identity(build_info):
    toolchain = build_info["toolchain"]
    linker = toolchain.get("native_linker", {})
    return (toolchain["rustc_version"], toolchain["cargo_version"], linker.get("linker_sha256"),
            linker.get("sdk"), linker.get("rustflags"))


def log_limits(backend="mlx"):
    limits = {**paired.retained_log_limits(POLICY), "result.json": RESULT_LIMIT}
    if backend == "candle-cuda":
        limits["cuda-pool.json"] = 4096
    return {f"{arm}.{name}": cap for arm in ARMS for name, cap in limits.items()}


def publish_arm(arm, run, work, artifact, backend):
    limits = log_limits(backend)
    for name, cap in limits.items():
        owner, suffix = name.split(".", 1)
        if owner != arm:
            continue
        src = run / suffix
        if not src.exists() and not proof.linked(src):
            continue
        yield name, proof.publish_file(src, artifact / name, work, cap, log=suffix not in ("result.json", "cuda-pool.json"),
                                       destination_root=artifact)


def bounded_owned_file(path, root, limit):
    proof.unlinked_path(path, root)
    if not path.is_file() or path.lstat().st_nlink != 1 or path.stat().st_size > limit:
        raise ValueError(f"unowned, missing or oversized comparison artifact: {path.name}")
    return path


def attempt_identity(env):
    run = env.get("GITHUB_RUN_ID", "")
    attempt = env.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"[1-9][0-9]{0,18}", run) or not re.fullmatch(r"[1-9][0-9]{0,8}", attempt):
        raise ValueError("invalid GitHub proof attempt identity")
    return {"run_id": run, "run_attempt": attempt}


def validate_guard(row, policy=POLICY):
    if (type(row.get("returncode")) is not int or row["returncode"] != 0
            or type(row.get("pid")) is not int or row["pid"] <= 0
            or type(row.get("elapsed_seconds")) not in (int, float)
            or not 0 < row["elapsed_seconds"] <= policy["deadlineSeconds"]
            or type(row.get("peak_host_bytes")) is not int
            or not 0 < row["peak_host_bytes"] <= policy["childFootprintCapBytes"]):
        raise ValueError("adapter child did not complete within its fixed external supervisor budget")
    gpu_bytes = row.get("peak_gpu_bytes")
    if policy["backend"] == "windows-cuda":
        if type(gpu_bytes) is not int or not 0 < gpu_bytes <= policy["childGpuCapBytes"]:
            raise ValueError("adapter child lacks a bounded CUDA allocation observation")
    elif gpu_bytes is not None:
        raise ValueError("MLX adapter child must use the unified-memory observation")


def validate_cuda_pool(path, guard, policy):
    row = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=wan.unique_object)
    fields = ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")
    values = [row.get(field) for field in fields]
    if (row.get("cudaDeviceUuid") != policy["cudaDeviceUuid"] or row.get("pid") != guard["pid"]
            or any(type(value) is not int or not 0 < value <= policy["childGpuCapBytes"] for value in values)
            or not values[0] <= min(values[1], values[2]) or not max(values[1], values[2]) <= values[3]):
        raise ValueError("adapter CUDA allocation receipt differs from the supervised child and device")


def execute(args, contract):
    env = os.environ.copy()
    backend = args.backend
    proof.require_host(backend, env, "nax" if backend == "mlx" else "hosted")
    if env.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("adapter comparison requires the registered self-hosted proof lane")
    identity = attempt_identity(env)
    tmp = Path(env["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    profile = PROFILES[backend, contract["request"]["profile"]]
    receipt = {**contract, **identity, "accepted": False, "arms": {}, "files": {}}
    proof.write_json(artifact / "receipt.json", receipt)
    baseline = ROOT / "sga-adapter-baseline"
    work = tmp / f"sga-adapter-{receipt['run_id']}-{receipt['run_attempt']}"
    proof.unlinked_path(work, tmp)
    work.mkdir()
    try:
        if paired.git(baseline, "rev-parse", "HEAD") != BASE:
            raise ValueError("adapter comparison baseline checkout differs from the workflow pin")
        fixture = work / "fixture"
        receipt["fixture_sha256"] = profile.materialize(fixture)
        policy_path = work / "policy.json"
        proof.write_json(policy_path, contract["policy"])
        policy = load_policy(policy_path)
        receipt["hardware"] = paired.hardware(backend, policy)
        env.update(CARGO_INCREMENTAL="0", PAIR_SGA_FIXTURE=str(fixture))
        if backend == "mlx":
            env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
        else:
            env.update(CUDA_VISIBLE_DEVICES=policy.cuda_device_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", SCENEWORKS_CUDA_PROOF_POOL_BYTES=str(10 * GIB))
        executables = {}
        for arm, source in zip(ARMS, (baseline, ROOT), strict=True):
            arm_env = env.copy()
            binary, receipt["arms"][arm] = build(work, source, arm, contract, profile, arm_env)
            executables[arm] = (binary, arm_env)
        if compiler_identity(receipt["arms"]["base"]) != compiler_identity(receipt["arms"]["head"]):
            raise ValueError("comparison revisions require the same compiler identity")
        claim_dir = work / "claim"
        claim_dir.mkdir()
        observations = {}
        claim_policy, claim_deadline, last_child_start = policy, None, None
        if backend == "candle-cuda":
            reserve = paired.RELEASE_RESERVE_SECONDS + (policy.term_grace_millis + policy.poll_millis) / 1000
            budget = (len(ARMS) + 1) * policy.deadline_seconds + reserve
            proof.write_json(claim_dir / "policy.json", dict(contract["policy"], deadlineSeconds=budget))
            claim_policy = load_policy(claim_dir / "policy.json")
            claim_deadline = time.monotonic() + budget
            last_child_start = claim_deadline - reserve - policy.deadline_seconds
        with execution_claim(backend, claim_policy, claim_dir, env,
                             original_deadline_monotonic=claim_deadline) as claim:
            receipt["physical_claim"] = claim
            for arm in ARMS:
                binary, arm_env = executables[arm]
                run = work / (arm + "-run")
                try:
                    guard, stdout = paired.invoke(binary, work / arm, run, arm_env, policy, env,
                                                  last_child_start=last_child_start)
                    observation = asdict(guard)
                    validate_guard(observation, contract["policy"])
                    if backend == "candle-cuda":
                        validate_cuda_pool(bounded_owned_file(run / "cuda-pool.json", work, 4096),
                                           observation, contract["policy"])
                    result = bounded_owned_file(run / "result.json", work, RESULT_LIMIT)
                    observations[arm] = profile.observe(stdout, result, backend)
                finally:
                    receipt["files"].update(publish_arm(arm, run, work, artifact, backend))
        if (not claim.get("post_execution_verified") or profile.identify(fixture) != receipt["fixture_sha256"]
                or observations["base"] != observations["head"]):
            raise ValueError("physical release, fixture identity or bit-exact adapter comparison failed")
        receipt["accepted"] = True
    except BaseException as error:
        receipt["failure"] = proof.redact(str(error), env)[:4096]
        raise
    finally:
        try:
            paired.remove_work(work, tmp)
        except (OSError, ValueError):
            receipt["accepted"] = False
            receipt["cleanup_failed"] = True
            raise
        finally:
            proof.write_json(artifact / "receipt.json", receipt)


def verify(args, contract):
    tmp = Path(os.environ["RUNNER_TEMP"])
    path = bounded_owned_file(args.output / "receipt.json", tmp, MIB)
    row = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=wan.unique_object)
    if (not isinstance(row, dict)
            or any(row.get(key) != value for key, value in {**contract, **attempt_identity(os.environ)}.items())
            or type(row.get("accepted")) is not bool):
        raise ValueError("proof receipt does not match the current plan and attempt")
    backend = contract["backend"]
    limits = log_limits(backend)
    files = row["files"]
    if {entry.name for entry in args.output.iterdir()} != {"receipt.json", *files}:
        raise ValueError("unexpected comparison upload files")
    for name, item in files.items():
        if name not in limits or item["path"] != name:
            raise ValueError("unexpected comparison artifact name")
        file = bounded_owned_file(args.output / name, tmp, limits[name])
        if file.stat().st_size != item["bytes"] or sha(file) != item["sha256"]:
            raise ValueError("comparison artifact changed after publication")
    if not row["accepted"]:
        return
    profile = PROFILES[backend, contract["request"]["profile"]]
    if (row["physical_claim"].get("post_execution_verified") is not True
            or set(row["fixture_sha256"]) != profile.FIXTURE_FILES
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in row["fixture_sha256"].values())):
        raise ValueError("accepted comparison lacks verified physical release or fixture digests")
    proof_rows = []
    for arm in ARMS:
        build_info = row["arms"][arm]
        if (any(build_info[key] != value for key, value in contract[arm].items())
                or build_info["overlay_sha256"] != contract["overlay_sha256"][arm]
                or not re.fullmatch(r"[0-9a-f]{64}", build_info["binary_sha256"])):
            raise ValueError("compiled proof source differs from the comparison plan")
        if not {arm + "." + suffix for suffix in ("stdout.log", "stderr.log", "result.json", "supervisor.json")} <= files.keys():
            raise ValueError("accepted comparison lacks complete guarded child evidence")
        guard = json.loads((args.output / f"{arm}.supervisor.json").read_text(encoding="utf-8"))
        validate_guard(guard, contract["policy"])
        if backend == "candle-cuda":
            if f"{arm}.cuda-pool.json" not in files:
                raise ValueError("accepted comparison is missing its owned CUDA allocation receipt")
            validate_cuda_pool(args.output / f"{arm}.cuda-pool.json", guard, contract["policy"])
        proof_rows.append(profile.observe((args.output / f"{arm}.stdout.log").read_text(encoding="utf-8"),
                                          args.output / f"{arm}.result.json", backend))
    if (compiler_identity(row["arms"]["base"]) != compiler_identity(row["arms"]["head"])
            or proof_rows[0] != proof_rows[1]):
        raise ValueError("retained model outputs fail the frozen bit-exact bound")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = plan(args.backend, json.loads(os.environ[args.request_env], object_pairs_hook=wan.unique_object),
                    args.expected_sha)
    if args.plan:
        print(json.dumps(contract, sort_keys=True))
    elif args.output is None:
        parser.error("a fresh attempt-owned proof artifact directory is required")
    elif args.verify_artifacts:
        verify(args, contract)
    else:
        execute(args, contract)


if __name__ == "__main__":
    main()
