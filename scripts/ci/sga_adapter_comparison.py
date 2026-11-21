#!/usr/bin/env python3
"""Explicit same-job Tier 1 adapter comparison against the integrated ENG-4a baseline."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import flux1_comparison as paired
from scripts.ci import model_inference_proofs as proof
from scripts.ci import wan_adapter_tier1_fixture as wan
from scripts.ci.flux1_pair.fixture import sha
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import load_policy

BASE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294"
PROFILES = {wan.PROFILE: wan}
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
    if (backend != "mlx" or not isinstance(value, dict)
            or set(value) != {"selection", "profile", "runner", "download_missing"}
            or value["selection"] != "sga-adapter-tier1" or value["runner"] != "nax"
            or value["download_missing"] is not True or not isinstance(value["profile"], str)
            or value["profile"] not in PROFILES):
        raise ValueError("select one registered MLX adapter profile with runner=nax and download_missing=true")
    return PROFILES[value["profile"]]


def plan(backend, value, head):
    profile = request_value(backend, value)
    source = proof.repository_identity(head)
    if head == BASE or subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head],
                                      cwd=ROOT, timeout=30).returncode != 0:
        raise ValueError("candidate must descend from the exact ENG-4a baseline")
    with TemporaryDirectory(prefix="sga-adapter-policy-") as directory:
        policy_path = Path(directory) / "policy.json"
        proof.write_json(policy_path, POLICY)
        load_policy(policy_path)
    baseline_tree = paired.git(ROOT, "rev-parse", BASE + "^{tree}")
    return {"schema": 1, "protocol": "sga-adapter-tier1-v1", "runner": proof.NAX_ROUTES["nax"]["labels"],
            "asset_setup": "sga-adapter-comparison", "backend": backend, "request": value,
            "base": {"revision": BASE, "tree": baseline_tree}, "head": source,
            "policy": POLICY, "build_timeout_seconds_per_arm": BUILD_TIMEOUT_SECONDS,
            "compared": "bare and per-expert component velocities; timing and peaks are diagnostic only",
            **profile.contract()}


def build(work, origin, arm, expected, profile, env):
    root = work / arm
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(origin), str(root)],
                   check=True, timeout=90)
    subprocess.run(["git", "checkout", "--quiet", "--detach", expected["revision"]], cwd=root,
                   check=True, timeout=90)
    observed = {"revision": paired.git(root, "rev-parse", "HEAD"),
                "tree": paired.git(root, "rev-parse", "HEAD^{tree}")}
    if observed != expected or paired.git(root, "status", "--porcelain"):
        raise ValueError("comparison build checkout does not match its exact clean Git tree")
    new_binary = root / "crates/media/mlx-gen" / profile.PACKAGE / "src/bin" / (PROOF_BIN + ".rs")
    new_binary.parent.mkdir(exist_ok=True)
    proof.unlinked_path(new_binary, root)
    with new_binary.open("xb") as destination:
        destination.write(profile.PROBE.read_bytes())
    toolchain = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--locked", "-p", profile.PACKAGE, "--bin", PROOF_BIN, "--no-default-features"]
    subprocess.run(command, env=env, cwd=root, check=True, timeout=BUILD_TIMEOUT_SECONDS)
    binary = target / "debug" / PROOF_BIN
    return binary, {**observed, "probe_sha256": sha(new_binary), "toolchain": toolchain,
                    "build_command": command, "binary_sha256": sha(binary)}


def compiler_identity(build_info):
    toolchain = build_info["toolchain"]
    linker = toolchain["native_linker"]
    return (toolchain["rustc_version"], toolchain["cargo_version"], linker["linker_sha256"],
            linker["sdk"], linker["rustflags"])


def log_limits():
    limits = {**paired.retained_log_limits(POLICY), "result.json": RESULT_LIMIT}
    return {f"{arm}.{name}": cap for arm in ARMS for name, cap in limits.items()}


def publish_arm(arm, run, work, artifact):
    limits = log_limits()
    for name, cap in limits.items():
        owner, suffix = name.split(".", 1)
        if owner != arm:
            continue
        src = run / suffix
        if not src.exists() and not proof.linked(src):
            continue
        yield name, proof.publish_file(src, artifact / name, work, cap, log=suffix != "result.json",
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


def validate_guard(row):
    if (type(row.get("returncode")) is not int or row["returncode"] != 0
            or type(row.get("pid")) is not int or row["pid"] <= 0
            or type(row.get("elapsed_seconds")) not in (int, float)
            or not 0 < row["elapsed_seconds"] <= POLICY["deadlineSeconds"]
            or type(row.get("peak_host_bytes")) is not int
            or not 0 < row["peak_host_bytes"] <= POLICY["childFootprintCapBytes"]
            or row.get("peak_gpu_bytes") is not None):
        raise ValueError("Wan child did not complete within its fixed external unified-memory guard")


def execute(args, contract):
    env = os.environ.copy()
    proof.require_host(args.backend, env, "nax")
    if env.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("adapter comparison requires its secondary MLX runner label")
    identity = attempt_identity(env)
    tmp = Path(env["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    profile = PROFILES[contract["request"]["profile"]]
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
        proof.write_json(policy_path, POLICY)
        policy = load_policy(policy_path)
        receipt["hardware"] = paired.hardware(args.backend, policy)
        env.update(CARGO_INCREMENTAL="0", PAIR_SGA_FIXTURE=str(fixture), MACOSX_DEPLOYMENT_TARGET="15.0")
        executables = {}
        for arm, source in zip(ARMS, (baseline, ROOT), strict=True):
            arm_env = env.copy()
            binary, receipt["arms"][arm] = build(work, source, arm, contract[arm], profile, arm_env)
            executables[arm] = (binary, arm_env)
        if compiler_identity(receipt["arms"]["base"]) != compiler_identity(receipt["arms"]["head"]):
            raise ValueError("comparison revisions require the same compiler identity")
        claim_dir = work / "claim"
        claim_dir.mkdir()
        observations = {}
        with execution_claim(args.backend, policy, claim_dir, env) as claim:
            receipt["physical_claim"] = claim
            for arm in ARMS:
                binary, arm_env = executables[arm]
                run = work / (arm + "-run")
                try:
                    guard, stdout = paired.invoke(binary, work / arm, run, arm_env, policy, env)
                    validate_guard(asdict(guard))
                    result = bounded_owned_file(run / "result.json", work, RESULT_LIMIT)
                    observations[arm] = profile.observe(stdout, result)
                finally:
                    receipt["files"].update(publish_arm(arm, run, work, artifact))
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
    limits = log_limits()
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
    profile = PROFILES[contract["request"]["profile"]]
    if (row["physical_claim"].get("post_execution_verified") is not True
            or set(row["fixture_sha256"]) != profile.FIXTURE_FILES
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in row["fixture_sha256"].values())):
        raise ValueError("accepted comparison lacks verified physical release or fixture digests")
    proof_rows = []
    for arm in ARMS:
        build_info = row["arms"][arm]
        if (any(build_info[key] != value for key, value in contract[arm].items())
                or build_info["probe_sha256"] != contract["probe_sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}", build_info["binary_sha256"])):
            raise ValueError("compiled proof source differs from the comparison plan")
        if not {arm + "." + suffix for suffix in ("stdout.log", "stderr.log", "result.json", "supervisor.json")} <= files.keys():
            raise ValueError("accepted comparison lacks complete guarded child evidence")
        validate_guard(json.loads((args.output / f"{arm}.supervisor.json").read_text(encoding="utf-8")))
        proof_rows.append(profile.observe((args.output / f"{arm}.stdout.log").read_text(encoding="utf-8"),
                                          args.output / f"{arm}.result.json"))
    if (compiler_identity(row["arms"]["base"]) != compiler_identity(row["arms"]["head"])
            or proof_rows[0] != proof_rows[1]):
        raise ValueError("retained Wan model outputs fail the frozen bit-exact bound")


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
