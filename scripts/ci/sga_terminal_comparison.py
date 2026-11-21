#!/usr/bin/env python3
"""Explicit source-bound real-weight comparisons on one physically claimed device."""

import argparse
from dataclasses import asdict
import importlib
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
from scripts.ci import model_inference_proofs as proof
from scripts.ci import sga_adapter_comparison as source_pair
from scripts.ci.flux1_pair.fixture import sha
from scripts.ci.model_proof_resources import execution_claim
from scripts.ci.sga_paired_performance import PAIRS, compare_peaks, compare_timing
from scripts.ci.stage_pinned_hub_snapshots import hub_root
from scripts.ci.wan_adapter_tier1_fixture import unique_object
from scripts.media_campaign_supervisor import load_policy

SELECTION = "sga-terminal-tier2"
PROFILES = {("candle-cuda", "ltx25"): "ltx25_tier2_fixture",
            ("mlx", "ltx25"): "ltx25_tier2_mlx",
            ("candle-cuda", "z-image"): "z_image_tier2_fixture",
            ("mlx", "wan"): "wan_adapter_tier2_fixture",
            ("candle-cuda", "wan"): "wan_adapter_tier2_fixture"}
DEVICE_KEY = {"candle-cuda": "cuda_device_uuid", "mlx": "runner"}
ARMS = source_pair.ARMS
ORDER = (ARMS, tuple(reversed(ARMS)))
SAMPLES = [(f"warmup-{arm}", arm) for arm in ARMS] + [
    (f"pair-{index+1}-{arm}", arm) for index in range(PAIRS) for arm in ORDER[index % len(ORDER)]
]
PREFLIGHT_SECONDS = 180


def profile_for(backend, request):
    binding = DEVICE_KEY.get(backend)
    if (binding is None or not isinstance(request, dict)
            or set(request) != {"selection", "profile", "variant", binding, "download_missing"}
            or request["selection"] != SELECTION or request["download_missing"] is not True
            or not isinstance(request["profile"], str) or (backend, request["profile"]) not in PROFILES
            or not isinstance(request[binding], str)):
        raise ValueError("select one registered native Tier 2 route with its device binding")
    if (backend == "mlx" and request[binding] != "nax") or (backend == "candle-cuda" and not
            re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", request[binding])):
        raise ValueError("terminal route requires the secondary Mac or an exact physical CUDA UUID")
    module = importlib.import_module("scripts.ci." + PROFILES[backend, request["profile"]])
    variant = request["variant"]
    if (not isinstance(variant, str) or variant not in module.CASE
            or getattr(module, "VARIANT_BACKENDS", {}).get(variant, backend) != backend):
        raise ValueError("incorrect declared native Tier 2 variant")
    return module


def plan(backend, request, head):
    profile = profile_for(backend, request)
    candidate = proof.repository_identity(head)
    base = source_pair.BASE
    if head == base or subprocess.run(["git", "merge-base", "--is-ancestor", base, head],
                                      cwd=ROOT, timeout=30).returncode != 0:
        raise ValueError("terminal head must descend from the exact ENG-4a baseline")
    binding = request[DEVICE_KEY[backend]]
    policy = profile.policy(request["variant"], binding)
    with TemporaryDirectory(prefix="sga-terminal-policy-") as directory:
        checked_policy = Path(directory) / "policy.json"
        proof.write_json(checked_policy, policy)
        load_policy(checked_policy)
    overlays = {}
    for arm, revision in zip(ARMS, (base, head), strict=True):
        overlay = profile.overlay(lambda path: subprocess.check_output(["git", "show", f"{revision}:{path}"],
                                  cwd=ROOT, timeout=30), arm, backend, source_pair.PROOF_BIN)
        overlays[arm] = source_pair.overlay_identity(overlay)
    runner = proof.NAX_ROUTES[binding]["labels"] if backend == "mlx" else proof.RUNNERS[backend]
    return {"protocol": SELECTION, "schema": 1, "runner": runner, "backend": backend,
            "asset_setup": "sga-adapter-comparison", "request": request, "policy": policy,
            "base": {"revision": base, "tree": paired.git(ROOT, "rev-parse", base + "^{tree}")},
            "head": candidate, "overlay_sha256": overlays, "expected_samples": [name for name, _ in SAMPLES],
            **profile.contract(request["variant"])}


def sample_limits(profile, policy):
    limits = {**paired.retained_log_limits(policy), "result.json": source_pair.RESULT_LIMIT,
              **getattr(profile, "ARTIFACT_LIMITS", {})}
    if policy["backend"] == "windows-cuda":
        limits["cuda-pool.json"] = 4096
    return limits


def collect_sample(name, root, limits, out, work):
    for filename, cap in limits.items():
        source = root / filename
        if source.exists() or proof.linked(source):
            target = f"{name}.{filename}"
            yield target, proof.publish_file(source, out / target, work, cap,
                                             log=filename.endswith((".log", ".jsonl")), destination_root=out)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def read_sample(name, directory, profile, contract, *, upload=False):
    def file(leaf):
        return directory / (f"{name}.{leaf}" if upload else leaf)

    guard = load_json(file("supervisor.json"))
    source_pair.validate_guard(guard, contract["policy"])
    peaks = {"host_process_bytes": guard["peak_host_bytes"]}
    if contract["backend"] == "candle-cuda":
        pool = load_json(file("cuda-pool.json"))
        source_pair.validate_cuda_pool(file("cuda-pool.json"), guard, contract["policy"])
        peaks.update(cuda_process_bytes=guard["peak_gpu_bytes"], cuda_pool_used_bytes=pool["poolUsedHighBytes"],
                     cuda_pool_reserved_bytes=pool["poolReservedHighBytes"])
    req = contract["request"]
    readback = profile.observe(file("stdout.log").read_text(encoding="utf-8"), file("result.json"),
                               req[DEVICE_KEY[contract["backend"]]], req["variant"])
    measured_peaks = readback.get("peaks", {})
    if peaks.keys() & measured_peaks.keys():
        raise ValueError("terminal profile attempted to override an external peak observation")
    peaks.update(measured_peaks)
    return {**readback, "peaks": peaks}


def summarize(samples, preflight=None):
    if list(samples) != [name for name, _ in SAMPLES]:
        raise ValueError("terminal decision requires every frozen sample in exact execution order")
    reference = samples["warmup-base"]
    if "plan" in reference["identity"] and reference["identity"]["plan"] != preflight:
        raise ValueError("terminal runtime changed the compiled preflight plan")
    if "e2e_ns" not in reference["timing"] or any(
           row["identity"] != reference["identity"] or row["timing"].keys() != reference["timing"].keys()
           for row in samples.values()):
        raise ValueError("terminal output or metric surface differed across paired and warmup samples")
    metrics = {}
    for domain, compare in (("timing", compare_timing), ("peaks", compare_peaks)):
        metrics[domain] = {}
        for field in sorted(reference[domain]):
            base, head = ([samples[f"pair-{i}-{arm}"][domain][field] for i in range(1, PAIRS + 1)] for arm in ARMS)
            metrics[domain][field] = compare(base, head)
    statuses = {item["status"] for domain in metrics.values() for item in domain.values()}
    status = "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    return {"status": status, "fidelity": reference["identity"], **metrics}


def require_sm120(uuid):
    from scripts.media_campaign_windows import trusted_nvidia_smi

    capability = subprocess.check_output([trusted_nvidia_smi(), "--id=" + uuid, "--query-gpu=compute_cap",
                                          "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    if capability != "12.0":
        raise ValueError("terminal CUDA comparison requires the approved consumer Blackwell capability")
    return "sm_120"


def execute(args, contract, profile):
    env = os.environ.copy()
    cuda = args.backend == "candle-cuda"
    proof.require_host(args.backend, env, "hosted" if cuda else "nax")
    if env.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("terminal comparisons require the self-hosted real-weights lane")
    identity = source_pair.attempt_identity(env)
    tmp = Path(env["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    receipt = {**contract, **identity, "accepted": False, "arms": {}, "files": {}}
    proof.write_json(artifact / "receipt.json", receipt)
    work = tmp / f"sga-terminal-{identity['run_id']}-{identity['run_attempt']}"
    baseline = ROOT / "sga-adapter-baseline"
    proof.unlinked_path(work, tmp)
    work.mkdir()
    try:
        request = contract["request"]
        policy_path = work / "policy.json"
        proof.write_json(policy_path, contract["policy"])
        policy = load_policy(policy_path)
        binding, variant = request[DEVICE_KEY[args.backend]], request["variant"]
        receipt["hardware"] = paired.hardware(args.backend, policy)
        if cuda:
            receipt["hardware"]["capability"] = require_sm120(binding)
        model_root = env.get(getattr(profile, "HUB_ENV", "MODEL_PROOF_HUB_ROOT")) or env.get("MODEL_PROOF_HUB_ROOT", "")
        model_hub = hub_root(dict(env, MODEL_PROOF_HUB_ROOT=model_root))
        snapshot, receipt["assets"] = profile.staged_assets(variant, model_hub)
        child_env = {key: value for key, value in env.items() if not proof.SENSITIVE_ENVIRONMENT.search(key)}
        child_env.update(CARGO_INCREMENTAL="0", PAIR_TIER2_VARIANT=variant, PAIR_TIER2_SNAPSHOT=str(snapshot))
        if hasattr(profile, "asset_environment"):
            child_env.update(profile.asset_environment(variant, model_hub))
        if cuda:
            child_env.update(CUDA_VISIBLE_DEVICES=binding, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                             SCENEWORKS_CUDA_PROOF_POOL_BYTES=str(contract["settings"]["pool_cap_bytes"]))
        executables, preflight_rows = {}, []
        for arm, origin in zip(ARMS, (baseline, ROOT), strict=True):
            arm_env = child_env.copy()
            binary, receipt["arms"][arm] = source_pair.build(work, origin, arm, contract, profile, arm_env,
                                                              release=True, cuda_features=getattr(profile, "BUILD_FEATURES", "cuda"))
            executables[arm] = (binary, arm_env)
            preflight = subprocess.run([str(binary), "--preflight"], cwd=work / arm, text=True, check=True,
                                       env=dict(arm_env, CUDA_VISIBLE_DEVICES=""), capture_output=True,
                                       timeout=PREFLIGHT_SECONDS)
            if len(preflight.stdout.encode("utf-8")) > source_pair.RESULT_LIMIT:
                raise ValueError("terminal constructor preflight exceeded its retained metadata limit")
            preflight_rows.append(profile.preflight(preflight.stdout, variant))
            receipt["arms"][arm]["preflight_stdout"] = preflight.stdout
        if preflight_rows[0] != preflight_rows[1]:
            raise ValueError("terminal source arms resolved different preflight metadata")
        if source_pair.compiler_identity(receipt["arms"]["base"]) != source_pair.compiler_identity(receipt["arms"]["head"]):
            raise ValueError("terminal source arms have different compiler identities")
        claim_dir = work / "claim"
        claim_dir.mkdir()
        claim_policy, claim_deadline, last_child_start = policy, None, None
        if cuda:
            reserve = paired.RELEASE_RESERVE_SECONDS + (policy.term_grace_millis + policy.poll_millis) / 1000
            budget = (len(SAMPLES) + 1) * policy.deadline_seconds + reserve
            claim_file = claim_dir / "policy.json"
            proof.write_json(claim_file, dict(contract["policy"], deadlineSeconds=budget))
            claim_policy = load_policy(claim_file)
            claim_deadline = time.monotonic() + budget
            last_child_start = claim_deadline - reserve - policy.deadline_seconds
        observations = {}
        limits = sample_limits(profile, contract["policy"])
        with execution_claim(args.backend, claim_policy, claim_dir, env,
                             original_deadline_monotonic=claim_deadline) as claim:
            receipt["physical_claim"] = claim
            for name, arm in SAMPLES:
                binary, arm_env = executables[arm]
                run = work / name
                try:
                    paired.invoke(binary, work / arm, run, arm_env, policy, env, last_child_start=last_child_start)
                    for leaf, cap in limits.items():
                        if leaf != "guard.jsonl":
                            source_pair.bounded_owned_file(run / leaf, work, cap)
                    observations[name] = read_sample(name, run, profile, contract)
                finally:
                    receipt["files"].update(collect_sample(name, run, limits, artifact, work))
        if (claim.get("post_execution_verified") is not True
                or profile.staged_assets(variant, model_hub, stage=False) != (snapshot, receipt["assets"])):
            raise ValueError("terminal physical release or staged model asset identity changed")
        receipt["comparison_result"] = summarize(observations, preflight_rows[0])
        if receipt["comparison_result"]["status"] != "pass":
            raise ValueError("terminal performance decision is " + receipt["comparison_result"]["status"])
        receipt["accepted"] = True
    except BaseException as error:
        receipt["failure"] = proof.redact(str(error), env)[:4096]
        raise
    finally:
        try:
            paired.remove_source_worktrees(work, {"base": baseline, "head": ROOT})
            paired.remove_work(work, tmp)
        except (OSError, ValueError, subprocess.SubprocessError):
            receipt["accepted"] = False
            receipt["cleanup_failed"] = True
            raise
        finally:
            proof.write_json(artifact / "receipt.json", receipt)


def verify(args, contract, profile):
    tmp = Path(os.environ["RUNNER_TEMP"])
    receipt = load_json(source_pair.bounded_owned_file(args.output / "receipt.json", tmp, source_pair.MIB))
    if (any(receipt.get(key) != value for key, value in {**contract, **source_pair.attempt_identity(os.environ)}.items())
            or type(receipt.get("accepted")) is not bool):
        raise ValueError("terminal receipt differs from the current exact plan or attempt")
    limits = sample_limits(profile, contract["policy"])
    allowed = {f"{name}.{leaf}": cap for name, _ in SAMPLES for leaf, cap in limits.items()}
    if {item.name for item in args.output.iterdir()} != {"receipt.json", *receipt["files"]}:
        raise ValueError("unexpected terminal upload files")
    for name, item in receipt["files"].items():
        if name not in allowed or item["path"] != name:
            raise ValueError("unexpected terminal artifact position")
        path = source_pair.bounded_owned_file(args.output / name, tmp, allowed[name])
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("terminal upload artifact changed")
    if not receipt["accepted"]:
        return
    if "expected_asset_identity" in contract and receipt.get("assets") != contract["expected_asset_identity"]:
        raise ValueError("accepted terminal receipt lacks the frozen asset content identities")
    required = set(allowed) - {f"{name}.guard.jsonl" for name, _ in SAMPLES}
    if not required <= receipt["files"].keys() or receipt["physical_claim"].get("post_execution_verified") is not True:
        raise ValueError("accepted terminal receipt lacks exact supervised sample evidence")
    preflight_rows = []
    for arm in ARMS:
        build = receipt["arms"][arm]
        if (any(build.get(key) != value for key, value in contract[arm].items())
                or build["overlay_sha256"] != contract["overlay_sha256"][arm]
                or not re.fullmatch(r"[0-9a-f]{64}", build["binary_sha256"])):
            raise ValueError("terminal binary evidence differs from the exact compiled source")
        preflight_rows.append(profile.preflight(build["preflight_stdout"], contract["request"]["variant"]))
    if preflight_rows[0] != preflight_rows[1]:
        raise ValueError("retained terminal source arms resolved different preflight metadata")
    if source_pair.compiler_identity(receipt["arms"]["base"]) != source_pair.compiler_identity(receipt["arms"]["head"]):
        raise ValueError("terminal artifact compiler identity changed")
    observations = {name: read_sample(name, args.output, profile, contract, upload=True) for name, _ in SAMPLES}
    verified = summarize(observations, preflight_rows[0])
    if verified != receipt["comparison_result"] or verified["status"] != "pass":
        raise ValueError("terminal upload does not reproduce its acceptance decision")


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
    request = json.loads(os.environ[args.request_env], object_pairs_hook=unique_object)
    profile = profile_for(args.backend, request)
    contract = plan(args.backend, request, args.expected_sha)
    if args.plan:
        print(json.dumps(contract, sort_keys=True))
    elif args.output is None:
        parser.error("a fresh attempt-owned proof directory is required")
    elif args.verify_artifacts:
        verify(args, contract, profile)
    else:
        execute(args, contract, profile)


if __name__ == "__main__":
    main()
