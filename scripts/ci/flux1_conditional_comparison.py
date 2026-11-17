#!/usr/bin/env python3
"""Manual, one-case FLUX.1 conditional base/head proof on one physical backend."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import flux1_comparison as base
from scripts.ci import model_inference_proofs as proof
from scripts.ci.flux1_pair import conditional, conditional_evidence as evidence, conditional_probe, fixture, overlay
from scripts.ci.model_proof_resources import CudaClaimRejected, CudaReleaseRejected, execution_claim
from scripts.media_campaign_supervisor import load_policy

BASE = overlay.BASE
LANES = base.LANES
PHASES = ("preflight", "base-run", "head-run")
GIB = 1024**3
MAX_DEADLINE_SECONDS = 1200
# Explicit per-case child caps. Every case shares the frozen 24 GiB host-footprint / 18 GiB CUDA
# budget except pulid-pid, which adds the PiD 2k->4k decoder and its Gemma 2 2B caption encoder.
#
# Candle CUDA GPU: PidEngine::load materializes both checkpoints at f32 (candle-gen-pid
# `engine::LOAD_DTYPE`): the student (2.72 GB BF16 -> 5.45 GB) and Gemma (5.23 GB BF16 -> 10.46 GB),
# with a 4.99 GB BF16 + 9.98 GB f32 device transient while the first Gemma shard is coerced. The
# 1024-pixel probe decodes to 4096x4096 (`budget::estimated_decode_peak_gib`, patch 16, hidden 1536:
# 4.69 GiB untiled). PiD alone is ~19.5 GiB, matching run 43553801528's 23.1 GB above baseline;
# 40 GiB keeps ~18 GiB headroom on the 96 GB RTX PRO 6000.
#
# Candle host: the weights do NOT go straight to the device. `candle_core::safetensors::load` reads
# each whole file into a host Vec (`safetensors_progress::read_file`) before uploading its tensors,
# so the student (2.72 GB) and each Gemma shard (4.99 GB, 0.24 GB) pass through host memory one
# file at a time, plus the 4096x4096 f32 output (0.2 GB). Run 43560615425 plateaued at ~24.1 GB
# through denoise and reached 28.87 GB at PiD start while still climbing. If the allocator retains
# every freed file buffer, the peak is ~24.1 + 2.72 + 4.99 + 0.24 + 0.2 = 32.3 GB; 48 GiB (51.5 GB)
# keeps ~19 GB headroom and stays far below the Windows runner's ~516 GB available host memory.
# The pre-migration base arm hit the same climb, so this is existing loader behaviour.
#
# MLX on nax-macos-2 is unified memory: the one footprint cap covers weights, activations and the
# Metal allocator together, and there is no separate GPU cap. mlx-gen-pid keeps PiD and Gemma in
# BF16 (~8 GB) but the PuLID stack, the 4096x4096 decode and the allocator cache share the same
# pool, so pulid-pid gets the same 48 GiB footprint cap, well below the runner's 128 GiB
# (`physical_memory_bytes` 137438953472 in run 43558295383's receipt).
DEFAULT_CAPS = {"child_footprint_cap_bytes": 24 * GIB, "child_gpu_cap_bytes": 18 * GIB}
CASE_CAPS = {"pulid-pid": {"child_footprint_cap_bytes": 48 * GIB, "child_gpu_cap_bytes": 40 * GIB}}
# The watchdog samples the child's instantaneous footprint once per poll; arms run ~11 s, so a 1 s
# poll can miss the final denoise/decode peak (run 43569270221). The supervisor now also records
# each process's kernel lifetime peak, and conditional proofs poll at least every 250 ms.
MAX_POLL_MILLIS = 250
# Physical memory each conditional runner reports; a host cap must stay below it.
RUNNER_HOST_BYTES = {"mlx": 137_438_953_472, "candle-cuda": 516_136_534_016}


def case_caps(backend, case, policy):
    """Refuse a policy above its case's explicit caps; return the caps for the plan record."""
    caps = CASE_CAPS.get(case, DEFAULT_CAPS)
    if policy.deadline_seconds > MAX_DEADLINE_SECONDS:
        raise ValueError("conditional child exceeds frozen 20 minute budget")
    if policy.poll_millis > MAX_POLL_MILLIS:
        raise ValueError(f"conditional proof must poll at least every {MAX_POLL_MILLIS} ms")
    if policy.child_footprint_cap_bytes > caps["child_footprint_cap_bytes"]:
        raise ValueError(f"conditional child exceeds {case} host cap of "
                         f"{caps['child_footprint_cap_bytes'] // GIB} GiB")
    if backend == "candle-cuda" and policy.child_gpu_cap_bytes > caps["child_gpu_cap_bytes"]:
        raise ValueError(f"conditional CUDA child exceeds {case} GPU cap of "
                         f"{caps['child_gpu_cap_bytes'] // GIB} GiB")
    return {"poll_millis": policy.poll_millis,
            "child_footprint_cap_bytes": caps["child_footprint_cap_bytes"],
            "child_gpu_cap_bytes": caps["child_gpu_cap_bytes"] if backend == "candle-cuda" else None}


def plan(backend, value, head, tier):
    if tier != "conditional-tier1" or backend not in LANES or head == BASE:
        raise ValueError("select a distinct head and a conditional Tier 1 backend")
    if type(value) is not dict or set(value) != ({"selection", "case", "assets", "policy"}
            | ({"runner"} if backend == "mlx" else set())) or value["selection"] != "flux1-conditional-pair":
        raise ValueError("comparison needs exact conditional selection, case, assets and policy")
    if backend == "mlx" and value.get("runner") != "nax":
        raise ValueError("conditional MLX proof needs the secondary runner")
    variant, assets = conditional.case_metadata(value["case"], backend, value["assets"])
    if value["policy"]["backend"] != proof.POLICY_BACKENDS[backend]:
        raise ValueError("resource policy differs from conditional lane")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        base.write(path, value["policy"])
        policy = load_policy(path)
        caps = case_caps(backend, value["case"], policy)
    source = proof.repository_identity(head)
    fixture.geometry(variant)
    case = value["case"]
    request = dict(value, assets=assets)
    return {"runner": LANES[backend], "asset_setup": "flux1-comparison", "backend": backend,
            "base": BASE, "head": source["revision"], "head_tree": source["tree"],
            "tier": "tier1", "case": case, "variant": variant, "request": request,
            "geometry_sha256": fixture.sha(fixture.HERE / "geometry.tsv"),
            "reference_sha256": conditional.REFERENCE_SHA256,
            "asset_revision_provenance": "caller-declared repository/model revision; SHA-256 of staged bytes is authoritative",
            "settings": dict(evidence.SETTINGS, case=case, variant=variant,
                             guidance=3.5 if variant == "dev" else None),
            "tolerance": {"class": "AP13(a)", "states": "bit-exact", "events": "exact",
                          "rgb8": "bit-exact", "provenance": "SGA implementation plan §7.3; no adjustable limit"},
            "child_caps": caps,
            "cuda_claim_budget_seconds": cuda_claim_budget(policy)[0] if backend == "candle-cuda" else None}


def build(work, sources, arm, backend, case, env):
    root = work / arm
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(sources[arm]), str(root)], check=True)
    sha = base.git(sources[arm], "rev-parse", "HEAD")
    subprocess.run(["git", "checkout", "--quiet", "--detach", sha], cwd=root, check=True)
    if base.git(root, "rev-parse", "HEAD") != sha:
        raise ValueError("conditional source revision changed")
    package = conditional_probe.install(root, backend, arm, "tier1", case)
    family = "mlx-gen" if backend == "mlx" else "candle-gen"
    path = f"crates/media/{family}/{package}/src/bin/flux1-pair.rs"
    base.git(root, "add", "-N", path)
    delta = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    tc = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--locked", "-p", package, "--bin", "flux1-pair"]
    if backend == "candle-cuda":
        command += ["--features", "cuda"]
    subprocess.run(command, env=env, cwd=root, check=True)
    binary = target / "debug" / ("flux1-pair.exe" if os.name == "nt" else "flux1-pair")
    return binary, {"revision": sha, "tree": base.git(root, "rev-parse", "HEAD^{tree}"),
                    "overlay_sha256": hashlib.sha256(delta).hexdigest(), "toolchain": tc,
                    "binary_sha256": fixture.sha(binary), "build_command": command,
                    "macos_deployment_target": env.get("MACOSX_DEPLOYMENT_TARGET") if backend == "mlx" else None}


def cuda_claim_budget(policy):
    reserve = base.RELEASE_RESERVE_SECONDS + policy.term_grace_millis / 1000 + policy.poll_millis / 1000
    return (len(PHASES) + 1) * policy.deadline_seconds + reserve, reserve


def retain_logs(work, artifact, receipt):
    errors = []
    for phase in PHASES:
        limits = base.retained_log_limits(receipt["request"]["policy"])
        if receipt["request"]["policy"]["backend"] == "windows-cuda" and phase != "preflight":
            limits["cuda-pool.json"] = 4096
        for suffix, cap in limits.items():
            source = work / phase / suffix
            name = phase + "." + suffix
            try:
                if not source.exists() and not proof.linked(source):
                    if receipt["accepted"] and suffix != "guard.jsonl":
                        errors.append(name + ": missing completed child log")
                    continue
                receipt["logs"][name] = proof.publish_file(source, artifact / name, work, cap, log=True,
                                                           destination_root=artifact)
            except (OSError, ValueError) as error:
                errors.append(name + ": " + str(error))
    return errors


def execute(args, contract):
    backend, case, variant = args.backend, contract["case"], contract["variant"]
    proof.require_host(backend, dict(os.environ), "nax" if backend == "mlx" else "hosted")
    tmp = Path(os.environ["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    receipt = {"schema": 1, "protocol": "flux1-conditional-ap13-v1", **contract,
               "accepted": False, "run_id": os.environ["GITHUB_RUN_ID"],
               "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "host": platform.platform(),
               "runner_name": os.environ["RUNNER_NAME"], "arms": {}, "logs": {}}
    work = tmp / f"f1c-{receipt['run_id']}-{receipt['run_attempt']}"
    created_work = False
    primary_error = None
    try:
        proof.unlinked_path(work, tmp)
        work.mkdir()
        created_work = True
        sources = {"head": ROOT, "base": ROOT / "flux1-baseline"}
        if base.git(sources["base"], "rev-parse", "HEAD") != BASE:
            raise ValueError("conditional baseline is not the pre-migration engine")
        staged, receipt["assets"] = conditional.prepare_assets(work, ROOT, contract["request"]["assets"])
        if case.startswith("control-"):
            source_path, source_sha = conditional.write_control_source(work, case)
        else:
            source_path, source_sha = Path(receipt["assets"]["reference_image"]["staged"]), conditional.REFERENCE_SHA256
        receipt["source"] = {"sha256": source_sha, "role": "control" if case.startswith("control-") else "reference",
                             "path": str(source_path)}
        snapshot = work / ("random-" + variant)
        fixture.make_snapshot(snapshot, variant)
        receipt["fixture"] = fixture.snapshot_identity(snapshot, variant, "tier1")
        policy_path = work / "policy.json"
        base.write(policy_path, contract["request"]["policy"])
        policy = load_policy(policy_path)
        receipt["hardware"] = base.hardware(backend, policy)
        exes, environments = {}, {}
        for arm in evidence.ARMS:
            env = os.environ.copy()
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["CARGO_INCREMENTAL"] = "0"
            if backend == "mlx":
                env["MACOSX_DEPLOYMENT_TARGET"] = "26.2"
            else:
                env.update(CUDA_VISIBLE_DEVICES=policy.cuda_device_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            binary, info = build(work, sources, arm, backend, case, env)
            exes[arm], environments[arm] = binary, env
            receipt["arms"][arm] = {"build": info, "renders": {}}
        receipt["allocator"] = {"scope": "fresh process per revision/case on the same physical device",
                                "mlx": "default MLX unified allocator", "candle": "default CUDA allocator",
                                "cublas_workspace_config": ":4096:8"}
        base.write(artifact / "receipt.json", receipt)
        claim_dir = work / "claim"
        claim_dir.mkdir()
        claim_policy = policy
        claim_deadline = last_child_start = None
        if backend == "candle-cuda":
            budget, reserve = cuda_claim_budget(policy)
            base.write(claim_dir / "policy.json", dict(contract["request"]["policy"], deadlineSeconds=budget))
            claim_policy = load_policy(claim_dir / "policy.json")
            claim_deadline = time.monotonic() + budget
            last_child_start = claim_deadline - reserve - policy.deadline_seconds
        lease_env = environments["base"]
        with execution_claim(backend, claim_policy, claim_dir, lease_env,
                             original_deadline_monotonic=claim_deadline) as claim:
            receipt["physical_claim"] = claim
            shared = {"PAIR_CASE": case, "PAIR_SNAPSHOT": str(snapshot), "PAIR_ASSETS": str(staged),
                      "PAIR_SOURCE": str(source_path)}
            if "pid_gemma" in receipt["assets"]:
                shared["PAIR_PID_GEMMA"] = receipt["assets"]["pid_gemma"]["staged"]
            preflight = work / "preflight"
            env = dict(environments["head"], **shared)
            _, stdout = base.invoke(exes["head"], work / "head", preflight, env, policy, lease_env,
                                    preflight=True, last_child_start=last_child_start)
            plan_data = json.loads((preflight / "plan.json").read_text(encoding="utf-8"))
            evidence.validate_plan(plan_data, case, backend)
            if stdout.splitlines().count("FLUX1_CONDITIONAL_PREFLIGHT " + case) != 1:
                raise ValueError("missing conditional pre-load preflight")
            receipt["preflight"] = plan_data
            base.write(artifact / "receipt.json", receipt)
            for arm in evidence.ARMS:
                out = work / (arm + "-run")
                env = dict(environments[arm], **shared)
                guard, stdout = base.invoke(exes[arm], work / arm, out, env, policy, lease_env,
                                            last_child_start=last_child_start)
                observed = evidence.read_arm(out, stdout, arm, case, backend, artifact)
                if arm == "head" and observed["planned_route"] != plan_data:
                    raise ValueError("conditional render plan differs from preflight")
                receipt["arms"][arm]["renders"][case] = dict(observed, guard=asdict(guard))
            evidence.compare(*(receipt["arms"][arm]["renders"][case] for arm in evidence.ARMS))
        if not claim.get("post_execution_verified"):
            raise ValueError("conditional physical release not verified")
        if fixture.snapshot_identity(snapshot, variant, "tier1") != receipt["fixture"]:
            raise ValueError("conditional base fixture changed during comparison")
        for role, row in receipt["assets"].items():
            if role == "pid_gemma":
                if {p.relative_to(row["staged"]).as_posix() for p in Path(row["staged"]).rglob("*") if p.is_file()} != set(row["files"]):
                    raise ValueError("staged PiD Gemma inventory changed")
                if any(fixture.sha(Path(row["staged"]) / name) != digest for name, digest in row["files"].items()):
                    raise ValueError("staged PiD Gemma file changed")
            elif fixture.sha(Path(row["staged"])) != row["sha256"]:
                raise ValueError("staged conditional asset changed during comparison: " + role)
        if fixture.sha(source_path) != source_sha:
            raise ValueError("conditional source changed during comparison")
        receipt["accepted"] = True
    except BaseException as exc:
        primary_error = exc
        receipt["failure"] = str(exc)
        if isinstance(exc, (CudaClaimRejected, CudaReleaseRejected)):
            receipt["cuda_diagnostic"] = exc.diagnostic
        raise
    finally:
        errors = retain_logs(work, artifact, receipt) if created_work else []
        failed_retention = bool(errors and receipt["accepted"])
        if errors:
            receipt["log_errors"] = errors
            receipt["accepted"] = False
        base.write(artifact / "receipt.json", receipt)
        if created_work:
            try:
                base.remove_work(work, tmp)
            except (OSError, ValueError) as cleanup_error:
                receipt["cleanup_failure"] = str(cleanup_error)
                receipt["accepted"] = False
                base.write(artifact / "receipt.json", receipt)
                if primary_error is not None:
                    primary_error.add_note(f"Conditional cleanup also failed: {cleanup_error}")
                elif not failed_retention:
                    raise
        if failed_retention:
            raise ValueError("conditional log publication failed: " + ", ".join(errors))


def verify(args, contract):
    tmp = Path(os.environ["RUNNER_TEMP"])
    proof.unlinked_path(args.output, tmp)
    path = args.output / "receipt.json"
    proof.unlinked_path(path, tmp)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("oversized conditional receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("protocol") != "flux1-conditional-ap13-v1"
            or any(receipt.get(key) != value for key, value in contract.items())
            or type(receipt.get("accepted")) is not bool
            or receipt.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or receipt.get("run_attempt") != os.environ["GITHUB_RUN_ATTEMPT"]):
        raise ValueError("conditional receipt identity mismatch")
    if receipt["accepted"]:
        expected_assets = contract["request"]["assets"]
        actual_assets = receipt.get("assets", {})
        if set(actual_assets) != set(expected_assets) | {"reference_image"}:
            raise ValueError("conditional receipt asset inventory differs from request")
        for role, expected in expected_assets.items():
            actual = actual_assets[role]
            if (any(actual.get(key) != value for key, value in expected.items())
                    or actual.get("staging") != "copy" or actual.get("bytes", 0) <= 0):
                raise ValueError("conditional receipt lost staged asset identity: " + role)
        if (actual_assets["reference_image"].get("sha256") != conditional.REFERENCE_SHA256
                or actual_assets["reference_image"].get("staging") != "copy"):
            raise ValueError("conditional receipt lost reference provenance")
        evidence.validate_plan(receipt["preflight"], contract["case"], contract["backend"])
        if receipt["arms"]["head"]["renders"][contract["case"]]["planned_route"] != receipt["preflight"]:
            raise ValueError("conditional receipt preflight and render plans differ")
    allowed = {"receipt.json"}
    limits = {phase + "." + suffix: cap for phase in PHASES
              for suffix, cap in base.retained_log_limits(contract["request"]["policy"]).items()}
    if contract["request"]["policy"]["backend"] == "windows-cuda":
        limits.update({phase + ".cuda-pool.json": 4096 for phase in ("base-run", "head-run")})
    for name, item in receipt["logs"].items():
        if name not in limits or item["path"] != name:
            raise ValueError("unexpected conditional log identity")
        log = args.output / name
        proof.unlinked_path(log, tmp)
        if (not log.is_file() or log.stat().st_nlink != 1 or log.stat().st_size > limits[name]
                or log.stat().st_size != item["bytes"] or fixture.sha(log) != item["sha256"]):
            raise ValueError("invalid conditional log: " + name)
        allowed.add(name)
    required = {phase + "." + suffix for phase in PHASES
                for suffix in ("stdout.log", "stderr.log", "supervisor.json")}
    if contract["request"]["policy"]["backend"] == "windows-cuda":
        required.update(phase + ".cuda-pool.json" for phase in ("base-run", "head-run"))
    if receipt["accepted"] and (receipt.get("log_errors") or not required <= receipt["logs"].keys()):
        raise ValueError("accepted conditional receipt lacks child logs")
    for arm in evidence.ARMS:
        row = receipt["arms"].get(arm, {}).get("renders", {}).get(contract["case"])
        if not row and not receipt["accepted"]:
            continue
        name = arm + "-" + contract["case"] + ".png"
        png = args.output / name
        proof.unlinked_path(png, tmp)
        side = 1024 if contract["case"] == "pulid-pid" else 256
        if row["png"] != name or png.stat().st_size > side*side*4+1024 or fixture.sha(png) != row["png_sha256"]:
            raise ValueError("invalid conditional image artifact")
        allowed.add(name)
    if {item.name for item in args.output.iterdir()} != allowed:
        raise ValueError("unexpected conditional upload file")
    if receipt["accepted"]:
        if not receipt["physical_claim"]["post_execution_verified"]:
            raise ValueError("unreleased accepted conditional comparison")
        evidence.compare(*(receipt["arms"][arm]["renders"][contract["case"]] for arm in evidence.ARMS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    op = parser.add_mutually_exclusive_group()
    op.add_argument("--plan", action="store_true")
    op.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--backend", choices=list(LANES), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(os.environ[args.request_env])
    contract = plan(args.backend, value, args.expected_sha, os.environ["FLUX1_PAIR_TIER"])
    if args.plan:
        print(json.dumps(contract))
    elif args.output is None:
        parser.error("execution and verification require an output path")
    elif args.verify_artifacts:
        verify(args, contract)
    else:
        execute(args, contract)


if __name__ == "__main__":
    main()
