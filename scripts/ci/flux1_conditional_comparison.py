#!/usr/bin/env python3
"""Manual, one-case FLUX.1 conditional base/head proof on one physical backend."""
import argparse
import dataclasses
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
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
from scripts.ci.model_proof_resources import (CudaClaimRejected, CudaReleaseRejected, choose_card,
                                              claim_chosen_card, executed_card_policy, execution_claim,
                                              planned_card_policy, require_claimed_card)
from scripts.media_campaign_supervisor import load_policy

BASE = overlay.BASE
BASE_TREE = "97568d541e94ac4ff5e7572b5d22a8ea2a671936"
LANES = base.LANES
ENG4B_PROFILE = "eng4b-conditional-tier1"
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
# Device estimates stated to CUDA admission (claim, card selection, spawn). pulid-pid: run
# 43553801528 used 24.79 GB of the whole GPU (scripts/tests/test_flux1_conditional_comparison.py);
# 26 GiB (27.9 GB) covers it by 3.1 GB. Other cases have no recorded device peak and are admitted
# at their 18 GiB cap.
CASE_DEVICE_ESTIMATES = {"pulid-pid": ("flux1-conditional-pulid-pid-whole-gpu-peak-v1", 26 * GIB)}
CASE_CAPS = {"pulid-pid": {"child_footprint_cap_bytes": 48 * GIB, "child_gpu_cap_bytes": 40 * GIB}}
# The watchdog samples the child's instantaneous footprint once per poll; arms run ~11 s, so a 1 s
# poll can miss the final denoise/decode peak (run 43569270221). The supervisor now also records
# each process's kernel lifetime peak, and conditional proofs poll at least every 250 ms.
MAX_POLL_MILLIS = 250
# Physical memory each conditional runner reports; a host cap must stay below it.
RUNNER_HOST_BYTES = {"mlx": 137_438_953_472, "candle-cuda": 516_136_534_016}


def spawn_estimate(estimate):
    """The run_guarded device-estimate arguments for a stated case estimate (none: the cap)."""
    return {} if estimate is None else {"static_floor_gpu_bytes": estimate[1], "gpu_estimate_source": estimate[0]}


def check_spawn_estimate(admission, estimate):
    """A child's spawn admission priced the card at the case's stated estimate, or (a record from
    before the case stated one, or a case without one) at the child cap."""
    legacy = (admission.get("staticFloorGpuBytes") is None
              and admission.get("gpuEstimateSource") == "child-footprint-cap-fallback")
    stated = estimate is not None and (admission.get("staticFloorGpuBytes"), admission.get("gpuEstimateSource")) == (estimate[1], estimate[0])
    if not (legacy or stated):
        raise ValueError("conditional child was admitted on a different device estimate than its case states")


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
    eng4b = tier == ENG4B_PROFILE
    baseline = overlay.ENG4B_BASE if eng4b else BASE
    if tier not in ("conditional-tier1", ENG4B_PROFILE) or backend not in LANES or head == baseline:
        raise ValueError("select a distinct head and a conditional Tier 1 backend")
    if eng4b and backend != "candle-cuda":
        raise ValueError("ENG-4b conditional comparison requires Candle CUDA")
    selection = "flux1-eng4b-conditional-pair" if eng4b else "flux1-conditional-pair"
    if type(value) is not dict or set(value) != ({"selection", "case", "assets", "policy"}
            | ({"runner"} if backend == "mlx" else set())) or value["selection"] != selection:
        raise ValueError("comparison needs exact conditional selection, case, assets and policy")
    if backend == "mlx" and value.get("runner") != "nax":
        raise ValueError("conditional MLX proof needs the secondary runner")
    variant, assets = conditional.case_metadata(value["case"], backend, value["assets"])
    bindings = {}
    if eng4b:
        pins = json.loads((fixture.HERE / "conditional_eng4b_assets.json").read_text(encoding="utf-8"))
        case_pins = pins["cases"].get(value["case"])
        if case_pins is None:
            raise ValueError("ENG-4b conditional case must have archived pinned input evidence")
        if {role: {key: row[key] for key in ("revision", "sha256")} for role, row in assets.items()} != case_pins["assets"]:
            raise ValueError("ENG-4b conditional assets differ from the sealed input revisions or hashes")
        bindings = {"base_tree": overlay.ENG4B_BASE_TREE, "conditional_pins": case_pins,
                    "fixture_inventory_sha256": pins["dev_fixture_inventory_sha256"]}
    if value["policy"]["backend"] != proof.POLICY_BACKENDS[backend]:
        raise ValueError("resource policy differs from conditional lane")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        base.write(path, planned_card_policy(value["policy"]))
        policy = load_policy(path)
        caps = case_caps(backend, value["case"], policy)
    source = proof.repository_identity(head)
    fixture.geometry(variant)
    case = value["case"]
    request = dict(value, assets=assets)
    settings = dict(evidence.SETTINGS, case=case, variant=variant,
                    guidance=3.5 if variant == "dev" else None)
    revision_provenance = "caller-declared repository/model revision; SHA-256 of staged bytes is authoritative"
    if eng4b:
        settings["execution"] = "both arms compile and use the same plan"
        revision_provenance = "accepted historical receipt and exact staged SHA-256"
    return {"runner": LANES[backend], "asset_setup": "flux1-eng4b-comparison" if eng4b else "flux1-comparison",
            "backend": backend, "base": baseline, "head": source["revision"], "head_tree": source["tree"],
            "tier": fixture.ENG4B if eng4b else "tier1", "case": case, "variant": variant, "request": request,
            "geometry_sha256": fixture.sha(fixture.HERE / "geometry.tsv"),
            "reference_sha256": conditional.REFERENCE_SHA256,
            "asset_revision_provenance": revision_provenance, "settings": settings,
            "tolerance": {"class": "AP13(a)", "states": "bit-exact", "events": "exact",
                          "rgb8": "bit-exact", "provenance": "SGA implementation plan §7.3; no adjustable limit"},
            "child_caps": caps, **bindings,
            "cuda_claim_budget_seconds": cuda_claim_budget(policy, fixture.ENG4B if eng4b else "tier1")[0]
            if backend == "candle-cuda" else None}


def preflight_sites(tier):
    return ((("base", "base-preflight"), ("head", "head-preflight")) if tier == fixture.ENG4B
            else (("head", "preflight"),))


def phases(tier):
    return tuple(label for _, label in preflight_sites(tier)) + ("base-run", "head-run")


def expected_source(contract, arm):
    tree = contract.get("base_tree", BASE_TREE) if arm == "base" else contract["head_tree"]
    return {"revision": contract[arm], "tree": tree}


def build(work, sources, arm, backend, contract, env):
    root = work / arm
    expected = expected_source(contract, arm)
    source = base.add_source_worktree(sources[arm], root, expected["revision"])
    if source != expected:
        raise ValueError("conditional checkout does not match the expected source tree")
    package = conditional_probe.install(root, backend, arm, contract["tier"], contract["case"])
    family = "mlx-gen" if backend == "mlx" else "candle-gen"
    path = f"crates/media/{family}/{package}/src/bin/flux1-pair.rs"
    base.git(root, "add", "-N", path)
    if contract["tier"] == fixture.ENG4B:
        common = "crates/media/candle-gen/"
        text = "candle-gen-flux/src/packed_te.rs" if arm == "base" else "candle-gen/src/text_encoder.rs"
        required = {path, common + text, common + "candle-gen/src/sampler.rs",
                    f"crates/media/{family}/{package}/Cargo.toml"}
        changed = set(base.git(root, "diff", "--name-only", "HEAD").splitlines())
        if not required <= changed <= required | {"Cargo.lock"}:
            raise ValueError("conditional ENG-4b overlay did not bind the expected source paths")
    delta = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    tc = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--locked", "-p", package, "--bin", "flux1-pair"]
    if backend == "candle-cuda":
        command += ["--features", "cuda"]
    subprocess.run(command, env=env, cwd=root, check=True)
    binary = target / "debug" / ("flux1-pair.exe" if os.name == "nt" else "flux1-pair")
    return binary, {**source, "overlay_sha256": hashlib.sha256(delta).hexdigest(), "toolchain": tc,
                    "binary_sha256": fixture.sha(binary), "build_command": command,
                    "macos_deployment_target": env.get("MACOSX_DEPLOYMENT_TARGET") if backend == "mlx" else None}


def cuda_claim_budget(policy, tier="tier1"):
    reserve = base.RELEASE_RESERVE_SECONDS + policy.term_grace_millis / 1000 + policy.poll_millis / 1000
    return (len(phases(tier)) + 1) * policy.deadline_seconds + reserve, reserve


def retain_logs(work, artifact, receipt):
    errors = []
    for phase in phases(receipt["tier"]):
        limits = base.retained_log_limits(receipt["request"]["policy"])
        if receipt["request"]["policy"]["backend"] == "windows-cuda" and phase.endswith("-run"):
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


def verify_pinned_inputs(receipt, contract):
    files = json.dumps(receipt["fixture"]["files"], sort_keys=True, separators=(",", ":")).encode()
    if (hashlib.sha256(files).hexdigest() != contract["fixture_inventory_sha256"]
            or receipt["source"]["sha256"] != contract["conditional_pins"]["source_sha256"]):
        raise ValueError("ENG-4b fixture or source image differs from the sealed input bytes")


def execute(args, contract):
    backend, case, variant = args.backend, contract["case"], contract["variant"]
    eng4b = contract["tier"] == fixture.ENG4B
    proof.require_host(backend, dict(os.environ), "nax" if backend == "mlx" else "hosted")
    tmp = Path(os.environ["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    receipt = {"schema": 1, "protocol": "flux1-conditional-ap13-v1", **contract,
               "accepted": False, "run_id": os.environ["GITHUB_RUN_ID"],
               "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "host": platform.platform(),
               "runner_name": os.environ["RUNNER_NAME"], "arms": {}, "logs": {}}
    work = tmp / f"f1c-{receipt['run_id']}-{receipt['run_attempt']}"
    sources = {"head": ROOT, "base": ROOT / "flux1-baseline"}
    created_work = False
    primary_error = None
    try:
        proof.unlinked_path(work, tmp)
        work.mkdir()
        created_work = True
        if base.git(sources["base"], "rev-parse", "HEAD") != contract["base"]:
            raise ValueError("conditional baseline is not the profile's reviewed engine")
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
        if eng4b:
            verify_pinned_inputs(receipt, contract)
        request_policy = contract["request"]["policy"]
        estimate = CASE_DEVICE_ESTIMATES.get(case) if backend == "candle-cuda" else None
        # Builds never touch a GPU: a request that names no card builds first, then chooses (and if
        # need be switches) its card in the claim's own admission window.
        policy_path = work / "policy.json"
        base.write(policy_path, planned_card_policy(request_policy))
        policy = load_policy(policy_path)
        exes, environments = {}, {}
        for arm in evidence.ARMS:
            env = os.environ.copy()
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["CARGO_INCREMENTAL"] = "0"
            if backend == "mlx":
                env["MACOSX_DEPLOYMENT_TARGET"] = "26.2"
            else:
                env.update(CUDA_VISIBLE_DEVICES=policy.cuda_device_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            binary, info = build(work, sources, arm, backend, contract, env)
            exes[arm], environments[arm] = binary, env
            receipt["arms"][arm] = {"build": info, "renders": {}}
        receipt["allocator"] = {"scope": "fresh process per revision/case on the same physical device",
                                "mlx": "default MLX unified allocator", "candle": "default CUDA allocator",
                                "cublas_workspace_config": ":4096:8"}
        base.write(artifact / "receipt.json", receipt)
        claim_dir = work / "claim"
        claim_dir.mkdir()
        claim_deadline = last_child_start = budget = None
        if backend == "candle-cuda":
            budget, reserve = cuda_claim_budget(policy, contract["tier"])
            claim_deadline = time.monotonic() + budget
            last_child_start = claim_deadline - reserve - policy.deadline_seconds
        lease_env = environments["base"]
        # A request that names no card runs on the registered card with the most free memory that fits.
        payload, selection = choose_card(request_policy, work, estimate=estimate, deadline_monotonic=last_child_start)
        if selection is not None:
            receipt["cuda_card_selection"] = selection
        bound = {}

        def bind(chosen):
            """Bind the run to the card in force: its policy, hardware and child visibility."""
            base.write(policy_path, chosen)
            bound["policy"] = load_policy(policy_path)
            receipt["hardware"] = base.hardware(backend, bound["policy"])
            if backend == "candle-cuda":
                for arm_env in environments.values():
                    arm_env["CUDA_VISIBLE_DEVICES"] = chosen["cudaDeviceUuid"]

        def claim_policy(chosen):
            if budget is None:
                return bound["policy"]
            base.write(claim_dir / "policy.json", dict(chosen, deadlineSeconds=budget))
            held = load_policy(claim_dir / "policy.json")
            return dataclasses.replace(held, claim_gpu_estimate=estimate) if estimate else held

        bind(payload)
        with claim_chosen_card(backend, request_policy, payload, selection, claim_dir, lease_env,
                               claim_policy=claim_policy, rebind=bind, record=receipt, estimate=estimate,
                               claim_with=execution_claim, original_deadline_monotonic=claim_deadline) as claim:
            policy = bound["policy"]
            receipt["physical_claim"] = claim
            shared = {"PAIR_CASE": case, "PAIR_SNAPSHOT": str(snapshot), "PAIR_ASSETS": str(staged),
                      "PAIR_SOURCE": str(source_path)}
            if "pid_gemma" in receipt["assets"]:
                shared["PAIR_PID_GEMMA"] = receipt["assets"]["pid_gemma"]["staged"]
            plans = {}
            for arm, label in preflight_sites(contract["tier"]):
                preflight = work / label
                env = dict(environments[arm], **shared)
                _, stdout = base.invoke(exes[arm], work / arm, preflight, env, policy, lease_env,
                                        preflight=True, last_child_start=last_child_start,
                                        admission_estimate=spawn_estimate(estimate))
                plan_data = json.loads((preflight / "plan.json").read_text(encoding="utf-8"))
                evidence.validate_plan(plan_data, case, backend)
                if stdout.splitlines().count("FLUX1_CONDITIONAL_PREFLIGHT " + case) != 1:
                    raise ValueError("missing conditional pre-load preflight")
                plans[arm] = plan_data
            if eng4b and plans["base"] != plans["head"]:
                raise ValueError("ENG-4b conditional base and head plans differ")
            receipt["preflight"] = plans if eng4b else plans["head"]
            base.write(artifact / "receipt.json", receipt)
            for arm in evidence.ARMS:
                out = work / (arm + "-run")
                env = dict(environments[arm], **shared)
                guard, stdout = base.invoke(exes[arm], work / arm, out, env, policy, lease_env,
                                            last_child_start=last_child_start,
                                            admission_estimate=spawn_estimate(estimate))
                observed = evidence.read_arm(out, stdout, arm, case, backend, artifact, planned=arm in plans)
                if arm in plans and observed["planned_route"] != plans[arm]:
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
                base.remove_source_worktrees(work, sources)
                base.remove_work(work, tmp)
            except (OSError, ValueError, subprocess.SubprocessError) as cleanup_error:
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
        eng4b = contract["tier"] == fixture.ENG4B
        plans = receipt["preflight"] if eng4b else {"head": receipt["preflight"]}
        if eng4b:
            verify_pinned_inputs(receipt, contract)
            if plans.keys() != set(evidence.ARMS) or plans["base"] != plans["head"]:
                raise ValueError("ENG-4b conditional receipt lacks equivalent planned arms")
        for arm, _ in preflight_sites(contract["tier"]):
            evidence.validate_plan(plans[arm], contract["case"], contract["backend"])
            if receipt["arms"][arm]["renders"][contract["case"]]["planned_route"] != plans[arm]:
                raise ValueError("conditional receipt preflight and render plans differ")
        for arm in evidence.ARMS:
            build = receipt["arms"][arm]["build"]
            if (any(build[key] != value for key, value in expected_source(contract, arm).items())
                    or not all(re.fullmatch(r"[0-9a-f]{64}", build[key])
                               for key in ("overlay_sha256", "binary_sha256"))):
                raise ValueError("conditional receipt source, overlay or executable identity mismatch")
    # A card-free request ran on the card its recorded selection chose (and claimed) at the case's
    # stated estimate; a named card records no selection and verifies as before.
    estimate = CASE_DEVICE_ESTIMATES.get(contract["case"]) if contract["backend"] == "candle-cuda" else None
    executed = executed_card_policy(contract["request"]["policy"], receipt.get("cuda_card_selection"), estimate=estimate)
    if "physical_claim" in receipt and contract["backend"] == "candle-cuda" and (
            "cuda_card_selection" in receipt or receipt["physical_claim"].get("claim_gpu_estimate") is not None):
        require_claimed_card(receipt["physical_claim"], executed, estimate)
    allowed = {"receipt.json"}
    limits = {phase + "." + suffix: cap for phase in phases(contract["tier"])
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
    required = {phase + "." + suffix for phase in phases(contract["tier"])
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
    if contract["backend"] == "candle-cuda":
        for name in receipt["logs"]:
            if name.endswith(".supervisor.json"):
                guard = json.loads((args.output / name).read_text(encoding="utf-8"))
                if isinstance(guard.get("admission"), dict):
                    check_spawn_estimate(guard["admission"], estimate)
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
