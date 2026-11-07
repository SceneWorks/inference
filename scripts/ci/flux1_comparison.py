#!/usr/bin/env python3
"""Manual same-job FLUX.1 base/head proof. Planning executes no model code."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof
from scripts.ci.flux1_pair import evidence, fixture, overlay
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import SupervisionError, load_policy, run_guarded

BASE = overlay.BASE
TIERS = ("tier1", "tier2")
LANES = {"mlx": ["self-hosted", "macOS", "ARM64", "rw-llm"],
         "candle-cuda": ["self-hosted", "windows", "cuda", "real-weights"]}


def write(path, row):
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def plan(backend, value, head, tier):
    if backend not in LANES or tier not in TIERS or head == BASE:
        raise ValueError("select distinct head and supported proof tier/backend")
    if (set(value) != {"selection", "policy", *({"runner"} if backend == "mlx" else set())}
            or value["selection"] != "flux1-base-pair"
            or (backend == "mlx" and value.get("runner") != "nax")):
        raise ValueError("comparison requires exact flux1-base-pair selection and secondary MLX runner")
    if value["policy"]["backend"] != proof.POLICY_BACKENDS[backend]:
        raise ValueError("resource policy differs from lane")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        write(path, value["policy"])
        policy = load_policy(path)
        if policy.deadline_seconds > 1200 or policy.child_footprint_cap_bytes > 24 * 1024**3:
            raise ValueError("proof child exceeds frozen 20 minute/24 GiB budget")
        if backend == "candle-cuda" and policy.child_gpu_cap_bytes > 18 * 1024**3:
            raise ValueError("proof exceeds 18 GiB GPU child cap")
    source = proof.repository_identity(head)
    for variant in evidence.VARIANTS:
        fixture.geometry(variant)
    return {"runner": LANES[backend], "asset_setup": "flux1-comparison", "backend": backend,
            "base": BASE, "head": source["revision"], "head_tree": source["tree"],
            "tier": tier, "geometry_sha256": fixture.sha(fixture.HERE / "geometry.tsv"),
            "request": value, "settings": evidence.SETTINGS}


def build(work, sources, arm, backend, tier, env):
    root = work / arm
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(sources[arm]), str(root)], check=True)
    sha = git(sources[arm], "rev-parse", "HEAD")
    subprocess.run(["git", "checkout", "--quiet", "--detach", sha], cwd=root, check=True)
    if git(root, "rev-parse", "HEAD") != sha:
        raise ValueError("unexpected source revision")
    package = overlay.install(root, backend, arm, tier)
    # Record the exact controlled source delta, including the untracked binary.
    git(root, "add", "-N", f"crates/media/{'mlx-gen' if backend == 'mlx' else 'candle-gen'}/{package}/src/bin/flux1-pair.rs")
    changes = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    tc = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    cmd = ["cargo", "build", "--locked", "-p", package, "--bin", "flux1-pair"]
    if backend == "candle-cuda":
        cmd += ["--features", "cuda"]
    subprocess.run(cmd, env=env, cwd=root, check=True)
    binary = target / "debug" / ("flux1-pair.exe" if os.name == "nt" else "flux1-pair")
    return binary, {"revision": sha, "tree": git(root, "rev-parse", "HEAD^{tree}"),
                    "overlay_sha256": hashlib.sha256(changes).hexdigest(), "toolchain": tc,
                    "binary_sha256": fixture.sha(binary), "build_command": cmd,
                    "macos_deployment_target": env.get("MACOSX_DEPLOYMENT_TARGET") if backend == "mlx" else None}


def snapshots(work, tier):
    paths, identities = {}, {}
    provenance = json.loads((fixture.HERE / "headers.json").read_text(encoding="utf-8"))
    for variant in evidence.VARIANTS:
        if tier == "tier1":
            root = work / ("random-" + variant)
            fixture.make_snapshot(root, variant)
        else:
            src = provenance[variant]
            hub = Path(os.environ["FLUX1_HUB"])
            if not hub.is_absolute():
                raise ValueError("real proof requires the configured local model hub")
            root = hub / ("models--" + src["repo"].replace("/", "--")) / "snapshots" / src["snapshot"] / "q4"
        paths[variant] = root
        identities[variant] = fixture.snapshot_identity(root, variant, tier)
    return paths, identities


def invoke(binary, arm_root, out, env, policy, lease_env, *, preflight=False):
    out.mkdir()
    fds = ()
    if "SCENEWORKS_METAL_EXECUTION_LOCK_FD" in lease_env:
        fd = lease_env["SCENEWORKS_METAL_EXECUTION_LOCK_FD"]
        env["SCENEWORKS_METAL_EXECUTION_LOCK_FD"] = fd
        fds = (int(fd),)
    stderr_path = out / "stderr.log"
    try:
        guard = run_guarded([str(binary), *(["--preflight"] if preflight else [])],
                            cwd=arm_root, env=dict(env, PAIR_OUT=str(out)), policy=policy,
                            stdout_path=out / "stdout.log", stderr_path=stderr_path,
                            event_path=out / "guard.jsonl", pass_fds=fds, require_gpu_sample=not preflight)
    except SupervisionError:
        # The supervisor can refuse before spawn. Preserve its reason whether or not
        # it created a log; a failing child may have filled its entire stderr budget.
        try:
            with stderr_path.open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 4000))
                tail = log.read(4000).decode("utf-8", errors="replace")
        except OSError:
            tail = ""
        if tail:
            print(tail, file=sys.stderr)
        raise
    return guard, (out / "stdout.log").read_text(encoding="utf-8")


def hardware(backend, policy):
    if backend == "mlx":
        def sysctl(name):
            return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True, timeout=10).strip()
        return {"chip": sysctl("machdep.cpu.brand_string"), "physical_memory_bytes": int(sysctl("hw.memsize")),
                "os_build": sysctl("kern.osversion")}
    from scripts.media_campaign_windows import trusted_nvidia_smi
    fields = ["name", "uuid", "index", "memory.total", "driver_version"]
    command = [trusted_nvidia_smi(), "--id=" + policy.cuda_device_uuid, "--query-gpu=" + ",".join(fields),
               "--format=csv,noheader,nounits"]
    values = subprocess.check_output(command, text=True, timeout=15).strip().split(", ")
    if len(values) != len(fields) or values[1] != policy.cuda_device_uuid:
        raise ValueError("hardware inventory does not match the selected physical CUDA device")
    return dict(zip(fields, values))


def execute(args, contract):
    backend, tier = args.backend, contract["tier"]
    proof.require_host(backend, dict(os.environ), "nax" if backend == "mlx" else "hosted")
    tmp = Path(os.environ["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    receipt = {"schema": 1, "protocol": "flux1-ap13-v1", **contract, "accepted": False,
               "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
               "host": platform.platform(), "runner_name": os.environ["RUNNER_NAME"], "arms": {}}
    work = tmp / f"f1p-{receipt['run_id']}-{receipt['run_attempt']}"
    created_work = False
    try:
        proof.unlinked_path(work, tmp)
        work.mkdir()
        created_work = True
        sources = {"head": ROOT, "base": ROOT / "flux1-baseline"}
        if git(sources["base"], "rev-parse", "HEAD") != BASE:
            raise ValueError("baseline is not the pre-migration engine")
        paths, receipt["fixtures"] = snapshots(work, tier)
        policy_path = work / "policy.json"
        write(policy_path, contract["request"]["policy"])
        policy = load_policy(policy_path)
        receipt["hardware"] = hardware(backend, policy)
        # Both executables finish linking before either arm's first render.
        exes, environments = {}, {}
        for arm in evidence.ARMS:
            env = os.environ.copy()
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["CARGO_INCREMENTAL"] = "0"
            if backend == "mlx":
                # The proof route is the secondary NAX Mac; use the shipping
                # matrix-unit deployment floor instead of the hosted CI floor.
                env["MACOSX_DEPLOYMENT_TARGET"] = "26.2"
            if backend == "candle-cuda":
                env.update(CUDA_VISIBLE_DEVICES=policy.cuda_device_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            binary, build_info = build(work, sources, arm, backend, tier, env)
            exes[arm], environments[arm] = binary, env
            receipt["arms"][arm] = {"build": build_info, "renders": {}}
        receipt["allocator"] = {"scope": "fresh process per revision and variant on the same physical device",
                                "mlx": "provider default MLX allocator; unified host memory",
                                "candle": "provider default CUDA allocator; device memory tracked separately",
                                "cublas_workspace_config": ":4096:8"}
        write(artifact / "receipt.json", receipt)
        # Hold the same physical resource across the comparison. Per-child
        # deadlines, RSS and (on CUDA) device caps remain independently enforced.
        claim_dir = work / "claim"
        claim_dir.mkdir()
        lease_env = environments["base"]
        with execution_claim(backend, policy, claim_dir, lease_env) as claim:
            receipt["physical_claim"] = claim
            receipt["preflight"] = {}
            for variant in evidence.VARIANTS:
                out = work / ("preflight-" + variant)
                model = "flux1_" + variant
                env = dict(environments["head"], PAIR_MODEL=model, PAIR_SNAPSHOT=str(paths[variant]))
                _, stdout = invoke(exes["head"], work / "head", out, env, policy, lease_env, preflight=True)
                plan_data = json.loads((out / "plan.json").read_text(encoding="utf-8"))
                evidence.validate_plan(plan_data, variant, backend)
                if stdout.splitlines().count("FLUX1_PAIR_PREFLIGHT " + model) != 1:
                    raise ValueError("missing compiled native preflight")
                receipt["preflight"][variant] = plan_data
            write(artifact / "receipt.json", receipt)
            for variant in evidence.VARIANTS:
                for arm in evidence.ARMS:
                    out = work / (arm + "-" + variant)
                    env = dict(environments[arm], PAIR_MODEL="flux1_" + variant,
                               PAIR_SNAPSHOT=str(paths[variant]))
                    guard, stdout = invoke(exes[arm], work / arm, out, env, policy, lease_env)
                    observed = evidence.read_arm(out, stdout, arm, variant, backend, artifact)
                    if arm == "head" and observed["planned_route"] != receipt["preflight"][variant]:
                        raise ValueError("render plan differs from preflight")
                    receipt["arms"][arm]["renders"][variant] = dict(observed, guard=asdict(guard))
                evidence.compare(*(receipt["arms"][arm]["renders"][variant] for arm in evidence.ARMS))
        if not claim.get("post_execution_verified"):
            raise ValueError("physical release not verified")
        for variant in evidence.VARIANTS:
            if fixture.snapshot_identity(paths[variant], variant, tier) != receipt["fixtures"][variant]:
                raise ValueError("snapshot changed during A/B")
        receipt["accepted"] = True
    except Exception as exc:
        receipt["failure"] = str(exc)
        raise
    finally:
        write(artifact / "receipt.json", receipt)
        # Only this attempt's fresh private work tree; receipts and PNGs live outside it.
        if created_work and work.exists() and not work.is_symlink():
            shutil.rmtree(work)


def verify(args, contract):
    tmp = Path(os.environ["RUNNER_TEMP"])
    proof.unlinked_path(args.output, tmp)
    path = args.output / "receipt.json"
    proof.unlinked_path(path, tmp)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("oversized comparison receipt")
    r = json.loads(path.read_text(encoding="utf-8"))
    if (r.get("protocol") != "flux1-ap13-v1" or any(r.get(k) != v for k, v in contract.items())
            or type(r.get("accepted")) is not bool or r.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or r.get("run_attempt") != os.environ["GITHUB_RUN_ATTEMPT"]):
        raise ValueError("comparison receipt identity mismatch")
    allowed = {"receipt.json"}
    for arm in evidence.ARMS:
        for variant in evidence.VARIANTS:
            row = r["arms"].get(arm, {}).get("renders", {}).get(variant)
            if not row and not r["accepted"]:
                continue
            name = arm + "-" + variant + ".png"
            png = args.output / name
            proof.unlinked_path(png, tmp)
            if row["png"] != name or png.stat().st_size > 256*256*4 or fixture.sha(png) != row["png_sha256"]:
                raise ValueError("invalid image artifact")
            allowed.add(name)
    if {p.name for p in args.output.iterdir()} != allowed:
        raise ValueError("unexpected upload file")
    if r["accepted"]:
        if not r["physical_claim"]["post_execution_verified"]:
            raise ValueError("unreleased accepted comparison")
        for variant in evidence.VARIANTS:
            evidence.compare(*(r["arms"][arm]["renders"][variant] for arm in evidence.ARMS))


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
