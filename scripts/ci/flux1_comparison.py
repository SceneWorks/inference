#!/usr/bin/env python3
"""Manual same-job FLUX.1 base/head proof. Planning executes no model code."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof
from scripts.ci import stage_pinned_hub_snapshots as hub_snapshots
from scripts.ci.flux1_conditional_asset_inventory import confined, fetch_pinned
from scripts.ci.flux1_pair import evidence, fixture, overlay, performance
from scripts.ci.flux1_pair.cuda_probe import PairCudaProbe
from scripts.ci.model_proof_resources import CudaClaimRejected, CudaReleaseRejected, execution_claim
from scripts.media_campaign_supervisor import SupervisionError, load_policy, run_guarded

BASE = overlay.BASE
ENG4B = fixture.ENG4B
ENG4B_REAL = fixture.ENG4B_REAL
TIERS = ("tier1", "tier2", *fixture.ENG4B_TIERS)
LANES = {"mlx": ["self-hosted", "macOS", "ARM64", "rw-llm"],
         "candle-cuda": ["self-hosted", "windows", "cuda", "real-weights"]}


def write(path, row):
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def add_source_worktree(source, root, revision):
    """Reuse the exact Actions checkout without fetching from it as a remote."""
    proof.unlinked_path(root, root.parent)
    if root.exists() or git(source, "rev-parse", "HEAD") != revision:
        raise ValueError("proof requires a new worktree from the exact source checkout")
    # Local clone transport can ask a reused runner checkout to serve missing
    # historical promisor objects. The linked worktree needs only the exact
    # checked-out tree, whose files were already materialized by Actions.
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(root), revision],
        cwd=source, env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        check=True, timeout=120,
    )
    observed = {"revision": git(root, "rev-parse", "HEAD"),
                "tree": git(root, "rev-parse", "HEAD^{tree}")}
    if observed["revision"] != revision or git(root, "status", "--porcelain"):
        raise ValueError("proof worktree does not match the exact clean checkout")
    return observed


def remove_source_worktrees(work, sources):
    """Deregister the owned worktrees before deleting the rest of the attempt."""
    for arm, source in sources.items():
        root = work / arm
        marker = root / ".git"
        proof.unlinked_path(marker, work)
        if marker.is_file():
            subprocess.run(["git", "worktree", "remove", "--force", str(root)],
                           cwd=source, check=True, timeout=120)


def variants(tier):
    return ("dev",) if tier == ENG4B else evidence.VARIANTS


def planned_arms(tier):
    return evidence.ARMS if tier in fixture.ENG4B_TIERS else ("head",)


def preflight_key(tier, arm, variant):
    return f"{arm}-{variant}" if tier in fixture.ENG4B_TIERS else variant


def phase_names(tier):
    return tuple(f"preflight-{preflight_key(tier, arm, variant)}"
                 for arm in planned_arms(tier) for variant in variants(tier)) + tuple(
                     f"{arm}-{variant}" for variant in variants(tier) for arm in evidence.ARMS) + tuple(
                         name for name, _, _ in measured_cases(tier))


def measured_cases(tier):
    return performance.cases(variants(tier)) if tier == ENG4B_REAL else ()


def plan(backend, value, head, tier):
    eng4b = tier in fixture.ENG4B_TIERS
    measured = tier == ENG4B_REAL
    baseline, baseline_tree = overlay.eng4b_baseline(tier) if eng4b else (BASE, None)
    if backend not in LANES or tier not in TIERS or head == baseline:
        raise ValueError("select distinct head and supported proof tier/backend")
    if eng4b and backend != "candle-cuda":
        raise ValueError("ENG-4b FLUX.1 comparison requires the Candle CUDA lane")
    selection = "flux1-eng4b-pair" if eng4b else "flux1-base-pair"
    fields = {"selection", "policy", *({"runner"} if backend == "mlx" else set())}
    if measured and type(value) is dict and type(value.get("download_missing")) is bool:
        fields.add("download_missing")
    if (type(value) is not dict or set(value) != fields
            or value["selection"] != selection or (backend == "mlx" and value.get("runner") != "nax")):
        raise ValueError(f"comparison requires exact {selection} selection and secondary MLX runner where applicable")
    if value["policy"]["backend"] != proof.POLICY_BACKENDS[backend]:
        raise ValueError("resource policy differs from lane")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.json"
        write(path, value["policy"])
        policy = load_policy(path)
        deadline = performance.MAX_CHILD_SECONDS if measured else 1200
        if policy.deadline_seconds > deadline or policy.child_footprint_cap_bytes > 24 * 1024**3:
            raise ValueError(f"proof child exceeds frozen {deadline} second/24 GiB budget")
        if measured and policy.poll_millis != 1000:
            raise ValueError("terminal comparison requires the frozen one-second guard poll")
        if backend == "candle-cuda" and policy.child_gpu_cap_bytes > 18 * 1024**3:
            raise ValueError("proof exceeds 18 GiB GPU child cap")
    source = proof.repository_identity(head)
    for variant in variants(tier):
        fixture.geometry(variant)
    settings = evidence.SETTINGS
    identity = {}
    asset_setup = "flux1-comparison"
    if eng4b:
        identity["base_tree"] = baseline_tree
        asset_setup = "flux1-eng4b-terminal-comparison" if measured else "flux1-eng4b-comparison"
    if tier == ENG4B:
        settings = dict(settings, guidance={"dev": 3.5},
                        weights="synthetic MLX affine Q4 transformer, CLIP-L and T5; dense BF16 VAE",
                        depth={"double": 1, "single": 1, "clip": 1, "t5": 1},
                        fixture_seed=fixture.SEED, plans="compiled and used on both arms")
    if measured:
        identity["performance_contract"] = performance.CONTRACT
        identity["pinned_fixtures"] = {variant: fixture.real_snapshot_identity(variant, tier)
                                       for variant in variants(tier)}
        settings = dict(settings, weights="complete provisioned Q4 snapshots from the pinned file hashes",
                        depth={"double": 19, "single": 38, "clip": 12, "t5": 24},
                        plans="compiled and used on both arms", scope="Candle CUDA resident plain T2I")
    return {"runner": LANES[backend], "asset_setup": asset_setup,
            "backend": backend, "base": baseline, "head": source["revision"], "head_tree": source["tree"],
            "tier": tier, "geometry_sha256": fixture.sha(fixture.HERE / "geometry.tsv"),
            "request": value, "settings": settings, **identity,
            "cuda_claim_budget_seconds": cuda_claim_budget(policy, tier)[0] if backend == "candle-cuda" else None}


def build(work, sources, arm, backend, tier, env):
    root = work / arm
    sha = git(sources[arm], "rev-parse", "HEAD")
    source_identity = add_source_worktree(sources[arm], root, sha)
    package = overlay.install(root, backend, arm, tier)
    # Record the exact controlled source delta, including the untracked binary.
    probe_source = f"crates/media/{'mlx-gen' if backend == 'mlx' else 'candle-gen'}/{package}/src/bin/flux1-pair.rs"
    git(root, "add", "-N", probe_source)
    if tier in fixture.ENG4B_TIERS:
        common = "crates/media/candle-gen/"
        text = "candle-gen-flux/src/packed_te.rs" if arm == "base" else "candle-gen/src/text_encoder.rs"
        allowed = {probe_source, common + "candle-gen/src/sampler.rs"}
        if tier == ENG4B:
            allowed.add(common + text)
        if set(git(root, "diff", "--name-only", "HEAD").splitlines()) != allowed:
            raise ValueError("ENG-4b overlay did not modify exactly the expected source paths")
    changes = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    tc = proof.verified_toolchain(env, root=root)
    target = work / (arm + "-target")
    env["CARGO_TARGET_DIR"] = str(target)
    if tier == ENG4B_REAL:
        cmd, profile = list(performance.BUILD), "release"
    else:
        cmd, profile = ["cargo", "build", "--locked", "-p", package, "--bin", "flux1-pair"], "debug"
        if backend == "candle-cuda":
            cmd += ["--features", "cuda"]
    subprocess.run(cmd, env=env, cwd=root, check=True)
    binary = target / profile / ("flux1-pair.exe" if os.name == "nt" else "flux1-pair")
    return binary, {**source_identity,
                    "overlay_sha256": hashlib.sha256(changes).hexdigest(), "toolchain": tc,
                    "binary_sha256": fixture.sha(binary), "build_command": cmd,
                    "macos_deployment_target": env.get("MACOSX_DEPLOYMENT_TARGET") if backend == "mlx" else None}


def stage_terminal_snapshots(hub, provenance):
    hub = hub.resolve(strict=True)
    pins = {variant: {"repository": provenance[variant]["repo"],
                      "revision": provenance[variant]["snapshot"], "subfolder": "q4"}
            for variant in variants(ENG4B_REAL)}
    expected = {(pin["repository"], pin["revision"]): provenance[variant]["q4_files"]
                for variant, pin in pins.items()}
    by_destination = {}
    for (repository, revision), files in expected.items():
        root = hub_snapshots.snapshot_dir(hub, repository, revision)
        for relative, identity in files.items():
            name = "q4/" + relative
            destination = root / name
            confined(destination.resolve(), hub, destination)
            lock = hub_snapshots.staging_lock_path(hub, repository, revision, name)
            proof.unlinked_path(lock, hub)
            if lock.exists() and (not lock.is_file() or lock.stat().st_nlink != 1):
                raise ValueError("FLUX.1 staging lock must be a regular unshared file")
            by_destination[destination] = identity

    def terminal_manifest(repository, revision):
        tree, raw, upstream = hub_snapshots.pinned_manifest(repository, revision)
        selected = {}
        for relative, identity in expected[repository, revision].items():
            name = "q4/" + relative
            asset = upstream.get(name, {})
            lfs = asset.get("lfs")
            if (asset.get("size") != identity["bytes"]
                    or ((lfs is not None or name.endswith(".safetensors")) and (
                        not isinstance(lfs, dict) or lfs.get("sha256") != identity["sha256"]
                        or lfs.get("size") != identity["bytes"]))):
                raise ValueError(f"FLUX.1 Hub inventory differs from the exact fixture for {name}")
            selected[name] = asset
        return tree, raw, selected

    def fetch(url, destination):
        identity = by_destination[destination]
        confined(destination.parent.resolve(), hub, destination)
        return fetch_pinned(url, destination, expected_sha256=identity["sha256"],
                            expected_bytes=identity["bytes"])

    hub_snapshots.stage(pins, hub, manifest=terminal_manifest, fetch=fetch)


def snapshots(work, tier, *, download_missing=False):
    paths, identities = {}, {}
    provenance = json.loads((fixture.HERE / "headers.json").read_text(encoding="utf-8"))
    if tier not in ("tier1", ENG4B):
        hub = Path(os.environ["FLUX1_HUB"])
        if not hub.is_absolute():
            raise ValueError("real proof requires the configured local model hub")
        if tier == ENG4B_REAL and download_missing:
            hub.mkdir(parents=True, exist_ok=True)
            stage_terminal_snapshots(hub, provenance)
    for variant in variants(tier):
        if tier in ("tier1", ENG4B):
            root = work / ("random-" + variant)
            fixture.make_snapshot(root, variant, q4=tier == ENG4B)
        else:
            src = provenance[variant]
            root = hub_snapshots.snapshot_dir(hub, src["repo"], src["snapshot"]) / "q4"
        paths[variant] = root
        identities[variant] = fixture.snapshot_identity(root, variant, tier)
    return paths, identities


RELEASE_RESERVE_SECONDS = 30  # execution_claim's final CUDA release verification window


def cuda_claim_budget(policy, tier="tier1"):
    # One child allowance for lock acquisition/admission, one for each sequential
    # phase, and time for termination and the existing post-idle verification.
    reserve = RELEASE_RESERVE_SECONDS + policy.term_grace_millis / 1000 + policy.poll_millis / 1000
    return (len(phase_names(tier)) + 1) * policy.deadline_seconds + reserve, reserve


def retained_log_limits(policy):
    return {"stdout.log": policy["stdoutCapBytes"], "stderr.log": policy["stderrCapBytes"],
            "guard.jsonl": policy["eventCapBytes"], "supervisor.json": 16 * 1024**2}


def retained_phase_limits(phase, policy):
    limits = retained_log_limits(policy)
    if policy["backend"] == "windows-cuda" and not phase.startswith("preflight-"):
        limits["cuda-pool.json"] = 4096
    if phase.startswith("measure-"):
        limits.update(performance.RESULT_LIMITS)
    return limits


def retain_logs(work, artifact, receipt):
    records = receipt["logs"]
    errors = []
    for phase in phase_names(receipt["tier"]):
        limits = retained_phase_limits(phase, receipt["request"]["policy"])
        for suffix, cap in limits.items():
            source = work / phase / suffix
            name = phase + "." + suffix
            try:
                if not source.exists() and not proof.linked(source):
                    if receipt["accepted"] and suffix != "guard.jsonl":
                        errors.append(f"{name}: missing completed child log")
                    continue
                records[name] = proof.publish_file(source, artifact / name, work, cap,
                                                   log=suffix not in performance.RESULT_LIMITS,
                                                   destination_root=artifact)
            except (OSError, ValueError) as error:
                errors.append(f"{name}: {error}")
    return errors


def invoke(binary, arm_root, out, env, policy, lease_env, *, preflight=False, measure=False, last_child_start=None):
    if last_child_start is not None and time.monotonic() >= last_child_start:
        raise ValueError("insufficient CUDA claim time for another fully guarded child and release")
    out.mkdir()
    fds = ()
    if "SCENEWORKS_METAL_EXECUTION_LOCK_FD" in lease_env:
        fd = lease_env["SCENEWORKS_METAL_EXECUTION_LOCK_FD"]
        env["SCENEWORKS_METAL_EXECUTION_LOCK_FD"] = fd
        fds = (int(fd),)
    stderr_path = out / "stderr.log"
    try:
        mode = ["--preflight"] if preflight else ["--measure"] if measure else []
        guard = run_guarded([str(binary), *mode],
                            cwd=arm_root, env=dict(env, PAIR_OUT=str(out)), policy=policy,
                            stdout_path=out / "stdout.log", stderr_path=stderr_path,
                            event_path=out / "guard.jsonl", pass_fds=fds, require_gpu_sample=not preflight,
                            probe=PairCudaProbe(policy, out / "cuda-pool.json")
                            if policy is not None and policy.backend == "windows-cuda" and os.name == "nt" and not preflight else None)
    except SupervisionError as error:
        proof.write_json(out / "supervisor.json", {
            "reason": error.reason, "detail": error.detail, "pid": error.pid, "admission": error.admission,
            "host_memory": error.watchdog_host_memory, "child_footprint": error.watchdog_child_footprint,
            "device_memory": error.watchdog_device_memory,
        }, owned=out.parent)
        # The supervisor can refuse before spawn. Preserve its reason whether or not
        # it created a log; a failing child may have filled its entire stderr budget.
        try:
            proof.unlinked_path(stderr_path, out.parent)
            if not stderr_path.is_file() or stderr_path.stat().st_nlink != 1:
                raise ValueError("unowned child stderr")
            with stderr_path.open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 4000))
                tail = log.read(4000).decode("utf-8", errors="replace")
        except (OSError, ValueError):
            tail = ""
        if tail:
            print(proof.redact(tail, env), file=sys.stderr)
        raise
    proof.write_json(out / "supervisor.json", asdict(guard), owned=out.parent)
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


def capture_measured_output(out, reference):
    raw = out / "output.rgb"
    proof.unlinked_path(raw, out)
    size = 256 * 256 * 3
    if not raw.is_file() or raw.stat().st_size != size or raw.stat().st_nlink != 1:
        raise ValueError("incomplete measured RGB8 image")
    with raw.open("rb") as source:
        rgb = source.read(size + 1)
    if hashlib.sha256(rgb).hexdigest() != reference["rgb_sha256"]:
        raise ValueError("measured RGB8 differs from fidelity")
    with (out / "output.png").open("xb") as image:
        image.write(evidence.png(rgb))


def remove_work(work, temporary):
    """Remove only the tree created by this attempt. Git packs can be read-only on Windows."""
    if work.parent != temporary:
        raise ValueError("comparison cleanup requires a direct child of RUNNER_TEMP")

    def unlink_readonly(function, name, error):
        if function is not os.unlink or getattr(error, "winerror", None) != 5:
            raise error
        path = Path(name)
        proof.unlinked_path(path, work)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & stat.S_IWRITE:
            raise error
        # Git may create pack/idx files without Windows' write bit. Never
        # chmod a reparse point or a file linked to another checkout.
        path.chmod(info.st_mode | stat.S_IWRITE)
        function(name)

    for attempt, delay in enumerate((0, 0.25, 1.0)):
        if delay:
            time.sleep(delay)
        proof.unlinked_path(work, temporary)
        if not work.exists():
            return
        if sys.platform != "win32":
            shutil.rmtree(work)
            return
        try:
            shutil.rmtree(work, onexc=unlink_readonly)
            return
        except OSError as error:
            # Access denied, sharing violation, lock violation. Some Git or
            # antivirus handles outlive their owner briefly on the runner.
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 2:
                raise


def execute(args, contract):
    backend, tier = args.backend, contract["tier"]
    measured = tier == ENG4B_REAL
    proof.require_host(backend, dict(os.environ), "nax" if backend == "mlx" else "hosted")
    tmp = Path(os.environ["RUNNER_TEMP"])
    artifact = proof.owned_output(args.output, tmp)
    receipt = {"schema": 1, "protocol": "flux1-ap13-v1", **contract, "accepted": False,
               "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
               "host": platform.platform(), "runner_name": os.environ["RUNNER_NAME"], "arms": {}, "logs": {}}
    work = tmp / f"f1p-{receipt['run_id']}-{receipt['run_attempt']}"
    sources = {"head": ROOT, "base": ROOT / "flux1-baseline"}
    created_work = False
    primary_error = None
    try:
        proof.unlinked_path(work, tmp)
        work.mkdir()
        created_work = True
        if git(sources["base"], "rev-parse", "HEAD") != contract.get("base", BASE):
            raise ValueError("baseline is not the profile's exact reviewed engine")
        if tier in fixture.ENG4B_TIERS and git(sources["base"], "rev-parse", "HEAD^{tree}") != contract["base_tree"]:
            raise ValueError("ENG-4b baseline tree differs from the reviewed merge")
        paths, receipt["fixtures"] = snapshots(
            work, tier, download_missing=contract["request"].get("download_missing", False))
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
        if measured:
            performance.require_matched_builds(receipt["arms"])
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
        claim_policy = policy
        claim_deadline = last_child_start = None
        if backend == "candle-cuda":
            budget, reserve = cuda_claim_budget(policy, tier)
            write(claim_dir / "policy.json", dict(contract["request"]["policy"], deadlineSeconds=budget))
            claim_policy = load_policy(claim_dir / "policy.json")
            claim_deadline = time.monotonic() + budget
            last_child_start = claim_deadline - reserve - policy.deadline_seconds
        with execution_claim(backend, claim_policy, claim_dir, lease_env,
                             original_deadline_monotonic=claim_deadline) as claim:
            receipt["physical_claim"] = claim
            receipt["preflight"] = {}
            for arm in planned_arms(tier):
                for variant in variants(tier):
                    key = preflight_key(tier, arm, variant)
                    out = work / ("preflight-" + key)
                    model = "flux1_" + variant
                    env = dict(environments[arm], PAIR_MODEL=model, PAIR_SNAPSHOT=str(paths[variant]))
                    _, stdout = invoke(exes[arm], work / arm, out, env, policy, lease_env, preflight=True,
                                       last_child_start=last_child_start)
                    plan_data = json.loads((out / "plan.json").read_text(encoding="utf-8"))
                    evidence.validate_plan(plan_data, variant, backend)
                    if stdout.splitlines().count("FLUX1_PAIR_PREFLIGHT " + model) != 1:
                        raise ValueError("missing compiled native preflight")
                    receipt["preflight"][key] = plan_data
            if measured:
                require_equal_plans(receipt["preflight"], tier)
            write(artifact / "receipt.json", receipt)
            for variant in variants(tier):
                for arm in evidence.ARMS:
                    out = work / (arm + "-" + variant)
                    env = dict(environments[arm], PAIR_MODEL="flux1_" + variant,
                               PAIR_SNAPSHOT=str(paths[variant]))
                    guard, stdout = invoke(exes[arm], work / arm, out, env, policy, lease_env,
                                           last_child_start=last_child_start)
                    planned = arm in planned_arms(tier)
                    observed = evidence.read_arm(out, stdout, arm, variant, backend, artifact, planned=planned)
                    if planned and observed["planned_route"] != receipt["preflight"][preflight_key(tier, arm, variant)]:
                        raise ValueError("render plan differs from preflight")
                    receipt["arms"][arm]["renders"][variant] = dict(observed, guard=asdict(guard))
                evidence.compare(*(receipt["arms"][arm]["renders"][variant] for arm in evidence.ARMS))
            if measured:
                receipt["performance_samples"] = {}
                for name, variant, arm in measured_cases(tier):
                    out = work / name
                    env = dict(environments[arm], PAIR_MODEL="flux1_" + variant,
                               PAIR_SNAPSHOT=str(paths[variant]))
                    invoke(exes[arm], work / arm, out, env, policy, lease_env, measure=True,
                           last_child_start=last_child_start)
                    reference = receipt["arms"][arm]["renders"][variant]
                    capture_measured_output(out, reference)
                    receipt["performance_samples"][name] = read_measured_sample(receipt, name, variant, arm, out)
        if not claim.get("post_execution_verified"):
            raise ValueError("physical release not verified")
        for variant in variants(tier):
            if fixture.snapshot_identity(paths[variant], variant, tier) != receipt["fixtures"][variant]:
                raise ValueError("snapshot changed during A/B")
        if measured:
            receipt["performance"] = performance.evaluate(receipt["performance_samples"], variants(tier))
            status = receipt["performance"]["status"]
            if status != "pass":
                raise ValueError("FLUX.1 terminal performance is " + status + " at the frozen sample budget")
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
        write(artifact / "receipt.json", receipt)
        if created_work:
            try:
                remove_source_worktrees(work, sources)
                remove_work(work, tmp)
            except (OSError, ValueError, subprocess.SubprocessError) as cleanup_error:
                receipt["cleanup_failure"] = str(cleanup_error)
                receipt["accepted"] = False
                write(artifact / "receipt.json", receipt)
                if primary_error is not None:
                    primary_error.add_note(f"Comparison cleanup also failed: {cleanup_error}")
                elif not failed_retention:
                    raise
        if failed_retention:
            raise ValueError("comparison log publication failed: " + ", ".join(errors))


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
    tier = contract["tier"]
    phases = phase_names(tier)
    limits = {phase + "." + suffix: cap for phase in phases
              for suffix, cap in retained_phase_limits(phase, contract["request"]["policy"]).items()}
    for name, item in r["logs"].items():
        if name not in limits or item["path"] != name:
            raise ValueError("unexpected comparison log identity")
        log = args.output / name
        proof.unlinked_path(log, tmp)
        if (not log.is_file() or log.stat().st_nlink != 1 or log.stat().st_size > limits[name]
                or log.stat().st_size != item["bytes"] or fixture.sha(log) != item["sha256"]):
            raise ValueError("invalid comparison log artifact: " + name)
        allowed.add(name)
    # guard.jsonl is the optional observer stream monitored by run_guarded.
    # The actual supervisor result is captured for every completed child.
    required = {name for name in limits if not name.endswith(".guard.jsonl")}
    completed = r["accepted"] or (tier == ENG4B_REAL and "performance" in r)
    if completed and (r.get("log_errors") or not required <= r["logs"].keys()):
        raise ValueError("accepted comparison lacks complete verified child logs")
    for arm in evidence.ARMS:
        for variant in variants(tier):
            row = r["arms"].get(arm, {}).get("renders", {}).get(variant)
            if not row and not completed:
                continue
            name = arm + "-" + variant + ".png"
            png = args.output / name
            proof.unlinked_path(png, tmp)
            if row["png"] != name or png.stat().st_size > 256*256*4 or fixture.sha(png) != row["png_sha256"]:
                raise ValueError("invalid image artifact")
            allowed.add(name)
    if {p.name for p in args.output.iterdir()} != allowed:
        raise ValueError("unexpected upload file")
    if completed:
        if not r["physical_claim"]["post_execution_verified"]:
            raise ValueError("unreleased accepted comparison")
        if tier in fixture.ENG4B_TIERS:
            verify_eng4b_identity(r, contract, args.backend)
        for variant in variants(tier):
            evidence.compare(*(r["arms"][arm]["renders"][variant] for arm in evidence.ARMS))
        if tier == ENG4B_REAL:
            samples = {name: read_measured_sample(r, name, variant, arm, args.output, published=True)
                       for name, variant, arm in measured_cases(tier)}
            report = performance.evaluate(samples, variants(tier))
            if r["performance_samples"] != samples or r["performance"] != report:
                raise ValueError("performance report disagrees with verified measurement artifacts")
            if r["accepted"] != (report["status"] == "pass" and "cleanup_failure" not in r):
                raise ValueError("terminal comparison acceptance disagrees with measured results")


def read_measured_sample(receipt, name, variant, arm, directory, *, published=False):
    preflight = receipt["preflight"][preflight_key(ENG4B_REAL, arm, variant)]
    reference = receipt["arms"][arm]["renders"][variant]
    policy = receipt["request"]["policy"]
    limits = retained_phase_limits(name, policy)

    def read(suffix):
        path = directory / (f"{name}.{suffix}" if published else suffix)
        proof.unlinked_path(path, directory)
        if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > limits[suffix]:
            raise ValueError("invalid measurement artifact: " + path.name)
        with path.open("rb") as source:
            return source.read(limits[suffix] + 1)

    return performance.read_sample(read, variant, preflight, reference, policy)


def require_equal_plans(plans, tier):
    for variant in variants(tier):
        if plans[preflight_key(tier, "base", variant)] != plans[preflight_key(tier, "head", variant)]:
            raise ValueError("terminal comparison requires equivalent base and head plans")


def verify_eng4b_identity(receipt, contract, backend):
    tier = contract["tier"]
    models = variants(tier)
    keys = {preflight_key(tier, arm, variant) for arm in evidence.ARMS for variant in models}
    if receipt["preflight"].keys() != keys or set(receipt["fixtures"]) != set(models):
        raise ValueError("ENG-4b receipt does not bind both plans and the exact fixture")
    if tier == ENG4B:
        fixture_row = receipt["fixtures"]["dev"]
        if (fixture_row["fixture_storage"] != dict(Q4=fixture.Q4, dense_vae="BF16")
                or fixture_row["tier"] != ENG4B or fixture_row["variant"] != "dev"
                or fixture_row["depths"] != [1, 1]):
            raise ValueError("ENG-4b fixture storage or depths differ from the profile")
    else:
        if receipt["fixtures"] != contract["pinned_fixtures"]:
            raise ValueError("terminal comparison snapshot differs from the published hashes")
        require_equal_plans(receipt["preflight"], tier)
        performance.require_matched_builds(receipt["arms"])
    for arm in evidence.ARMS:
        row = receipt["arms"][arm]
        build = row["build"]
        if (build["revision"] != contract[arm] or build["tree"] != contract[f"{arm}_tree"]
                or not all(re.fullmatch(r"[0-9a-f]{64}", build[field])
                           for field in ("overlay_sha256", "binary_sha256"))):
            raise ValueError("ENG-4b build identity differs from the paired revisions")
        for variant in models:
            plan = receipt["preflight"][preflight_key(tier, arm, variant)]
            evidence.validate_plan(plan, variant, backend)
            if row["renders"][variant]["planned_route"] != plan:
                raise ValueError("ENG-4b render plan differs from that arm's preflight")


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
