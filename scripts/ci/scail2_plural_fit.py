#!/usr/bin/env python3
"""One candidate-only, explicitly guarded SCAIL-2 plural qualification campaign per native backend."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import flux1_comparison as paired, model_inference_proofs as proof
from scripts.ci import video_tier1_comparison as tier1, video_tier1_evidence as evidence
from scripts.ci import wddm_process_memory as wddm
from scripts.ci.model_proof_resources import execution_claim, proof_cuda_shared_release, validate_shared_release_record
from scripts.ci.scail_plural import contract as c
from scripts.ci.scail_plural.cuda_stage import KeepSystemGrowth, RenderEnd
from scripts.ci.scail_plural.wddm_retry import SameTickSamples
from scripts.media_campaign_supervisor import SupervisionError, SystemProbe


def row(backend):
    return dict(tier1.case_row(backend, "scail2"), fqn=c.FQN)


def source(backend, cuda_uuid=None):
    directory = ROOT / "scripts/ci/scail_plural"
    shim = (directory / ("mlx.rs.in" if backend == "mlx" else "cuda.rs.in")).read_text()
    if backend == "candle-cuda":
        shim = shim.replace("@POOL@", (ROOT / "scripts/ci/wan_pair/cuda_pool.rs.in").read_text())
    gpu0 = backend == "candle-cuda" and cuda_uuid == c.GPU0
    return (
        (directory / "probe.rs.in")
        .read_text()
        .replace("@CHECK_CUDA_DEVICE@", "    let cuda_device = cuda_identity()?;\n" if gpu0 else "")
        .replace("@RECORD_CUDA_DEVICE@", '    let mut result = result;\n    result["cuda_device"] = cuda_device;\n' if gpu0 else "")
        .replace("@BACKEND@", shim)
        .replace("@CARRIERS@", json.dumps(c.CARRIERS, separators=(",", ":")))
        .replace("@PINS@", ",\n".join(f'("{n}","{d}")' for n, (_, d) in c.FILES.items()))
    ) + ((directory / "cuda_identity.rs.in").read_text() if gpu0 else "")


def plan(backend, request, head):
    identity = proof.repository_identity(head)
    devices = c.CUDA_INDICES if backend == "candle-cuda" else (None,)
    continuation = (backend == "candle-cuda" and c.digest_json(request) == c.digest_json(
        {"selection": c.SELECTION, "policy": c.policy(backend, cuda_uuid=c.GPU0), "continuation": c.CONTINUATION}))
    if not continuation and not any(
        c.digest_json(request) == c.digest_json({"selection": c.SELECTION, "policy": c.policy(backend, cuda_uuid=uuid)})
        for uuid in devices
    ):
        raise ValueError(
            "plural campaign requires its complete fixed count set and exact guard policy"
        )
    return {
        "protocol": c.PROTOCOL,
        "head": identity["revision"],
        "tree": identity["tree"],
        "request": request,
        "backend": backend,
        "runner": tier1.LANES[backend],
        "counts": [2, 3, 4, 5] if continuation else c.COUNTS[backend],
        **({"recovered_six": c.RECOVERED_SIX} if continuation else {}),
        **({"cuda_completion": "supervisor-owned-windows-job"}
           if not continuation and request["policy"].get("cudaDeviceUuid") == c.GPU0 else {}),
        "asset_setup": "hub-snapshots",
        "hub_snapshots": {c.SLOT: {"repository": c.REPO, "revision": c.REV, "subfolder": "q4"}},
        "fixture_module_sha256": hashlib.sha256(source(backend, request["policy"].get("cudaDeviceUuid")).encode()).hexdigest(),
        "inventory_sha256": c.INVENTORY,
        "files": c.FILES,
        "case": row(backend),
    }


class Observer:
    def __init__(self, cuda, approved):
        self.base = cuda if cuda is not None else SystemProbe(approved)
        self.host_baseline = None
        self.host_samples = []
        self.card_samples = []
        self.windows_peak = {}
        self.render_end = RenderEnd() if cuda else None
        self.shared = None
        if cuda is not None:
            cuda.system = KeepSystemGrowth(cuda.system)
            cuda.dedicated = SameTickSamples(cuda.dedicated, approved.poll_millis)
            self.shared = SameTickSamples(
                wddm.DedicatedSamples(cuda.dedicated_adapter, approved.cuda_device_uuid, shared=True),
                approved.poll_millis,
            )

    def __getattr__(self, name):
        return getattr(self.base, name)

    def tree_footprint(self, owner):
        value = self.base.tree_footprint(owner)
        if value and self.host_baseline is None:
            self.host_baseline = value
        if value:
            self.host_samples.append([time.monotonic_ns(), value])
            if self.shared:
                counters = owner.last_memory_counters
                if counters["owned_process_count"] != 1:
                    raise ValueError("plural render spawned an unexpected process")
                for key, observed in counters.items():
                    self.windows_peak[key] = max(self.windows_peak.get(key, 0), observed)
        return value

    def gpu_free_and_tree_bytes(self, owner):
        before = self.render_end.before(owner)
        attempted_before = self.base.unknown_pid_polls
        try:
            free, used = self.base.gpu_free_and_tree_bytes(owner)
        except SupervisionError as error:
            # The watchdog can tolerate a failed NVIDIA tick. Account for the
            # shared gauge on that tick just as PairCudaProbe does for dedicated.
            if self.shared is not None:
                self.shared.poll(owner, cuda_active=True)
            if self.base.unknown_pid_polls != attempted_before:
                self.render_end.reject(owner, self.base, before, len(self.card_samples), error)
            raise
        self.shared.poll(owner, cuda_active=bool(self.base.dedicated.record["samples"]))
        at_ns = time.monotonic_ns()
        self.render_end.observe(owner, self.base, before, len(self.card_samples), at_ns, used)
        self.card_samples.append([at_ns, free, used])
        return free, used

    def close(self):
        if self.shared:
            self.shared.close()
            self.base.close()

    def fit_observation(self):
        return {
            "host_baseline": self.host_baseline,
            "host_samples_monotonic_ns_bytes": self.host_samples,
            "card_samples_monotonic_ns_free_owned_or_fallback_bytes": self.card_samples,
            "windows_peak": self.windows_peak,
            "wddm_shared_counter": "GPU Process Memory/Shared Usage" if self.shared else None,
            "wddm_shared": self.shared.record if self.shared else None,
            **({"cuda_render_end": self.render_end.record, "wddm_retry": {"dedicated": self.base.dedicated.diagnostics,
                               "shared": self.shared.diagnostics}} if self.shared else {}),
        }


def build(work, contract, policy):
    backend = contract["backend"]
    scratch = work / "src"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(scratch)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", contract["head"]], cwd=scratch, check=True
    )
    if paired.git(scratch, "rev-parse", "HEAD^{tree}") != contract["tree"]:
        raise ValueError("wrong checkout tree")
    package = contract["case"]["package"]
    lib = (
        scratch
        / "crates/media"
        / ("mlx-gen" if backend == "mlx" else "candle-gen")
        / package
        / "src/lib.rs"
    )
    with lib.open("a", encoding="utf8") as f:
        f.write("\n#[cfg(test)]\nmod plural_fit;\n")
    native = lib.with_name("plural_fit.rs")
    native.write_bytes(source(backend, policy.cuda_device_uuid).encode())
    if tier1.sha(native) != contract["fixture_module_sha256"]:
        raise ValueError("wrong native fixture")
    env = tier1.build_env(policy)
    env["CARGO_TARGET_DIR"] = str(work / "target")
    toolchain = proof.verified_toolchain(env, root=scratch)
    cmd = proof.command(contract["case"], discover=True)
    cmd.insert(2, "--release")
    result = subprocess.run(
        cmd, cwd=scratch, env=env, stdout=subprocess.PIPE, encoding="utf8", check=True
    )
    discovery = proof.verify_discovery(contract["case"], result.stdout)
    binary = Path(discovery["executable"])
    if not binary.resolve().is_relative_to((work / "target").resolve()):
        raise ValueError("test executable escaped owned build root")
    return (
        scratch,
        binary,
        env,
        {
            "command": cmd,
            "discovery": discovery,
            "binary_sha256": tier1.sha(binary),
            "toolchain": toolchain,
        },
    )


def measured_outcome(native, guard, pool, adapter, hw, backend, count, approved, head, *, completed_job=None, fresh_cuda=False):
    if fresh_cuda and (completed_job is None or completed_job[1] != "owned-job-completion"):
        raise ValueError("fresh GPU 0 qualification requires the supervised Windows Job")
    if completed_job is not None:
        claim, basis = completed_job
        if backend != "candle-cuda" or approved.cuda_device_uuid != c.GPU0:
            raise ValueError("completed Job proof requires physical GPU 0")
        require_release(claim, backend, hw, approved)
        if basis == "owned-job-completion" and count in (c.COUNTS["candle-cuda"] if fresh_cuda else (2, 3, 4, 5)):
            validate_shared_release_record(claim, [{"supervision": guard}],
                dict(backend=backend, **c.cuda_constraints(c.GPU0)))
            if claim["owned_execution_completion"]["waited_returncode"] != 0:
                raise ValueError("owned Windows Job did not exit successfully")
        elif not (basis == "archived-supervisor-return" and count == 6 and head == c.RECOVERED_SIX["inference_revision"]):
            raise ValueError("unexpected completed Job evidence source")
    try:
        candidate = c.measured_row(native, guard, pool, adapter, hw, backend, count, approved, head, completed_job=completed_job, fresh_cuda=fresh_cuda)
    except c.IncompleteWddmSamples:
        return {"accepted": False, "telemetry_failure": "incomplete_wddm_samples"}
    return {"accepted": True, "candidate_row_pending_physical_review": candidate}


def require_release(claim, backend, hardware, policy):
    if claim.get("post_execution_verified") is not True or (
        backend == "candle-cuda" and claim.get("lease_released") is not True
    ):
        raise ValueError("physical resource release unverified")
    if backend == "candle-cuda":
        c.validate_cuda_hardware(hardware, policy.cuda_device_uuid)
        if claim.get("device_uuid") != policy.cuda_device_uuid or type(claim.get("physical_index")) is not int or (
            claim["physical_index"] != c.CUDA_INDICES[policy.cuda_device_uuid]
        ):
            raise ValueError("release belongs to a different physical CUDA device")
        # Replay the prospective GPU 0 admission and release samples. Earlier GPU 1
        # archives keep their original release contract.
        if policy.cuda_device_uuid == c.GPU0:
            tier1.verify_cuda_claim(claim, hardware, policy)
            release = claim["release_background_activity"]
            samples = release["samples"]
            verification = claim["release_metadata_verification"]
            status = verification["status"]
            if not (release["sample_count"] == len(samples) and (
                (status == "verified" and len(samples) == 5)
                or (status == "inconclusive-driver-query-timing" and 1 <= len(samples) <= 5
                    and verification["diagnostic"] == release)
            )):
                raise ValueError("missing selected CUDA release samples")
            for number, sample in enumerate(samples, 1):
                if not (sample["sample_number"] == number and sample["predicates"]["unique_uuid"] is True
                        and sample["selected_index_matches_admission"] is True
                        and sample["post_owned_exit_primary_metadata_valid"] is True
                        and proof_cuda_shared_release({"uuid":policy.cuda_device_uuid, **sample["selected_card"]},
                            policy.cuda_device_uuid, claim["physical_index"], int(hardware["memory.total"]))):
                    raise ValueError("release sample belongs to a different physical CUDA device")


def execute(args, contract):
    backend = contract["backend"]
    proof.require_host(backend, os.environ, "nax")
    if os.environ.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("plural fits require an owned physical execution lease")
    temp = Path(os.environ["RUNNER_TEMP"])
    dest = proof.owned_output(args.output, temp)
    work = temp / f"scail-plural-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    proof.unlinked_path(work, temp)
    work.mkdir()
    phases = [f"count-{n}" for n in contract["counts"]]
    fresh_cuda = "cuda_completion" in contract
    require_job = fresh_cuda or contract["request"].get("continuation") == c.CONTINUATION
    receipt = {
        "contract": contract,
        "request": contract["request"],
        "accepted": False,
        "counts": {},
        "logs": {},
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "runner": os.environ["RUNNER_NAME"],
        "coverage": {
            "status": "native observations; coordinator physical review required before import",
            "allocator": "native lifetime high-water",
            "host": "Darwin kernel lifetime footprint or Windows Job private commit and process working set",
            "cuda_nonpool": "owned WDDM dedicated/shared and tenant-aware child growth; absolute card occupancy includes external pressure",
            "cuda_nonpool_unsampled_overshoot_upper_bytes": None,
        },
    }
    try:
        approved = tier1.read_policy(work, "policy.json", contract["request"]["policy"])
        hw = receipt["hardware"] = paired.hardware(backend, approved, pci_bus=approved.cuda_device_uuid == c.GPU0)
        adapter = receipt["adapter"] = (
            tier1.cuda_adapter(approved) if backend == "candle-cuda" else None
        )
        # The runs-on labels choose the Mac; RUNNER_NAME is recorded provenance only.
        if backend == "mlx" and hw["chip"] != "Apple M5 Max":
            raise ValueError("wrong proof Mac hardware")
        if backend == "candle-cuda":
            c.validate_cuda_hardware(hw, approved.cuda_device_uuid)
        expected_device = c.cuda_identity(hw, adapter) if approved.cuda_device_uuid == c.GPU0 else None
        stage = json.loads(Path(os.environ["MODEL_PROOF_HUB_SNAPSHOT_RECEIPT"]).read_text())[
            "snapshots"
        ][c.SLOT]
        snapshot = Path(os.environ[c.SLOT])
        c.validate_stage(stage, snapshot)
        receipt["stage"] = stage
        root, binary, build_env, receipt["build"] = build(work, contract, approved)
        for count, name in zip(contract["counts"], phases):
            claim_dir = work / (name + "-claim")
            claim_dir.mkdir()
            claim_policy = tier1.read_policy(
                claim_dir, "policy.json", dict(contract["request"]["policy"], deadlineSeconds=2520)
            )
            record = receipt["counts"][str(count)] = {"accepted": False}
            deadline = time.monotonic() + 2520
            with execution_claim(
                backend, claim_policy, claim_dir, build_env, original_deadline_monotonic=deadline,
                require_owned_completion=require_job,
                constraints=c.cuda_constraints(approved.cuda_device_uuid) if backend == "candle-cuda" else None,
            ) as claim:
                record["claim"] = claim
                env = {k: v for k, v in build_env.items() if k.upper() in tier1.RUNTIME_ENV}
                # The paired timing harness fixes cuBLAS's workspace for bit comparison.
                # Normal resident worker qualification uses the native default allocator.
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
                env.update(
                    {
                        c.SLOT: str(snapshot),
                        "SCAIL_PLURAL_COUNT": str(count),
                        "MODEL_INFERENCE_CASE_FQN": c.FQN,
                    }
                )
                if expected_device is not None:
                    env["SCAIL_EXPECTED_CUDA_DEVICE"] = json.dumps(expected_device)
                if backend == "mlx":
                    env["WAN_VAE_BUDGET_GIB"] = "28"
                out = work / name
                _, stdout = paired.invoke(
                    binary,
                    root,
                    out,
                    env,
                    approved,
                    build_env,
                    last_child_start=deadline - 2430,
                    arguments=["--exact", c.FQN, "--ignored", "--nocapture", "--test-threads=1"],
                    capture_cuda_basis=backend == "candle-cuda",
                    dedicated_adapter=adapter,
                    observer=Observer, completion_claim=claim if require_job else None,
                )
                native = json.loads(evidence.owned_file(out, "native.json", c.MIB).read_bytes())
                record["native"] = native
                record["body"] = proof.verify_execution(contract["case"], stdout)
                guard = json.loads(
                    evidence.owned_file(out, "supervisor.json", 16 * c.MIB).read_bytes()
                )
                pool = (
                    json.loads(evidence.owned_file(out, "cuda-pool.json", 4096).read_bytes())
                    if adapter
                    else None
                )
            require_release(claim, backend, hw, approved)
            record.update(measured_outcome(
                native, guard, pool, adapter, hw, backend, count, approved, contract["head"],
                completed_job=(claim, "owned-job-completion") if require_job else None, fresh_cuda=fresh_cuda,
            ))
        receipt["accepted"] = all(record["accepted"] for record in receipt["counts"].values())
        if not receipt["accepted"]:
            receipt["failure"] = "incomplete owned WDDM poll series"
    except BaseException as exc:
        receipt["failure"] = proof.redact(str(exc), dict(os.environ))
        raise
    finally:
        errors = paired.retain_logs(work, dest, receipt, phases)
        if errors:
            receipt.update(accepted=False, log_errors=errors)
        paired.remove_work(work, temp)
        proof.write_json(dest / "receipt.json", receipt)
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf8") as f:
            f.write("SCAIL_PLURAL_RECEIPT_SHA256=" + tier1.sha(dest / "receipt.json") + "\n")
    if not receipt["accepted"]:
        raise ValueError("incomplete plural memory measurements")


def verify(args, contract):
    dest = args.output
    proof.unlinked_path(dest, Path(os.environ["RUNNER_TEMP"]))
    file = evidence.owned_file(dest, "receipt.json", 8 * c.MIB)
    if tier1.sha(file) != os.environ.get("SCAIL_PLURAL_RECEIPT_SHA256"):
        raise ValueError("unsealed producer receipt")
    receipt = json.loads(file.read_bytes())
    if (
        c.digest_json(contract) != c.digest_json(receipt["contract"])
        or receipt["request"] != contract["request"]
        or receipt["run_id"] != os.environ["GITHUB_RUN_ID"]
        or receipt["run_attempt"] != os.environ["GITHUB_RUN_ATTEMPT"]
    ):
        raise ValueError("receipt belongs to another plan")
    backend = contract["backend"]
    limits = paired.retained_log_limits(receipt["request"]["policy"])
    if backend == "candle-cuda":
        limits["cuda-pool.json"] = 4096
    allowed = {
        f"count-{n}.{suffix}": cap for n in contract["counts"] for suffix, cap in limits.items()
    }
    if not set(receipt["logs"]).issubset(allowed) or set(p.name for p in dest.iterdir()) != set(
        receipt["logs"]
    ) | {"receipt.json"}:
        raise ValueError("unexpected evidence files")
    for name, published in receipt["logs"].items():
        if tier1.sha(evidence.owned_file(dest, name, allowed[name])) != published["sha256"]:
            raise ValueError("changed published evidence")
    records = receipt["counts"]
    # The producer serializes JSON keys in sorted order; replay the declared
    # schedule rather than treating the order of a JSON object as evidence.
    attempted = contract["counts"][:len(records)]
    if set(records) != {str(count) for count in attempted}:
        raise ValueError("attempted counts are not a prefix of the physical campaign")
    prefixes = {f"count-{count}" for count in attempted}
    if any(name.partition(".")[0] not in prefixes for name in receipt["logs"]):
        raise ValueError("evidence belongs to an unattempted count")
    if receipt["accepted"]:
        if attempted != contract["counts"] or not all(record["accepted"] is True for record in records.values()):
            raise ValueError("incomplete campaign")
    elif not (receipt.get("failure") or receipt.get("log_errors")):
        raise ValueError("unexplained partial campaign")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        approved = tier1.read_policy(Path(d), "policy.json", contract["request"]["policy"])
    fresh_cuda = "cuda_completion" in contract
    require_job = fresh_cuda or contract["request"].get("continuation") == c.CONTINUATION
    for count in attempted:
        record = records[str(count)]
        if type(record["accepted"]) is not bool:
            raise ValueError("invalid count outcome")
        if record["accepted"] is False and "telemetry_failure" not in record:
            if count != attempted[-1]:
                raise ValueError("attempted another count after a hard failure")
            if "candidate_row_pending_physical_review" in record:
                raise ValueError("unverified count exposes a physical price")
            continue  # hard failure before a verified measurement
        name = f"count-{count}"
        required = {f"{name}.{suffix}" for suffix in limits if suffix != "guard.jsonl"}
        if not required.issubset(receipt["logs"]):
            raise ValueError("missing completed child evidence")
        require_release(record["claim"], backend, receipt["hardware"], approved)
        guard = json.loads((dest / (name + ".supervisor.json")).read_bytes())
        pool = (
            json.loads((dest / (name + ".cuda-pool.json")).read_bytes())
            if receipt["adapter"]
            else None
        )
        body = proof.verify_execution(contract["case"], (dest / (name + ".stdout.log")).read_text())
        actual = measured_outcome(
            record["native"], guard, pool, receipt["adapter"], receipt["hardware"],
            backend, count, approved, contract["head"], completed_job=(record["claim"], "owned-job-completion")
            if require_job else None, fresh_cuda=fresh_cuda,
        )
        stored = {key: record[key] for key in (
            "accepted", "telemetry_failure", "candidate_row_pending_physical_review"
        ) if key in record}
        if record["body"] != body or actual != stored:
            raise ValueError("physical/model receipt changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--backend", choices=c.COUNTS, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = plan(args.backend, json.loads(os.environ[args.request_env]), args.expected_sha)
    if args.plan:
        print(json.dumps(contract))
        return
    if not args.output or c.digest_json(contract) != os.environ.get("VIDEO_TIER1_PLAN_SHA256"):
        raise ValueError("campaign differs from hosted plan")
    (verify if args.verify_artifacts else execute)(args, contract)


if __name__ == "__main__":
    main()
