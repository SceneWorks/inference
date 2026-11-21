#!/usr/bin/env python3
"""Manual Wan curated Tier 1 proof. Both revisions run on the same leased device."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import platform
import re
import subprocess
import sys
import time
import tomllib
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import flux1_comparison as paired
from scripts.ci import model_inference_proofs as proof
from scripts.ci import wan_tier1_evidence as evidence
from scripts.ci.flux1_pair.fixture import sha
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import load_policy, validate_admission

BASE = "38dab65f5d9640396cf2946cd4ee866ce46630c6"
SELECTION = "wan-curated-tier1"
FQN = "wan_pair_fixture::curated_tier1"
ARMS = ("base", "head")
LANES = {"mlx": ["self-hosted", "macOS", "ARM64", "rw-mage"],
         "candle-cuda": proof.RUNNERS["candle-cuda"]}
CASES = {
    "mlx": ("mlx-a14b-t2v-resident-heun", "mlx-a14b-i2v-resident-heun",
            "mlx-a14b-i2v-swapped-heun", "mlx-ti2v5b-t2v-heun"),
    "candle-cuda": ("cuda-ti2v5b-t2v-heun",),
}
PACKAGE = {"mlx": "mlx-gen-wan", "candle-cuda": "candle-gen-wan"}
GIB = 1024**3
MIB = 1024**2
SETTINGS = {
    "warmups": 1, "samples_per_child": 1, "pairs": 5, "seconds_ratio_ceiling": 1.02, "peak_ratio_ceiling": 1.02,
    "timing": "synchronized warm denoise call including in-call expert loads; mean seconds per completed outer step; median head/base",
    "interval": "min/max of five paired ratios; 93.75% median coverage for independent exchangeable pairs",
    "parity": "bit-identical finite terminal tensor and identical ordered events exposed by the production route",
    "allocator": "same physical device, one fresh child per arm in each adjacent AB/BA pair; :4096:8 cuBLAS workspace",
}
MODULE = "wan_pair_fixture.rs"
VARIANTS = dict(zip((*CASES["mlx"], *CASES["candle-cuda"]),
                    ("mlx-a14b-t2v", "mlx-a14b-i2v", "mlx-a14b-i2v", "mlx-ti2v5b", "cuda-ti2v5b"), strict=True))


def measurements(case):
    for index in range(SETTINGS["pairs"]):
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        for arm in order:
            yield index, arm, f"{case}-p{index + 1}-{arm}"


def phases(backend):
    return tuple(name for case in CASES[backend] for _, _, name in measurements(case))


def plan(backend, value, head):
    if (backend not in LANES or not isinstance(value, dict) or set(value) != {"selection", "base", "policy"}
            or value["selection"] != SELECTION or value["base"] != BASE or head == BASE):
        raise ValueError("expected the fixed Wan curated Tier 1 selection and a distinct revision")
    source = proof.repository_identity(head)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=ROOT, check=True)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "policy.json"
        proof.write_json(path, value["policy"])
        policy = load_policy(path)
    if (policy.backend != proof.POLICY_BACKENDS[backend] or policy.deadline_seconds > 180
            or policy.child_footprint_cap_bytes > 20 * GIB or policy.poll_millis > 250
            or policy.term_grace_millis > 10000 or policy.host_free_reserve_bytes < GIB
            or max(policy.stdout_cap_bytes, policy.stderr_cap_bytes, policy.event_cap_bytes) > 16 * MIB):
        raise ValueError("Wan Tier 1 exceeds the fixed child, logging, poll or reserve budget")
    if backend == "candle-cuda" and (policy.child_gpu_cap_bytes > 12 * GIB or policy.gpu_free_reserve_bytes < GIB):
        raise ValueError("Wan Tier 1 CUDA requires a GPU reserve and caps its child at 12 GiB")
    reserve = paired.RELEASE_RESERVE_SECONDS + (policy.term_grace_millis + policy.poll_millis) / 1000
    budget = (1 + len(phases(backend))) * policy.deadline_seconds + reserve
    # Bind both the random generator and its production source-header capture before
    # resource scheduling. Reads only tracked Git objects during the hosted plan.
    def tracked(revision, path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    files = {name: tracked(head, f"scripts/ci/wan_pair/{name}") for name in ("headers.json", "fixture.py")}
    inputs = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    headers = json.loads(files["headers.json"])
    fixtures = {}
    for case in CASES[backend]:
        variant = VARIANTS[case]
        header = headers[variant]
        fixtures[case] = {"variant": variant, "source": header["source"],
                          "geometry_sha256": evidence.canonical_hash(header["geometry"]),
                          "files": sorted([*header["geometry"], "inputs.safetensors"])}
    toolchains = {}
    for arm, rev in (("base", BASE), ("head", head)):
        raw = tracked(rev, "rust-toolchain.toml")
        toolchains[arm] = {"declaration_sha256": hashlib.sha256(raw).hexdigest(),
                           "channel": tomllib.loads(raw.decode())["toolchain"]["channel"]}
    module_hashes = {arm: hashlib.sha256(overlay_source(head, backend, arm)).hexdigest() for arm in ARMS}
    return {"runner": LANES[backend], "asset_setup": SELECTION, "backend": backend,
            "base": BASE, "base_tree": paired.git(ROOT, "rev-parse", BASE + "^{tree}"),
            "head": source["revision"], "head_tree": source["tree"], "request": value,
            "fixture_contracts": fixtures, "toolchain_contracts": toolchains, "fixture_inputs_sha256": inputs, "fixture_source_sha256": module_hashes, "cases": list(CASES[backend]), "fqn": FQN, "settings": SETTINGS,
            "claim_budget_seconds": budget, "release_reserve_seconds": reserve}


def case_row(backend):
    package = PACKAGE[backend]
    target = package.replace("-", "_")
    return {"package": package, "target": target, "target_kind": "lib", "fqn": FQN,
            "cargo_features": ["model-inference-proof", "cuda" if backend == "candle-cuda" else "mlx-gen/test-support"],
            "cargo_target_metadata": {"name": target, "kind": ["lib"], "crate_types": ["lib"]},
            "backend": backend, "expected_devices": ["mlx-metal" if backend == "mlx" else "candle-cuda"],
            "execution_role": "direct"}


# The dispatch supplies no environment overrides. Keep just the runner toolchain, native build
# and dynamic-loader prerequisites. Both children receive the same fixed render settings.
BUILD_ENV = {
    "PATH", "HOME", "USER", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PUBLIC",
    "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA",
    "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "CARGO_HOME", "RUSTUP_HOME", "MODEL_PROOF_RUST_BIN",
    "CARGO_BUILD_JOBS", "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG", "CARGO_PROFILE_DEV_STRIP",
    "CARGO_PROFILE_TEST_STRIP", "CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP",
    "CARGO_PROFILE_DEV_DEBUG_ASSERTIONS", "CARGO_PROFILE_TEST_DEBUG_ASSERTIONS", "CUDA_PATH", "CUDA_HOME",
    "LIB", "INCLUDE", "LIBPATH", "VCTOOLSINSTALLDIR", "WINDOWSSDKDIR", "WINDOWSSDKVERSION",
    "VCINSTALLDIR", "VSINSTALLDIR", "UNIVERSALCRTSDKDIR", "UCRTVERSION", "VSCMD_ARG_TGT_ARCH",
    "VSCMD_ARG_HOST_ARCH", "RUNNER_OS", "RUNNER_NAME", "MODEL_PROOF_RUNNER_ENVIRONMENT",
}
RUNTIME_ENV = {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP",
               "APPDATA", "LOCALAPPDATA", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG",
               "PMETAL_METALLIB_PATH"}


def build_env(policy):
    env = {key: value for key, value in os.environ.items() if key.upper() in BUILD_ENV}
    env.update(CARGO_INCREMENTAL="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if policy.backend == "windows-cuda":
        env.update(CUDA_VISIBLE_DEVICES=policy.cuda_device_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    else:
        env["MACOSX_DEPLOYMENT_TARGET"] = "26.2"
    return env


def overlay_source(head, backend, arm):
    template = "scripts/ci/wan_pair/" + ("mlx_probe.rs.in" if backend == "mlx" else "cuda_probe.rs.in")
    blob = subprocess.check_output(["git", "show", f"{head}:{template}"], cwd=ROOT)
    common = subprocess.check_output(["git", "show", f"{head}:scripts/ci/wan_pair/common.rs.in"], cwd=ROOT)
    if blob.count(b"@WAN_PAIR_COMMON@") != 1:
        raise ValueError("unrecognized Wan fixture template")
    blob = blob.replace(b"@WAN_PAIR_COMMON@", common)
    # Only the migrated private-call signatures differ. Neither arm can select another
    # sampler, request or execution path through dispatch data.
    if backend == "mlx":
        token = b"@WAN_MLX_SWAP_ARM@"
        argument = arm.encode("ascii")
    else:
        token = b"@WAN_CANDLE_DENOISE_ARGS@"
        argument = b'Some("heun"), &request.cancel' if arm == "base" else b"&request, crate::dense_curated_phase(&request, knobs.seed)?.as_ref()"
    if blob.count(token) != 1:
        raise ValueError("fixture requires exactly one fixed private-signature binding")
    return blob.replace(token, argument)


def checkout_and_overlay(work, contract, arm):
    backend = contract["backend"]
    root = work / arm
    revision = contract[arm]
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(root)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", revision], cwd=root, check=True)
    if (paired.git(root, "rev-parse", "HEAD") != revision
            or paired.git(root, "rev-parse", "HEAD^{tree}") != contract[arm + "_tree"]):
        raise ValueError("scratch proof source differs from the planned revision/tree")
    pkg = PACKAGE[backend]
    source = Path("crates", "media", pkg.removesuffix("-wan"), pkg, "src")
    module = source / MODULE
    # Read Git objects at the verified candidate, never untracked workspace contents. The only
    # base source change is the exact test-only declaration appended to the crate root.
    blob = overlay_source(contract["head"], backend, arm)
    decl = b"#[cfg(test)]\nmod wan_pair_fixture;"
    entry = root / source / "lib.rs"
    if (root / module).exists() or decl in entry.read_bytes():
        raise ValueError("scratch crate already has a conflicting Wan fixture")
    (root / module).write_bytes(blob)
    with entry.open("ab") as handle:
        handle.write(b"\n" + decl + b"\n")
    paired.git(root, "add", "-N", "--", module.as_posix())
    delta = subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff", "HEAD"], cwd=root)
    return root, {"revision": revision, "tree": contract[arm + "_tree"],
                  "fixture_source_sha256": hashlib.sha256(blob).hexdigest(),
                  "overlay_sha256": hashlib.sha256(delta).hexdigest()}


def build(work, contract, arm, policy):
    root, record = checkout_and_overlay(work, contract, arm)
    env = build_env(policy)
    env["CARGO_TARGET_DIR"] = str(work / (arm + "-target"))
    toolchain = proof.verified_toolchain(env, root=root)
    row = case_row(contract["backend"])
    argv = proof.command(row, discover=True)
    result = subprocess.run(argv, cwd=root, env=env, stdout=subprocess.PIPE, text=True,
                            encoding="utf-8", check=True)
    discovery = proof.verify_discovery(row, result.stdout)
    binary = Path(discovery["executable"])
    if not binary.is_file() or not binary.resolve().is_relative_to(Path(env["CARGO_TARGET_DIR"]).resolve()):
        raise ValueError("libtest discovery did not select an owned target binary")
    record.update(toolchain=toolchain, discovery=discovery, build_command=argv, binary_sha256=sha(binary))
    return root, binary, env, record


def fixture_directory(work, case):
    # The two A14B I2V residency settings load precisely the same bytes.
    return work / ("fixture-" + case.replace("-swapped-", "-resident-"))


def verify_fixture_identity(value, expected):
    fields = {"fixture_schema", "depth", "seed", "source", "variant", "files", "geometry_sha256"}
    if (not isinstance(value, dict) or set(value) != fields or type(value["fixture_schema"]) is not int
            or value["fixture_schema"] != 1 or type(value["depth"]) is not int or value["depth"] != 1
            or value["seed"] != 104729 or value["source"] != expected["source"]
            or value["variant"] != expected["variant"] or value["geometry_sha256"] != expected["geometry_sha256"]
            or not isinstance(value["files"], dict) or sorted(value["files"]) != expected["files"]
            or any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d) for d in value["files"].values())):
        raise ValueError("fixture identity differs from committed case, geometry or source files")


def inspect_fixture(path, case, expected):
    from scripts.ci.wan_pair import fixture
    identity = fixture.fixture_identity(path, case)
    geometry = {}
    for name in expected["files"]:
        candidate = path / name
        proof.unlinked_path(candidate, path)
        if not candidate.is_file() or candidate.stat().st_nlink != 1:
            raise ValueError("unexpected linked fixture file")
        if name == "inputs.safetensors":
            continue
        with candidate.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if not 0 < size < 16 * MIB:
                raise ValueError("invalid synthesized weight header")
            tensor_header = json.loads(handle.read(size))
        geometry[name] = {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in tensor_header.items()}
    identity["geometry_sha256"] = evidence.canonical_hash(geometry)
    verify_fixture_identity(identity, expected)
    return identity


def read_policy(directory, name, raw):
    path = directory / name
    proof.write_json(path, raw)
    return load_policy(path)


def read_observed(output, case):
    raw_result = json.loads(evidence.owned_file(output, "result.json", 512 * 1024).read_bytes())
    spec = evidence.case_spec(case)
    size = math.prod(spec["tensor_shape"]) * 4
    initial = evidence.owned_file(output, "state-0.f32", size).read_bytes()
    terminal = evidence.owned_file(output, f"state-{spec['outer_steps']}.f32", size).read_bytes()
    # Serialize the tiny finite tensor as hex so the verifier can repeat byte-level validation
    # from the upload. Never publish an arbitrary file name, weight or opaque output directory.
    return {"result": raw_result, "initial_le_f32_hex": initial.hex(), "terminal_le_f32_hex": terminal.hex()}


def evaluate(observed, case, backend, policy, settings, stdout, guard, cuda_pool):
    spec = evidence.case_spec(case)
    facts = evidence.capture(observed["result"], bytes.fromhex(observed["initial_le_f32_hex"]),
                             bytes.fromhex(observed["terminal_le_f32_hex"]),
                             case, spec, settings, policy)
    body = proof.verify_execution(case_row(backend), stdout)
    if (type(guard["returncode"]) is not int or guard["returncode"] != 0
            or type(guard.get("elapsed_seconds")) not in (int, float) or not math.isfinite(guard["elapsed_seconds"])
            or sum(sample["elapsed_ns"] for sample in facts["samples"]) / 1e9 > guard["elapsed_seconds"]
            or not evidence.positive_int(guard["peak_host_bytes"])
            or guard["peak_host_bytes"] > policy.child_footprint_cap_bytes
            or guard["admission"]["policySha256"] != policy.sha256):
        raise ValueError("supervisor did not record an admitted bounded execution")
    validate_admission(guard["admission"], policy_sha256=policy.sha256)
    peaks = {"process_footprint_bytes": guard["peak_host_bytes"]}
    allocator_peak = observed["result"]["allocator_peak_bytes"]
    if backend == "mlx":
        if allocator_peak > policy.child_footprint_cap_bytes:
            raise ValueError("observed Metal allocator peak exceeded child policy")
        peaks["mlx_allocator_peak_bytes"] = allocator_peak
    if backend == "candle-cuda":
        if (cuda_pool["pid"] != guard["pid"] or cuda_pool["cudaDeviceUuid"] != policy.cuda_device_uuid
                or not evidence.positive_int(guard["peak_gpu_bytes"])
                or guard["peak_gpu_bytes"] > policy.child_gpu_cap_bytes):
            raise ValueError("supervised GPU observations do not match the child/physical UUID")
        used, used_high, reserved, reserved_high = [cuda_pool[key] for key in
            ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")]
        if (not all(map(evidence.positive_int, (used, used_high, reserved, reserved_high)))
                or not used <= used_high <= reserved_high <= policy.child_gpu_cap_bytes
                or not used <= reserved <= reserved_high or allocator_peak != reserved_high):
            raise ValueError("invalid CUDA pool high-water evidence")
        peaks.update(cuda_pool_used_high_bytes=used_high, cuda_pool_reserved_high_bytes=reserved_high)
    if backend == "candle-cuda":
        facts["cuda_process"] = evidence.cuda_process(guard, reserved_high)
    return dict(facts, body=body, peaks=peaks)


def execute(args, contract):
    from scripts.ci.wan_pair import fixture

    backend = contract["backend"]
    proof.require_host(backend, os.environ, "nax" if backend == "mlx" else "hosted")
    if os.environ.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("Wan paired proofs require the self-hosted resource lease")
    temporary = Path(os.environ["RUNNER_TEMP"])
    output = proof.owned_output(args.output, temporary)
    receipt = {"protocol": "wan-curated-tier1-v2", **contract, "accepted": False,
               "host": platform.platform(), "runner_name": os.environ["RUNNER_NAME"],
               "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
               "builds": {}, "fixtures": {}, "observations": {}, "execution_order": [], "comparisons": {}, "logs": {}}
    work = temporary / f"wan-{receipt['run_id']}-{receipt['run_attempt']}"
    proof.unlinked_path(work, temporary)
    work.mkdir()
    try:
        policy = read_policy(work, "policy.json", contract["request"]["policy"])
        receipt["hardware"] = paired.hardware(backend, policy)
        paths = {}
        for case in contract["cases"]:
            path = fixture_directory(work, case)
            if path not in paths.values():
                fixture.make_fixture(path, case)
            paths[case] = path
            receipt["fixtures"][case] = inspect_fixture(path, case, contract["fixture_contracts"][case])
        builds = {}
        for arm in ARMS:
            root, binary, env, receipt["builds"][arm] = build(work, contract, arm, policy)
            builds[arm] = root, binary, env
        if any(receipt["builds"][arm]["fixture_source_sha256"] != contract["fixture_source_sha256"][arm]
               for arm in ARMS):
            raise ValueError("base/head overlay differs from its fixed template and private signature")
        claim_dir = work / "claim"
        claim_dir.mkdir()
        budget = contract["claim_budget_seconds"]
        claim_policy = read_policy(claim_dir, "policy.json", dict(contract["request"]["policy"], deadlineSeconds=budget))
        deadline = time.monotonic() + budget
        last_start = deadline - policy.deadline_seconds - contract["release_reserve_seconds"]
        lease_env = builds["base"][2]
        with execution_claim(backend, claim_policy, claim_dir, lease_env,
                             original_deadline_monotonic=deadline) as claim:
            receipt["physical_claim"] = claim
            for case in contract["cases"]:
                results = {arm: [] for arm in ARMS}
                for _, arm, name in measurements(case):
                    root, binary, build_environment = builds[arm]
                    env = {k: v for k, v in build_environment.items() if k.upper() in RUNTIME_ENV}
                    env.update(WAN_PAIR_CASE=case, WAN_PAIR_FIXTURE=str(paths[case]), MODEL_INFERENCE_CASE_FQN=FQN)
                    out = work / name
                    started = time.monotonic_ns()
                    paired.invoke(binary, root, out, env, policy, lease_env, last_child_start=last_start,
                                  arguments=(FQN, "--exact", "--ignored", "--nocapture", "--test-threads=1"),
                                  capture_cuda_basis=True)
                    ended = time.monotonic_ns()
                    receipt["execution_order"].append({"phase": name, "start_ns": started, "end_ns": ended})
                    observed = read_observed(out, case)
                    info = {"observed": observed}
                    guard = json.loads(evidence.owned_file(out, "supervisor.json", 16 * MIB).read_bytes())
                    pool = (json.loads(evidence.owned_file(out, "cuda-pool.json", 4096).read_bytes())
                            if backend == "candle-cuda" else None)
                    stdout = evidence.owned_file(out, "stdout.log", policy.stdout_cap_bytes).read_text(encoding="utf-8")
                    info["result"] = evaluate(observed, case, backend, policy, SETTINGS, stdout, guard, pool)
                    receipt["observations"][name] = info
                    results[arm].append(info["result"])
                receipt["comparisons"][case] = evidence.compare(results["base"], results["head"], SETTINGS)
                if receipt["fixtures"][case] != inspect_fixture(paths[case], case, contract["fixture_contracts"][case]):
                    raise ValueError("fixture content changed across the adjacent measurements")
        if not claim.get("post_execution_verified"):
            raise ValueError("physical claim did not verify device release")
        receipt["accepted"] = all(r["accepted"] for r in receipt["comparisons"].values())
        if not receipt["accepted"]:
            raise ValueError("paired parity or performance failed or was inconclusive; see per-case comparisons")
    except BaseException as error:
        receipt["failure"] = proof.redact(str(error), dict(os.environ))
        raise
    finally:
        errors = paired.retain_logs(work, output, receipt, phases(backend))
        if errors:
            receipt.update(accepted=False, log_errors=errors)
        proof.write_json(output / "receipt.json", receipt)
        try:
            paired.remove_work(work, temporary)
        except OSError as error:
            receipt.update(accepted=False, cleanup_failure=str(error))
            proof.write_json(output / "receipt.json", receipt)
            raise
        finally:
            # This workflow-owned channel is never inherited by either build or learned child.
            # Bind the exact producer receipt to the subsequent verification/upload step.
            with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as bindings:
                bindings.write("WAN_PAIR_RECEIPT_SHA256=" + sha(output / "receipt.json") + "\n")
        if errors:
            raise ValueError("failed to retain bounded proof logs: " + ", ".join(errors))


def check_complete_provenance(receipt, contract):
    from scripts.ci.model_proof_resources import STRICT_CUDA_ACTIVITY

    backend = contract["backend"]
    if (set(receipt["builds"]) != set(ARMS) or receipt.get("log_errors") or receipt.get("cleanup_failure")
            or set(receipt["fixtures"]) != set(contract["cases"]) or receipt["runner_name"] != os.environ["RUNNER_NAME"]
            or receipt["host"] != platform.platform()):
        raise ValueError("accepted proof lacks source builds, complete publication or its actual host identity")
    for case in contract["cases"]:
        verify_fixture_identity(receipt["fixtures"][case], contract["fixture_contracts"][case])
    if backend == "mlx" and receipt["fixtures"][CASES["mlx"][1]] != receipt["fixtures"][CASES["mlx"][2]]:
        raise ValueError("resident/swapped I2V cases used different fixture bytes")
    claim = receipt["physical_claim"]
    if claim.get("post_execution_verified") is not True or claim.get("runner") != receipt["runner_name"]:
        raise ValueError("unverified runner or physical device release")
    with TemporaryDirectory() as directory:
        policy = read_policy(Path(directory), "policy.json", contract["request"]["policy"])
    hardware = receipt["hardware"]
    if hardware != paired.hardware(backend, policy):
        raise ValueError("physical hardware differs from live runner")
    if backend == "mlx":
        from scripts.ci.metal_execution_lock import CANONICAL_PATH, PROTOCOL
        if (not hardware["chip"] or not evidence.positive_int(hardware["physical_memory_bytes"]) or not hardware["os_build"]
                or claim.get("protocol") != PROTOCOL or claim.get("path") != str(CANONICAL_PATH)
                or not all(evidence.positive_int(claim.get(k)) for k in ("pid", "inode", "device"))
                or type(claim.get("borrowed")) is not bool or not claim.get("census")):
            raise ValueError("missing physical Metal lease, census or hardware identity")
    else:
        baseline = claim["admission_baseline"]
        if (not hardware["name"] or not hardware["driver_version"] or int(hardware["memory.total"]) <= 0
                or hardware["uuid"] != policy.cuda_device_uuid or int(hardware["index"]) != claim.get("physical_index")
                or claim.get("device_uuid") != policy.cuda_device_uuid or baseline.get("uuid") != policy.cuda_device_uuid
                or not isinstance(baseline.get("process_identities"), list) or type(baseline.get("reserved_baseline_mib")) is not int
                or claim.get("mechanism") != "existing PUBLIC/SceneWorks/gpu-locks UUID lock"
                or claim.get("activity_profile") != STRICT_CUDA_ACTIVITY or not claim.get("idle_policy")):
            raise ValueError("missing exact CUDA claim, stable idle admission or device identity")
    # The verifier runs in the same job after scratch cleanup. Re-query the actual compiler and
    # hardware. These two Wan revisions use the same pinned native toolchain on this physical host.
    tc = proof.verified_toolchain(build_env(policy), root=ROOT)
    row = case_row(backend)
    path_type = PureWindowsPath if backend == "candle-cuda" else PurePosixPath
    for arm in ARMS:
        build = receipt["builds"][arm]
        target = build["discovery"]["target"]
        features = set(build["discovery"]["features"])
        # Cargo lists enabled features of the selected crate; dependency-qualified feature
        # mlx-gen/test-support is covered by the exact invocation, not this crate-local list.
        expected_features = ({"default", "training", "model-inference-proof"} if backend == "mlx"
                             else {"default", "cuda", "model-inference-proof"})
        exe = path_type(build["discovery"]["executable"])
        toolchain = build["toolchain"]
        version = contract["toolchain_contracts"][arm]
        if (any(toolchain.get(k) != v for k, v in version.items())
                or any(not toolchain.get(k) or toolchain[k] != tc[k] for k in
                       ("cargo_version", "rustc_version", "binaries", "declaration_sha256", "channel"))
                or build["revision"] != contract[arm] or build["tree"] != contract[arm + "_tree"]
                or build["fixture_source_sha256"] != contract["fixture_source_sha256"][arm]
                or build["build_command"] != proof.command(row, discover=True)
                or any(target.get(k) != v for k, v in row["cargo_target_metadata"].items())
                or features != expected_features or build["discovery"].get("model_bodies_executed") != 0
                or not str(build["discovery"].get("package_id", "")).endswith(
                    ("#" + row["package"] + "@0.0.0", "/" + row["package"] + "#0.0.0"))
                or not exe.is_absolute() or exe.parent.parts[-3:] != (arm + "-target", "debug", "deps")
                or not re.fullmatch(re.escape(row["target"]) + r"-[a-f0-9]+" + (r"\.exe" if backend == "candle-cuda" else ""), exe.name)
                or not all(isinstance(build.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", build[k])
                           for k in ("binary_sha256", "overlay_sha256"))):
            raise ValueError("accepted proof target, toolchain, source or measured binary mismatch")
    actual = receipt["execution_order"]
    if not isinstance(actual, list) or [r.get("phase") for r in actual] != list(phases(backend)):
        raise ValueError("measurement order differs from fixed adjacent counterbalanced pairs")
    previous = 0
    for measurement in actual:
        start, end = measurement["start_ns"], measurement["end_ns"]
        if not (evidence.positive_int(start) and evidence.positive_int(end) and previous <= start < end):
            raise ValueError("overlapping or unordered process timings")
        previous = end


def verify(args, contract):
    temporary = Path(os.environ["RUNNER_TEMP"])
    output = args.output
    if (not output.is_absolute() or ".." in output.parts or output.resolve() == temporary.resolve()
            or not output.resolve().is_relative_to(temporary.resolve())):
        raise ValueError("proof output must be a strict descendant of the owned runner temporary directory")
    proof.unlinked_path(output, temporary)
    raw = evidence.owned_file(output, "receipt.json", 16 * MIB).read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ.get("WAN_PAIR_RECEIPT_SHA256"):
        raise ValueError("artifact differs from supervisor receipt bound in this job")
    receipt = json.loads(raw)
    if (receipt.get("protocol") != "wan-curated-tier1-v2" or type(receipt.get("accepted")) is not bool
            or any(receipt.get(k) != v for k, v in contract.items())
            or receipt.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or receipt.get("run_attempt") != os.environ["GITHUB_RUN_ATTEMPT"]):
        raise ValueError("Wan comparison artifact contract/run mismatch")
    backend = contract["backend"]
    limits = paired.retained_log_limits(contract["request"]["policy"])
    if backend == "candle-cuda":
        limits["cuda-pool.json"] = 4096
    entries = {f"{phase}.{suffix}": cap for phase in phases(backend) for suffix, cap in limits.items()}
    logs = receipt["logs"]
    if not logs.keys() <= entries.keys() or set(p.name for p in output.iterdir()) != {"receipt.json", *logs}:
        raise ValueError("unexpected proof upload contents")
    for name, saved in logs.items():
        path = evidence.owned_file(output, name, entries[name])
        if saved != {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}:
            raise ValueError("proof log identity mismatch")
    if receipt["accepted"]:
        check_complete_provenance(receipt, contract)
    required = {name for name in entries if not name.endswith(".guard.jsonl")}
    observations = receipt["observations"]
    if not observations.keys() <= set(phases(backend)):
        raise ValueError("unregistered observation")
    if receipt["accepted"] and (receipt.get("failure") or not required <= logs.keys()
            or set(observations) != set(phases(backend)) or not receipt["physical_claim"]["post_execution_verified"]
            or set(receipt["fixtures"]) != set(contract["cases"]) or set(receipt["comparisons"]) != set(contract["cases"])):
        raise ValueError("accepted comparison lacks logs, fixture identity, measurements or release")
    with TemporaryDirectory() as directory:
        policy = read_policy(Path(directory), "policy.json", contract["request"]["policy"])
        for case in contract["cases"]:
            results = {arm: [] for arm in ARMS}
            for _, arm, name in measurements(case):
                row = observations.get(name)
                if row is None or "result" not in row:
                    if receipt["accepted"]:
                        raise ValueError("missing completed observation")
                    continue
                def read(suffix):
                    filename = name + "." + suffix
                    if filename not in logs:
                        raise ValueError("missing evidence for completed observation")
                    return (output / filename).read_text(encoding="utf-8")
                observed = evaluate(row["observed"], case, backend, policy, SETTINGS, read("stdout.log"),
                                    json.loads(read("supervisor.json")),
                                    json.loads(read("cuda-pool.json")) if backend == "candle-cuda" else None)
                if observed != row["result"]:
                    raise ValueError("recorded measurements differ from raw observations")
                results[arm].append(observed)
            if all(len(arm) == SETTINGS["pairs"] for arm in results.values()):
                result = evidence.compare(results["base"], results["head"], SETTINGS)
                if receipt["comparisons"].get(case) != result or (receipt["accepted"] and not result["accepted"]):
                    raise ValueError("unverified paired conclusion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--backend", choices=LANES, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = plan(args.backend, json.loads(os.environ[args.request_env]), args.expected_sha)
    if args.plan:
        print(json.dumps(contract))
    elif args.output is None:
        parser.error("execution and verification require an owned output path")
    elif args.verify_artifacts:
        verify(args, contract)
    else:
        execute(args, contract)


if __name__ == "__main__":
    main()
