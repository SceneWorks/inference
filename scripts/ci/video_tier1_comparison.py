#!/usr/bin/env python3
"""Manual video migration Tier 1 proofs. Both revisions run on the same leased device."""
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
from scripts.ci import video_tier1_cohorts as cohorts
from scripts.ci import video_tier1_evidence as evidence
from scripts.ci.flux1_pair.fixture import sha
from scripts.ci.model_proof_resources import execution_claim
from scripts.media_campaign_supervisor import load_policy, validate_admission

BASE = "38dab65f5d9640396cf2946cd4ee866ce46630c6"
SELECTION = cohorts.WAN
FQN = cohorts.FQN["wan"]
ARMS = ("base", "head")
LANES = {"mlx": ["self-hosted", "macOS", "ARM64", "rw-mage"],
         "candle-cuda": proof.RUNNERS["candle-cuda"]}
CASES = {backend: cohorts.cases(SELECTION, backend) for backend in LANES}
GIB = 1024**3
MIB = 1024**2
SETTINGS = {
    "warmups": 1, "samples_per_child": 1, "pairs": 5, "seconds_ratio_ceiling": 1.02, "peak_ratio_ceiling": 1.02,
    "timing": "synchronized warm denoise call including in-call expert loads; mean seconds per completed outer step; median head/base",
    "interval": "min/max of five paired ratios; 93.75% median coverage for independent exchangeable pairs",
    "parity": "bit-identical finite terminal tensor and identical ordered events exposed by the production route",
    "allocator": "same physical device, one fresh child per arm in each adjacent AB/BA pair; :4096:8 cuBLAS workspace",
}
VARIANTS = {name: case.variant for name, case in cohorts.BY_NAME.items()}


def tracked(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def protocol(selection):
    return "wan-curated-tier1-v2" if selection == SELECTION else selection + "-v1"


def seal_name(selection):
    return "WAN_PAIR_RECEIPT_SHA256" if selection == SELECTION else "VIDEO_PAIR_RECEIPT_SHA256"


def measurements(case):
    for index in range(SETTINGS["pairs"]):
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        for arm in order:
            yield index, arm, f"{case}-p{index + 1}-{arm}"


def phases(cases):
    return tuple(name for case in cases for _, _, name in measurements(case))


def plan(backend, value, head, allowed=tuple(cohorts.CASES)):
    if (backend not in LANES or not isinstance(value, dict) or set(value) != {"selection", "base", "policy"}
            or value["selection"] not in allowed):
        raise ValueError("expected a fixed video Tier 1 proof selection")
    selection = value["selection"]
    base = BASE if selection == SELECTION else cohorts.VIDEO_BASE
    if base != value["base"] or head == base:
        raise ValueError("comparison requires its trusted distinct migration base")
    cases = cohorts.cases(selection, backend)
    source = proof.repository_identity(head)
    subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT, check=True)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "policy.json"
        proof.write_json(path, value["policy"])
        policy = load_policy(path)
    if (policy.backend != proof.POLICY_BACKENDS[backend] or policy.deadline_seconds > 180
            or policy.child_footprint_cap_bytes > 20 * GIB or policy.poll_millis > 250
            or policy.term_grace_millis > 10000 or policy.host_free_reserve_bytes < GIB
            or max(policy.stdout_cap_bytes, policy.stderr_cap_bytes, policy.event_cap_bytes) > 16 * MIB):
        raise ValueError("video Tier 1 exceeds the fixed child, logging, poll or reserve budget")
    if backend == "candle-cuda" and (policy.child_gpu_cap_bytes > 12 * GIB or policy.gpu_free_reserve_bytes < GIB):
        raise ValueError("video Tier 1 CUDA requires a GPU reserve and caps its child at 12 GiB")
    reserve = paired.RELEASE_RESERVE_SECONDS + (policy.term_grace_millis + policy.poll_millis) / 1000
    budget = (1 + len(phases(cases))) * policy.deadline_seconds + reserve
    # Bind both the random generator and its production source-header capture before
    # resource scheduling. Reads only tracked Git objects during the hosted plan.
    directory = cohorts.DIRECTORY[selection]
    files = {name: tracked(head, f"scripts/ci/{directory}/{name}") for name in ("headers.json", "fixture.py")}
    if selection != SELECTION:
        dependencies = ["flux1_pair/fixture.py", "wan_pair/fixture.py"]
        if selection == cohorts.MOCHI_SVD:
            dependencies.append("ms_pair/aux_shapes.json")
        else:
            dependencies.append("scail_pair/overlay.py")
        files.update({path: tracked(head, "scripts/ci/" + path) for path in dependencies})
    inputs = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    headers = json.loads(files["headers.json"])
    fixtures = {}
    for case in cases:
        variant = VARIANTS[case]
        header = headers[variant]
        if selection != SELECTION and set(header["geometry"]) != {cohorts.WEIGHT_FILE[variant]}:
            raise ValueError("fixture weight files differ from the fixed source layout")
        fixtures[case] = {"variant": variant, "source": header["source"],
                          "geometry_sha256": evidence.canonical_hash(header["geometry"]),
                          "files": sorted([*header["geometry"], "inputs.safetensors"])}
        if selection != SELECTION:
            descriptor = cohorts.BY_NAME[case]
            spec = evidence.case_spec(case)
            if spec["settings"]["seed"] != 104729 or spec["settings"]["depth"] != descriptor.depth:
                raise ValueError("unrecognized fixed case recipe")
            fixtures[case].update(depth=descriptor.depth, outer_steps=spec["outer_steps"],
                                  tensor_shape=spec["tensor_shape"], case_spec_sha256=evidence.canonical_hash(spec))
    toolchains = {}
    for arm, rev in (("base", base), ("head", head)):
        raw = tracked(rev, "rust-toolchain.toml")
        toolchains[arm] = {"declaration_sha256": hashlib.sha256(raw).hexdigest(),
                           "channel": tomllib.loads(raw.decode())["toolchain"]["channel"]}
    module_hashes = {arm: overlay_hashes(head, backend, arm, cases, selection) for arm in ARMS}
    return {"runner": LANES[backend], "asset_setup": selection, "backend": backend,
            "base": base, "base_tree": paired.git(ROOT, "rev-parse", base + "^{tree}"),
            "head": source["revision"], "head_tree": source["tree"], "request": value,
            "fixture_contracts": fixtures, "toolchain_contracts": toolchains, "fixture_inputs_sha256": inputs,
            "fixture_source_sha256": module_hashes, "cases": list(cases),
            "fqn": cohorts.FQN[cohorts.BY_NAME[cases[0]].model], "settings": SETTINGS,
            "claim_budget_seconds": budget, "release_reserve_seconds": reserve}


def case_row(backend, model="wan"):
    package = cohorts.package(backend, model)
    target = package.replace("-", "_")
    return {"package": package, "target": target, "target_kind": "lib", "fqn": cohorts.FQN[model],
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
        pool_token = b"@CUDA_POOL_WITNESS@"
        if blob.count(pool_token) == 1:
            blob = blob.replace(pool_token, tracked(head, "scripts/ci/wan_pair/cuda_pool.rs.in"))
        elif pool_token in blob or blob.count(b"fn pool_proof(pair: &Pair)") != 1:
            raise ValueError("fixture requires exactly one fixed CUDA allocator witness")
        token = b"@WAN_CANDLE_DENOISE_ARGS@"
        argument = b'Some("heun"), &request.cancel' if arm == "base" else b"&request, crate::dense_curated_phase(&request, knobs.seed)?.as_ref()"
    if blob.count(token) != 1:
        raise ValueError("fixture requires exactly one fixed private-signature binding")
    return blob.replace(token, argument)


def overlay_hashes(head, backend, arm, cases, selection):
    checksums = {}
    for model in cohorts.models(cases):
        module, blob, edits = cohorts.overlay_files(head, backend, arm, model, tracked, overlay_source)
        if selection == SELECTION:
            return hashlib.sha256(blob).hexdigest()
        for path, data in {module: blob, **edits}.items():
            checksums[path.as_posix()] = hashlib.sha256(data).hexdigest()
    return checksums


def checkout_and_overlay(work, contract, arm):
    backend = contract["backend"]
    root = work / arm
    revision = contract[arm]
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(root)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", revision], cwd=root, check=True)
    if (paired.git(root, "rev-parse", "HEAD") != revision
            or paired.git(root, "rev-parse", "HEAD^{tree}") != contract[arm + "_tree"]):
        raise ValueError("scratch proof source differs from the planned revision/tree")
    hashes = {}
    changed = set()
    for model in cohorts.models(contract["cases"]):
        module, blob, edits = cohorts.overlay_files(contract["head"], backend, arm, model, tracked, overlay_source)
        entry = module.parent / "lib.rs"
        for path in (module, entry, *edits):
            proof.unlinked_path(root / path, root)
        if (root / module).exists():
            raise ValueError("proof module conflicts with shipping source")
        (root / module).write_bytes(blob)
        hashes[module.as_posix()] = hashlib.sha256(blob).hexdigest()
        changed.add(module.as_posix())
        paired.git(root, "add", "-N", "--", module.as_posix())
        if edits:
            for path, data in edits.items():
                if not (root / path).is_file() or (root / path).stat().st_nlink != 1:
                    raise ValueError("instrumentation requires the existing production source path")
                (root / path).write_bytes(data.replace(b"\r\n", b"\n"))
                changed.add(path.as_posix())
                hashes[path.as_posix()] = hashlib.sha256((root / path).read_bytes()).hexdigest()
        else:
            decl = f"#[cfg(test)]\nmod {module.stem};".encode()
            if (root / entry).stat().st_nlink != 1 or decl in (root / entry).read_bytes():
                raise ValueError("duplicate proof module declaration")
            with (root / entry).open("ab") as handle:
                handle.write(b"\n" + decl + b"\n")
            changed.add(entry.as_posix())
    names = subprocess.check_output(["git", "diff", "--name-only", "-z", "HEAD"], cwd=root).decode()
    if set(names.rstrip("\0").split("\0")) != changed:
        raise ValueError("proof overlay changed files outside its exact approved paths")
    fixture_hash = next(iter(hashes.values())) if contract["request"]["selection"] == SELECTION else hashes
    if fixture_hash != contract["fixture_source_sha256"][arm]:
        raise ValueError("installed overlay differs from the planned Git source")
    delta = subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff", "HEAD"], cwd=root)
    return root, {"revision": revision, "tree": contract[arm + "_tree"],
                  "fixture_source_sha256": fixture_hash,
                  "overlay_sha256": hashlib.sha256(delta).hexdigest()}


def build(work, contract, arm, policy):
    root, record = checkout_and_overlay(work, contract, arm)
    env = build_env(policy)
    env["CARGO_TARGET_DIR"] = str(work / (arm + "-target"))
    toolchain = proof.verified_toolchain(env, root=root)
    binaries, targets = {}, {}
    for model in cohorts.models(contract["cases"]):
        row = case_row(contract["backend"], model)
        argv = proof.command(row, discover=True)
        result = subprocess.run(argv, cwd=root, env=env, stdout=subprocess.PIPE, text=True,
                                encoding="utf-8", check=True)
        discovery = proof.verify_discovery(row, result.stdout)
        binary = Path(discovery["executable"])
        if not binary.is_file() or not binary.resolve().is_relative_to(Path(env["CARGO_TARGET_DIR"]).resolve()):
            raise ValueError("libtest discovery did not select an owned target binary")
        binaries[model] = binary
        targets[model] = dict(discovery=discovery, build_command=argv, binary_sha256=sha(binary))
        verify_target(targets[model], contract["backend"], model, arm)
    record["toolchain"] = toolchain
    if contract["request"]["selection"] == SELECTION:
        record.update(targets["wan"])
    else:
        record["targets"] = targets
    return root, binaries, env, record


def fixture_directory(work, case):
    # The two A14B I2V residency settings load precisely the same bytes.
    return work / ("fixture-" + case.replace("-swapped-", "-resident-"))


def verify_fixture_identity(value, expected):
    fields = {"fixture_schema", "depth", "seed", "source", "variant", "files", "geometry_sha256"}
    if (not isinstance(value, dict) or set(value) != fields or type(value["fixture_schema"]) is not int
            or value["fixture_schema"] != 1 or type(value["depth"]) is not int or value["depth"] != expected.get("depth", 1)
            or value["seed"] != 104729 or value["source"] != expected["source"]
            or value["variant"] != expected["variant"] or value["geometry_sha256"] != expected["geometry_sha256"]
            or not isinstance(value["files"], dict) or sorted(value["files"]) != expected["files"]
            or any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d) for d in value["files"].values())):
        raise ValueError("fixture identity differs from committed case, geometry or source files")


def inspect_fixture(path, case, expected):
    identity = cohorts.fixture(cohorts.BY_NAME[case].model).fixture_identity(path, case)
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


def cuda_adapter(policy):
    from scripts.ci import wddm_process_memory
    return wddm_process_memory.preflight(policy.cuda_device_uuid)


def evaluate(observed, case, backend, policy, settings, stdout, guard, cuda_pool, adapter=None):
    spec = evidence.case_spec(case)
    facts = evidence.capture(observed["result"], bytes.fromhex(observed["initial_le_f32_hex"]),
                             bytes.fromhex(observed["terminal_le_f32_hex"]),
                             case, spec, settings, policy)
    body = proof.verify_execution(case_row(backend, cohorts.BY_NAME[case].model), stdout)
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
                or (guard["peak_gpu_bytes"] is not None and (not evidence.positive_int(guard["peak_gpu_bytes"])
                    or guard["peak_gpu_bytes"] > policy.child_gpu_cap_bytes))):
            raise ValueError("supervised GPU observations do not match the child/physical UUID")
        used, used_high, reserved, reserved_high = [cuda_pool[key] for key in
            ("poolUsedBytes", "poolUsedHighBytes", "poolReservedBytes", "poolReservedHighBytes")]
        if (not all(map(evidence.positive_int, (used, used_high, reserved, reserved_high)))
                or not used <= used_high <= reserved_high <= policy.child_gpu_cap_bytes
                or not used <= reserved <= reserved_high or allocator_peak != reserved_high):
            raise ValueError("invalid CUDA pool high-water evidence")
        peaks.update(cuda_pool_used_high_bytes=used_high, cuda_pool_reserved_high_bytes=reserved_high)
    if backend == "candle-cuda":
        process = evidence.cuda_process(guard, reserved_high, adapter)
        if (process.get("dedicated", {}).get("sampled_peak_bytes") or 0) > policy.child_gpu_cap_bytes:
            raise ValueError("observed dedicated process usage exceeded the child GPU cap")
        facts["cuda_process"] = process
    return dict(facts, body=body, peaks=peaks)


def execute(args, contract):
    selection = contract["request"]["selection"]
    backend = contract["backend"]
    proof.require_host(backend, os.environ, "nax" if backend == "mlx" else "hosted")
    if os.environ.get("MODEL_PROOF_RUNNER_ENVIRONMENT") != "self-hosted":
        raise ValueError("video paired proofs require the self-hosted resource lease")
    temporary = Path(os.environ["RUNNER_TEMP"])
    output = proof.owned_output(args.output, temporary)
    receipt = {"protocol": protocol(selection), **contract, "accepted": False,
               "host": platform.platform(), "runner_name": os.environ["RUNNER_NAME"],
               "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
               "builds": {}, "fixtures": {}, "observations": {}, "execution_order": [], "comparisons": {}, "logs": {}}
    prefix = "wan" if selection == SELECTION else "video"
    work = temporary / f"{prefix}-{receipt['run_id']}-{receipt['run_attempt']}"
    proof.unlinked_path(work, temporary)
    work.mkdir()
    try:
        policy = read_policy(work, "policy.json", contract["request"]["policy"])
        receipt["hardware"] = paired.hardware(backend, policy)
        adapter = None
        if backend == "candle-cuda":
            adapter = receipt["dedicated_adapter"] = cuda_adapter(policy)
        paths = {}
        for case in contract["cases"]:
            descriptor = cohorts.BY_NAME[case]
            path = (fixture_directory(work, case) if selection == SELECTION else work / ("fixture-" + descriptor.variant))
            if path not in paths.values():
                cohorts.fixture(descriptor.model).make_fixture(path, case)
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
                    model = cohorts.BY_NAME[case].model
                    root, binaries, build_environment = builds[arm]
                    binary = binaries[model]
                    env = {k: v for k, v in build_environment.items() if k.upper() in RUNTIME_ENV}
                    prefix = "WAN" if selection == SELECTION else "VIDEO"
                    env.update({prefix + "_PAIR_CASE": case, prefix + "_PAIR_FIXTURE": str(paths[case]),
                                "MODEL_INFERENCE_CASE_FQN": cohorts.FQN[model]})
                    out = work / name
                    started = time.monotonic_ns()
                    paired.invoke(binary, root, out, env, policy, lease_env, last_child_start=last_start,
                                  arguments=(cohorts.FQN[model], "--exact", "--ignored", "--nocapture", "--test-threads=1"),
                                  capture_cuda_basis=True, dedicated_adapter=adapter)
                    ended = time.monotonic_ns()
                    receipt["execution_order"].append({"phase": name, "start_ns": started, "end_ns": ended})
                    observed = read_observed(out, case)
                    info = {"observed": observed}
                    guard = json.loads(evidence.owned_file(out, "supervisor.json", 16 * MIB).read_bytes())
                    pool = (json.loads(evidence.owned_file(out, "cuda-pool.json", 4096).read_bytes())
                            if backend == "candle-cuda" else None)
                    stdout = evidence.owned_file(out, "stdout.log", policy.stdout_cap_bytes).read_text(encoding="utf-8")
                    info["result"] = evaluate(observed, case, backend, policy, SETTINGS, stdout, guard, pool, adapter)
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
        errors = paired.retain_logs(work, output, receipt, phases(contract["cases"]))
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
                bindings.write(seal_name(selection) + "=" + sha(output / "receipt.json") + "\n")
        if errors:
            raise ValueError("failed to retain bounded proof logs: " + ", ".join(errors))


def verify_target(build, backend, model, arm):
    row = case_row(backend, model)
    discovery = build["discovery"]
    target = discovery["target"]
    # Dependency-qualified features appear in the command, crate-local ones in the artifact.
    features = {"model-inference-proof"}
    if backend == "candle-cuda":
        features.update(("default", "cuda"))
    elif model == "wan":
        features.update(("default", "training"))
    path_type = PureWindowsPath if backend == "candle-cuda" else PurePosixPath
    exe = path_type(discovery["executable"])
    if (build["build_command"] != proof.command(row, discover=True)
            or any(target.get(k) != v for k, v in row["cargo_target_metadata"].items())
            or set(discovery["features"]) != features or discovery.get("model_bodies_executed") != 0
            or not str(discovery.get("package_id", "")).endswith(
                ("#" + row["package"] + "@0.0.0", "/" + row["package"] + "#0.0.0"))
            or not exe.is_absolute() or exe.parent.parts[-3:] != (arm + "-target", "debug", "deps")
            or not re.fullmatch(re.escape(row["target"]) + r"-[a-f0-9]+" + (r"\.exe" if backend == "candle-cuda" else ""), exe.name)
            or not re.fullmatch(r"[0-9a-f]{64}", build["binary_sha256"])):
        raise ValueError("accepted proof target or measured binary mismatch")


def check_complete_provenance(receipt, contract):
    from scripts.ci.model_proof_resources import STRICT_CUDA_ACTIVITY

    backend = contract["backend"]
    if (set(receipt["builds"]) != set(ARMS) or receipt.get("log_errors") or receipt.get("cleanup_failure")
            or set(receipt["fixtures"]) != set(contract["cases"]) or receipt["runner_name"] != os.environ["RUNNER_NAME"]
            or receipt["host"] != platform.platform()):
        raise ValueError("accepted proof lacks source builds, complete publication or its actual host identity")
    shared_fixtures = {}
    for case in contract["cases"]:
        observed = receipt["fixtures"][case]
        verify_fixture_identity(observed, contract["fixture_contracts"][case])
        if observed != shared_fixtures.setdefault(observed["variant"], observed):
            raise ValueError("cases sharing a frozen fixture used different bytes")
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
        if receipt.get("dedicated_adapter") != cuda_adapter(policy):
            raise ValueError("WDDM process counter source differs from the selected physical adapter")
        baseline = claim["admission_baseline"]
        if (not hardware["name"] or not hardware["driver_version"] or int(hardware["memory.total"]) <= 0
                or hardware["uuid"] != policy.cuda_device_uuid or int(hardware["index"]) != claim.get("physical_index")
                or claim.get("device_uuid") != policy.cuda_device_uuid or baseline.get("uuid") != policy.cuda_device_uuid
                or not isinstance(baseline.get("process_identities"), list) or type(baseline.get("reserved_baseline_mib")) is not int
                or claim.get("mechanism") != "existing PUBLIC/SceneWorks/gpu-locks UUID lock"
                or claim.get("activity_profile") != STRICT_CUDA_ACTIVITY or not claim.get("idle_policy")):
            raise ValueError("missing exact CUDA claim, stable idle admission or device identity")
    # The verifier runs in the same job after scratch cleanup, with the pinned compiler.
    tc = proof.verified_toolchain(build_env(policy), root=ROOT)
    models = cohorts.models(contract["cases"])
    for arm in ARMS:
        build = receipt["builds"][arm]
        version = contract["toolchain_contracts"][arm]
        toolchain = build["toolchain"]
        if (any(toolchain.get(k) != v for k, v in version.items())
                or any(not toolchain.get(k) or toolchain[k] != tc[k] for k in
                       ("cargo_version", "rustc_version", "binaries", "declaration_sha256", "channel"))
                or build["revision"] != contract[arm] or build["tree"] != contract[arm + "_tree"]
                or build["fixture_source_sha256"] != contract["fixture_source_sha256"][arm]
                or not re.fullmatch(r"[0-9a-f]{64}", build["overlay_sha256"])):
            raise ValueError("accepted proof toolchain, source or overlay mismatch")
        targets = {"wan": build} if contract["request"]["selection"] == SELECTION else build["targets"]
        if set(targets) != set(models):
            raise ValueError("accepted proof package set differs from the hosted plan")
        for model in models:
            verify_target(targets[model], backend, model, arm)
    actual = receipt["execution_order"]
    if not isinstance(actual, list) or [r.get("phase") for r in actual] != list(phases(contract["cases"])):
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
    if hashlib.sha256(raw).hexdigest() != os.environ.get(seal_name(contract["request"]["selection"])):
        raise ValueError("artifact differs from supervisor receipt bound in this job")
    receipt = json.loads(raw)
    if (receipt.get("protocol") != protocol(contract["request"]["selection"]) or type(receipt.get("accepted")) is not bool
            or any(receipt.get(k) != v for k, v in contract.items())
            or receipt.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or receipt.get("run_attempt") != os.environ["GITHUB_RUN_ATTEMPT"]):
        raise ValueError("video comparison artifact contract/run mismatch")
    backend = contract["backend"]
    limits = paired.retained_log_limits(contract["request"]["policy"])
    if backend == "candle-cuda":
        limits["cuda-pool.json"] = 4096
    entries = {f"{phase}.{suffix}": cap for phase in phases(contract["cases"]) for suffix, cap in limits.items()}
    logs = receipt["logs"]
    if not logs.keys() <= entries.keys() or set(p.name for p in output.iterdir()) != {"receipt.json", *logs}:
        raise ValueError("unexpected proof upload contents")
    for name, saved in logs.items():
        path = evidence.owned_file(output, name, entries[name])
        if saved != {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}:
            raise ValueError("proof log identity mismatch")
    if backend == "candle-cuda" and receipt.get("observations"):
        from scripts.ci import wddm_process_memory
        wddm_process_memory.validate_adapter(receipt.get("dedicated_adapter"), contract["request"]["policy"]["cudaDeviceUuid"])
    if receipt["accepted"]:
        check_complete_provenance(receipt, contract)
    required = {name for name in entries if not name.endswith(".guard.jsonl")}
    observations = receipt["observations"]
    if not observations.keys() <= set(phases(contract["cases"])):
        raise ValueError("unregistered observation")
    if receipt["accepted"] and (receipt.get("failure") or not required <= logs.keys()
            or set(observations) != set(phases(contract["cases"])) or not receipt["physical_claim"]["post_execution_verified"]
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
                                    json.loads(read("cuda-pool.json")) if backend == "candle-cuda" else None,
                                    receipt.get("dedicated_adapter"))
                if observed != row["result"]:
                    raise ValueError("recorded measurements differ from raw observations")
                results[arm].append(observed)
            if all(len(arm) == SETTINGS["pairs"] for arm in results.values()):
                result = evidence.compare(results["base"], results["head"], SETTINGS)
                if receipt["comparisons"].get(case) != result or (receipt["accepted"] and not result["accepted"]):
                    raise ValueError("unverified paired conclusion")


def main(allowed=(cohorts.MOCHI_SVD, cohorts.SCAIL)):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--backend", choices=LANES, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = plan(args.backend, json.loads(os.environ[args.request_env]), args.expected_sha, allowed)
    if args.plan:
        print(json.dumps(contract))
    elif evidence.canonical_hash(contract) != os.environ.get("VIDEO_TIER1_PLAN_SHA256"):
        raise ValueError("runtime plan differs from the validated hosted dispatch")
    elif args.output is None:
        parser.error("execution and verification require an owned output path")
    elif args.verify_artifacts:
        verify(args, contract)
    else:
        execute(args, contract)


if __name__ == "__main__":
    main()
