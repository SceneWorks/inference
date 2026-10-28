#!/usr/bin/env python3
"""One owner-bound remote iOS source bundle: three compiles, exact lists, optional pure HOSTs.
No provisioning, weights, GPU/native bodies, signals, timeouts, cleanup or automatic retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

PACKAGES = (("mlx-gen", "mlx_gen"), ("mlx-gen-z-image", "mlx_gen_z_image"),
            ("mlx-gen-ios-catalog", "mlx_gen_ios_catalog"))
FORBIDDEN_MARKERS = ("ZIMAGE_CPU_GUARDS_READY_V1", "ZIMAGE_THREE_ROOT_LOWER_CPU_GUARDS_READY_V1",
                     "ZIMAGE_THREE_ROOT_LOWER_POINT_V1", "MLX_ROOT_METADATA_CPU_GUARDS_READY_V1")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def process_rows():
    text = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss=,command="], text=True, encoding="utf-8")
    return {int(f[0]): (int(f[1]), int(f[2]) * 1024, f[3])
            for line in text.splitlines() if len(f := line.strip().split(None, 3)) == 4}

def descendants(rows, parent):
    found = {parent}
    while True:
        more = {pid for pid, (ppid, _, _) in rows.items() if ppid in found}
        if more <= found:
            return found
        found |= more

def external_jobs(rows):
    own = descendants(rows, os.getpid())
    return [{"pid": pid, "command": command} for pid, (_, _, command) in rows.items()
            if pid not in own and not any(s in command for s in ("zsh -c ", "bash -c ", "zsh -lc ", "bash -lc "))
            and (re.search(r"(?:^|/)(?:cargo|rustc|clang|clang\+\+|c\+\+|cmake|ninja|make)(?:\s|$)", command)
                 or "/debug/deps/mlx" in command)]

def parse_log(text):
    messages = []
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    diagnostics = [m["message"] for m in messages if m.get("reason") == "compiler-message"]
    warnings = [d for d in diagnostics if d.get("level") == "warning"]
    errors = [d for d in diagnostics if d.get("level") == "error"]
    plain = re.findall(r"(?mi)^\s*(?:warning|error)(?:\[[^\]]+\])?:.*$", text)
    rust = re.findall(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", text)
    return messages, warnings, errors, plain, rust

def validate_host(step, text, rust):
    expected = step["expected_Rust_rows"]
    assert len(rust) == expected and all(r[:5] == ("ok", "1", "0", "0", "0") for r in rust), "wrong HOST counts"
    assert len(re.findall(r"test " + re.escape(step["fqn"]) + r" \.\.\.", text)) == expected, "wrong HOST FQN"
    assert not any(marker in text for marker in FORBIDDEN_MARKERS), "HOST entered a native fixture"
    if expected == 2:
        assert step.get("pure_HOST_fork_protocol"), "undeclared fork"
        assert step["fqn"].endswith("actual_legacy_listener_enabled_and_disabled_preserves_typed_causes")
        # The pinned parent creates a fresh absolute DONE, waits status.success, then reads b"ok".
        # Do not infer completion from an ignored/default test or an invented marker.

def prerequisite_probe(manifest, report, deadline):
    """Read only named prerequisites; never select an alternate SDK or launch Cargo."""
    result = {"checks": {}, "SDKs": [], "discovery": [], "no_compile_or_test": True,
              "automatic_SDK_substitution": False}
    checks = result["checks"]
    def check(name, observe, accepts):
        try:
            value = observe()
            checks[name] = {"value": value, "ok": bool(accepts(value))}
        except Exception as error:
            checks[name] = {"ok": False, "error": repr(error)}
        return checks[name]
    home = Path.home()
    check("Python", lambda: {"executable": sys.executable, "version": list(sys.version_info[:3])},
          lambda v: v["version"][:2] == [3, 12])
    check("platform", lambda: [platform.system(), platform.machine()],
          lambda v: v[0] == "Darwin" and v[1] in ("arm64", "aarch64"))
    for name, path in (("cargo", home / ".cargo/bin/cargo"),
                       ("rustc", home / ".rustup/toolchains/1.96.0-aarch64-apple-darwin/bin/rustc")):
        check(name, lambda p=path: {"path": str(p), "present": p.is_file()}, lambda v: v["present"])
    check("disk", lambda: {"free_bytes": shutil.disk_usage(report).free, "required_bytes": 40 * 1024**3},
          lambda v: v["free_bytes"] >= v["required_bytes"])
    check("RUST_MIN_STACK_absent", lambda: "RUST_MIN_STACK" not in os.environ, bool)
    default_sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk"
    selected = os.environ.get("SDKROOT", default_sdk)
    result["selected_SDKROOT"] = {"environment_value": os.environ.get("SDKROOT"), "effective": selected}
    check("selected_SDK26.5", lambda: {"path": selected, "directory": Path(selected).is_dir(),
                                     "name": Path(selected).name},
          lambda v: v["directory"] and v["name"] == "MacOSX26.5.sdk")
    seen = set()
    def sdk_headers(path, origin, required=False):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        row = {"path": key, "origin": origin, "headers": []}
        for item in manifest["SDK_headers"]:
            entry = {"relative_path": item["relative_path"], "expected_sha256": item["sha256"]}
            try:
                entry["sha256"] = sha(path / item["relative_path"])
                entry["matches"] = entry["sha256"] == item["sha256"]
            except Exception as error:
                entry.update(matches=False, error=repr(error))
            row["headers"].append(entry)
        result["SDKs"].append(row)
        if required:
            checks["selected_SDK_headers"] = {"ok": all(x["matches"] for x in row["headers"])}
    sdk_headers(Path(selected), "selected", required=True)
    def xcrun_sdk():
        if time.monotonic() >= deadline:
            raise RuntimeError("STOPNEXT finite admission expired before xcrun")
        command = ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"]
        child = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        return {"argv": command, "exit": child.returncode, "stdout": child.stdout[:4096],
                "stderr": child.stderr[:4096], "output_truncated": len(child.stdout) > 4096 or len(child.stderr) > 4096}
    xcrun = check("xcrun", xcrun_sdk, lambda v: v["exit"] == 0 and not v["output_truncated"]
                  and Path(v["stdout"].strip()).is_absolute())
    if xcrun["ok"]:
        sdk_headers(Path(xcrun["value"]["stdout"].strip()), "xcrun-observation-only")
    # Nonrecursive, bounded discovery: no caches, models, or unrelated application contents.
    def bounded_names(path, accepts, limit):
        names = []
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= 256:
                    return names, True
                if accepts(entry.name) and entry.is_dir():
                    names.append(entry.name)
                    if len(names) >= limit:
                        return names, True
        return sorted(names), False
    apps = check("Xcode_candidates", lambda: bounded_names(Path("/Applications"),
                 lambda name: name.startswith("Xcode") and name.endswith(".app"), 16),
                 lambda v: bool(v[0]) and not v[1])
    if "value" in apps:
        for name in apps["value"][0]:
            base = Path("/Applications") / name / "Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"
            try:
                names, truncated = bounded_names(base, lambda n: n.startswith("MacOSX") and n.endswith(".sdk"), 32)
                result["discovery"].append({"directory": str(base), "SDK_names": names, "truncated": truncated})
                for sdk_name in names:
                    sdk_headers(base / sdk_name, "installed-observation-only")
            except Exception as error:
                result["discovery"].append({"directory": str(base), "error": repr(error)})
    native_text = os.environ.get("PMETAL_MLX_PREBUILT_DIR")
    metallib_text = os.environ.get("PMETAL_METALLIB_PATH")
    result["native_environment"] = {"PMETAL_MLX_PREBUILT_DIR": native_text,
                                    "PMETAL_METALLIB_PATH": metallib_text}
    native = Path(native_text) if native_text else None
    metallib = Path(metallib_text) if metallib_text else None
    check("native_paths", lambda: bool(native and metallib and native.is_absolute() and metallib.is_absolute()
                                      and metallib == native / "mlx.metallib"), bool)
    for item in manifest["native_files"]:
        def native_hash(entry=item):
            if native is None or not native.is_absolute():
                raise ValueError("missing absolute PMETAL_MLX_PREBUILT_DIR")
            return {"path": str(native / entry["name"]), "sha256": sha(native / entry["name"]),
                    "expected_sha256": entry["sha256"]}
        check("native_file:" + item["name"], native_hash, lambda v: v["sha256"] == v["expected_sha256"])
    def fingerprint():
        if native is None or not native.is_absolute():
            raise ValueError("missing absolute PMETAL_MLX_PREBUILT_DIR")
        text = (native / "pmetal-mlx-prebuilt.txt").read_text(encoding="utf-8")
        expected = "fingerprint=096020673cabac7c1cfda2faee80393c5902a41e6d2f004e7e70f27debe4357c"
        return {"expected": expected, "present": expected in text}
    check("native_fingerprint", fingerprint, lambda v: v["present"])
    result["all_named_prerequisites_ok"] = all(row["ok"] for row in checks.values())
    result["discovery_complete"] = apps["ok"] and all("error" not in row and not row["truncated"]
                                                      for row in result["discovery"])
    result["compile_readiness_claim"] = False  # Presence/hash observations are not an execution qualification.
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--phase", choices=("compile-only", "compile-host", "prerequisite-probe"), default="compile-only")
    args = parser.parse_args()
    assert 0 < args.seconds <= 1800, "finite owner admission required"
    repo = Path(__file__).resolve().parents[2]
    manifest_path = (repo / args.manifest).resolve()
    assert manifest_path.is_relative_to(repo), "owner manifest must live in the frozen checkout"
    source_sha = os.environ["SOURCE_SHA"]
    manifest_sha = os.environ["SOURCE_MANIFEST_SHA256"]
    assert re.fullmatch(r"[0-9a-f]{40}", source_sha) and re.fullmatch(r"[0-9a-f]{64}", manifest_sha)
    assert os.environ.get("RUNNER_NAME") == "nax-macos-2", "never run this profile on the dev Mac"
    assert os.environ.get("GITHUB_SHA") == source_sha, "workflow ref is not immutable source binding"
    report = Path(os.environ["RUNNER_TEMP"]) / f"ios-pricing-compile-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    report.mkdir(parents=True, exist_ok=True)
    receipt_path = report / "receipt.json"
    assert not receipt_path.exists(), "one-shot receipt already exists"
    receipt = {"state": "STARTING", "source_sha": source_sha, "manifest_sha256": manifest_sha,
               "phase": args.phase, "steps": [], "binaries": {}, "complete": False,
               "signals_sent": False, "owned_jobs_remaining": 0, "no_automatic_next": True}
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    deadline = started + args.seconds
    receipt["start_UTC"] = datetime.now(timezone.utc).isoformat()
    save()
    try:
        assert platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
        assert sys.version_info[:2] == (3, 12), "preinstalled Python3.12 required"
        assert sha(manifest_path) == manifest_sha
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "CANONICAL_OWNER_SOURCE_BOUND", "not an actual owner-bound source manifest"
        assert manifest["compiler_packages"] == [p for p, _ in PACKAGES]
        assert manifest["artifact_features"] == {name: [] for _, name in PACKAGES}, "no silent capture/training feature"
        assert len(manifest["HOSTs"]) == manifest["logical_HOST_count"]
        assert len({x["fqn"] for x in manifest["HOSTs"]}) == len(manifest["HOSTs"])
        assert sum(x["expected_Rust_rows"] for x in manifest["HOSTs"]) == manifest["expected_HOST_Rust_rows"]
        (report / "source-manifest.json").write_bytes(manifest_path.read_bytes())
        def inputs(source_only=False):
            assert sha(manifest_path) == manifest_sha
            assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, encoding="utf-8").strip() == source_sha
            assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True, encoding="utf-8").strip(), "dirty/untracked remote checkout"
            for item in manifest["files"]:
                relative = Path(item["path"])
                assert not relative.is_absolute() and ".." not in relative.parts
                assert sha(repo / relative) == item["sha256"], str(relative)
            if not source_only:
                assert shutil.disk_usage(report).free >= 40 * 1024**3, "disk below40GiB"
                assert "RUST_MIN_STACK" not in os.environ, "no stack workaround"
        if args.phase == "prerequisite-probe":
            inputs(source_only=True)
            assert time.monotonic() < deadline, "STOPNEXT finite admission expired before probe"
            probe = prerequisite_probe(manifest, report, deadline)
            probe_path = report / "prerequisite-probe.json"
            probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
            receipt["prerequisite_probe"] = {"path": str(probe_path), "sha256": sha(probe_path),
                                             "all_named_prerequisites_ok": probe["all_named_prerequisites_ok"],
                                             "discovery_complete": probe["discovery_complete"]}
            inputs(source_only=True)
            receipt["probe_observations_complete"] = True
            receipt["state"] = "PROBE_RECORDED_NOT_COMPILE_QUALIFIED"
            save()
            return 0 if probe["all_named_prerequisites_ok"] and probe["discovery_complete"] else 1
        inputs()
        home = Path.home()
        cargo = home / ".cargo/bin/cargo"
        rustc = home / ".rustup/toolchains/1.96.0-aarch64-apple-darwin/bin/rustc"
        assert cargo.is_file() and rustc.is_file(), "preinstalled Rust1.96 required; no downloads"
        sdk = Path(os.environ.get("SDKROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk"))
        assert sdk.is_dir() and sdk.name == "MacOSX26.5.sdk", "preinstalled SDK26.5 required"
        for item in manifest["SDK_headers"]:
            assert sha(sdk / item["relative_path"]) == item["sha256"], "SDK source binding mismatch"
        native = Path(os.environ["PMETAL_MLX_PREBUILT_DIR"])
        metallib = Path(os.environ["PMETAL_METALLIB_PATH"])
        assert native.is_absolute() and metallib.is_absolute() and metallib == native / "mlx.metallib"
        for item in manifest["native_files"]:
            assert sha(native / item["name"]) == item["sha256"], "preprovisioned native096 artifact mismatch"
        assert "fingerprint=096020673cabac7c1cfda2faee80393c5902a41e6d2f004e7e70f27debe4357c" in (native / "pmetal-mlx-prebuilt.txt").read_text(encoding="utf-8")
        target = report / "cargo-target"
        temporary = report / "tmp"
        assert not target.exists() and not temporary.exists(), "unique outside-checkout target/TMPDIR required"
        target.mkdir(); temporary.mkdir()
        env = os.environ.copy()
        env.update(PATH=str(home / ".cargo/bin") + os.pathsep + env.get("PATH", ""),
                   CARGO_TARGET_DIR=str(target), TMPDIR=str(temporary) + os.sep,
                   CARGO_BUILD_JOBS="4", CARGO_INCREMENTAL="0", RUSTC_WRAPPER="", CARGO_NET_OFFLINE="true",
                   CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", RUSTFLAGS="-C debuginfo=0",
                   CARGO_PROFILE_DEV_BUILD_OVERRIDE_DEBUG="0", CARGO_PROFILE_TEST_BUILD_OVERRIDE_DEBUG="0",
                   SDKROOT=str(sdk), MACOSX_DEPLOYMENT_TARGET="26.2", RUST_TEST_THREADS="1")
        clear = manifest["clear_env_before_all"]
        for key in clear:
            env.pop(key, None)
        def dispatch_guard():
            assert time.monotonic() < deadline, "STOPNEXT finite admission expired"
            inputs()
            assert not external_jobs(process_rows()), "other compiler/native lane is busy"
            assert time.monotonic() < deadline, "STOPNEXT after preflight"
        def command(label, argv, kind, step=None):
            dispatch_guard()
            log = report / (label + ".log")
            assert not log.exists()
            row = {"label": label, "kind": kind, "argv": argv, "log": str(log), "samples": 0,
                   "sample_errors": 0, "rss_max_bytes_sampled_NOT_peak": 0, "resource_breach": False}
            receipt["steps"].append(row); receipt["owned_jobs_remaining"] = 1; save()
            print("START " + label, flush=True)
            began = time.monotonic()
            with log.open("wb") as output:
                child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=repo)
                row["pid"] = child.pid; save()
                while child.poll() is None:
                    try:
                        observed = process_rows(); own = descendants(observed, child.pid)
                        rss = sum(observed[pid][1] for pid in own if pid in observed)
                        row["samples"] += 1; row["rss_max_bytes_sampled_NOT_peak"] = max(row["rss_max_bytes_sampled_NOT_peak"], rss)
                        row["resource_breach"] |= rss > 4 * 1024**3 or bool(external_jobs(observed))
                    except Exception as error:
                        row["sample_errors"] += 1; row["sample_error"] = repr(error)
                    time.sleep(0.25)
                row["exit"] = child.wait()  # Natural completion only: no deadline/signal inside command.
            row["seconds"] = time.monotonic() - began; row["log_sha256"] = sha(log)
            text = log.read_text(encoding="utf-8", errors="replace")
            messages, warnings, errors, plain, rust = parse_log(text)
            row.update(warnings=warnings, errors=errors, plain_diagnostics=plain, Rust_rows=rust)
            receipt["owned_jobs_remaining"] = 0; save()  # Raw results BEFORE qualifier assertions.
            assert row["exit"] == 0 and not warnings and not errors and not plain, "unexpected exit/diagnostic"
            assert row["samples"] > 0 and not row["sample_errors"] and not row["resource_breach"], "resource STOPNEXT"
            if kind == "compile":
                package, unit = step
                found = [m for m in messages if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == unit and m.get("profile", {}).get("test") and m.get("executable")]
                assert len({m["executable"] for m in found}) == 1 and all(m.get("features", []) == [] for m in found)
                if unit == "mlx_gen_ios_catalog":
                    names = {m.get("target", {}).get("name") for m in messages if m.get("reason") == "compiler-artifact"}
                    assert set(manifest["catalog_dependency_targets_expected"]) <= names, "catalog dependency artifact coverage incomplete"
                for artifact in messages:
                    if artifact.get("reason") == "compiler-artifact":
                        name = artifact.get("target", {}).get("name"); features = artifact.get("features", [])
                        if name in manifest["provider_training_absent"]:
                            assert "training" not in features
                        if name == "mlx_gen_ios_catalog":
                            assert "device-capture" not in features
                binary = Path(found[0]["executable"])
                assert binary.is_relative_to(target) and binary.is_file()
                receipt["binaries"][unit] = {"path": str(binary), "sha256": sha(binary), "features": [], "actual_compiler_JSON": True}
                assert not rust
            elif kind == "list":
                assert re.findall(r"(?m)^(.+): test$", text) == [step["fqn"]] and not rust
            else:
                validate_host(step, text, rust)
            row["qualified"] = True; inputs(); save(); print("FINISH " + label, flush=True)
        for package, unit in PACKAGES:
            argv = [str(cargo), "+1.96.0", "test", "--manifest-path", str(repo / "Cargo.toml"),
                    "--config", str(repo / ".cargo/config.toml"), "--locked", "--offline", "-p", package,
                    "--no-default-features", "--target", "aarch64-apple-darwin", "--lib", "--no-run", "--message-format=json"]
            command("compile-" + unit, argv, "compile", (package, unit))
        for index, host in enumerate(manifest["HOSTs"], 1):
            binding = receipt["binaries"][host["binary_binding"]]; assert sha(binding["path"]) == binding["sha256"]
            command(f"list-{index:02}", [binding["path"], "--exact", host["fqn"], "--list", "--test-threads=1"], "list", host)
        if args.phase == "compile-host":
            assert sum(s["kind"] == "list" and s.get("qualified", False) for s in receipt["steps"]) == len(manifest["HOSTs"])
            for index, host in enumerate(manifest["HOSTs"], 1):
                binding = receipt["binaries"][host["binary_binding"]]; assert sha(binding["path"]) == binding["sha256"]
                command(f"HOST-{index:02}", [binding["path"], "--exact", host["fqn"], "--nocapture", "--test-threads=1"], "HOST", host)
        inputs()
        receipt["final_source_hashes_verified"] = len(manifest["files"])
        for binding in receipt["binaries"].values():
            assert sha(binding["path"]) == binding["sha256"], "compiled binary changed after binding"
        receipt["complete"] = True; receipt["state"] = "GREEN_SCOPED_" + args.phase
    except Exception as error:
        receipt["state"] = "STOPPED_BEFORE_NEXT_NO_RETRY"; receipt["failure"] = repr(error)
        print("STOP " + repr(error), flush=True)
    finally:
        try:
            observed = process_rows(); pids = [r["pid"] for r in receipt["steps"] if "pid" in r]
            live = [pid for pid in pids if pid in observed]
            binaries = {b["path"] for b in receipt["binaries"].values()}
            variants = [pid for pid, (_, _, cmd) in observed.items() if any(cmd == b or cmd.startswith(b + " ") for b in binaries)]
        except Exception as audit_error:
            live = ["UNVERIFIED"]; variants = ["UNVERIFIED"]
            receipt["release_audit_error"] = repr(audit_error)
        receipt.update(final_UTC=datetime.now(timezone.utc).isoformat(), own_direct_PIDs_remaining=live,
                       own_binary_variants_remaining=variants, owned_jobs_remaining=len(live),
                       logical_HOSTs=len(manifest["HOSTs"]) if "manifest" in locals() else None,
                       expected_HOST_Rust_rows=manifest["expected_HOST_Rust_rows"] if "manifest" in locals() else None,
                       catalog_capture_validation="SEPARATE_REQUIRED_LATER_NOT_ENABLED")
        if live or variants:
            receipt["complete"] = False; receipt["state"] = "RELEASE_AUDIT_FAILED"
        save(); print("RECEIPT " + str(receipt_path) + " SHA " + sha(receipt_path), flush=True)
    return 0 if receipt["complete"] else 1

if __name__ == "__main__":
    sys.exit(main())
