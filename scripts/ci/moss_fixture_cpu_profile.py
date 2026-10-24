#!/usr/bin/env python3
"""Hosted-only S4 CPU evidence and actual assertion-killing review mutations.

No device inventory or model downloads. Uses the existing owned CPU supervisor unchanged.
Compilation has the hosted job/45-minute compiler bound; each test keeps 180s/2GiB.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import moss_fixture_cpu as fixture
from scripts.ci.moss_cuda_unit import UnitFailure, supervise

PHASE_TEST = "training_memory::tests::every_phase_rejects_one_byte_below_and_accepts_boundary"
CADENCE_TEST = "training::review_admission_tests::resume_accepts_checkpoint_cadence_inside_accumulation_window"
METADATA_TEST = "training_memory::tests::persistent_metadata_raises_backward_checkpoint_and_admission_boundary"
ZERO_TEST = "training::review_admission_tests::concrete_trainer_override_refuses_before_missing_model_or_dataset_can_load"
STAGING_TEST = "test_support::tests::source_sized_request_staging_preserves_public_default_and_refuses_cap"
# Existing immutable CPU controls, selected from the actual d079 unfiltered listings.
CAUSAL_TESTS = {
    "candle": ("tests::training_contract_tests::cpu_production_graph_four_cases_checkpoint_gradients_updates_and_strict_reload",),
    "mlx": (
        "transformer::training_graph_tests::cpu_native_backward_updates_only_factors_and_reload_changes_generation",
        "transformer::training_bf16_cpu_tests::production_bf16_backbone_keeps_fp32_adapter_gradients_and_portable_reload",
        "transformer::review_fix_tests::all_projection_lokr_keeps_raw_factors_in_dense_and_checkpointed_backward",
    ),
}
DIAGNOSTIC_BASELINES = ("training_memory::tests::audit_workspace_is_charged_in_checkpoint_and_host_domains",)
FRACTION_TESTS = tuple("train::memory_budget_tests::" + name for name in (
    "fraction_uses_selected_total_not_free_or_safety_and_intersects_bytes",
    "fraction_invalid_values_and_sub_byte_caps_refuse_without_fallback",
    "fraction_cap_rounds_down_exactly_even_at_u64_max"))
MASK_TESTS = tuple("train::subject_mask::tests::" + name for name in (
    "area_average_preserves_nonbinary_constants_and_convex_range",
    "area_average_partial_crop_matches_independent_integer_overlap_oracle",
    "area_average_f64_taps_keep_supported_host_scratch_layout"))
SNAPSHOT_TESTS = (
    "moss_training_probe::snapshot::tests::streamed_factor_deltas_match_independent_known_values_and_saved_scalars",
    "moss_training_probe::snapshot::tests::streamed_reader_rejects_other_parent_layout_alpha_and_unadmitted_buffers",
    "moss_training_probe::snapshot::tests::optimizer_metadata_null_absent_and_map_preserve_strict_scalar_validation",
)
TRAINING = "crates/media/mlx-gen/mlx-gen-moss-image/src/training.rs"
MLX_MEMORY = "crates/media/mlx-gen/mlx-gen-moss-image/src/training_memory.rs"


def push_route_proof(root: Path = ROOT) -> dict:
    """Fail closed if checked-in triggers invalidate the literal branch-only push assumption."""
    proof = {}
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        event = re.search(r"(?ms)^on:\n(.*?)(?=^[^ \n#]|\Z)", source)
        if event is None:
            raise UnitFailure(f"cannot establish workflow trigger structure: {path.name}")
        trigger = event.group(1)
        if re.search(r"^  (workflow_run|repository_dispatch):", trigger, re.M):
            raise UnitFailure("new indirect trigger invalidates branch-only publication proof")
        pushes = re.findall(r"(?m)^  push:(.*(?:\n(?:    .*|\s*#.*))*)", trigger)
        if path.name == "ci.yml":
            expected = "\n    branches: [main]\n    tags: [\"runtime-*\"]"
        elif path.name == "kv-poc-campaign.yml":
            expected = '\n    branches: ["kv-poc-run/**"]'
        else:
            expected = None
        if (expected is None and pushes) or (expected is not None and pushes != [expected]):
            raise UnitFailure(f"push triggers changed; re-admit publication before proceeding: {path.name}")
        proof[path.name] = {"trigger_sha256": hashlib.sha256(trigger.encode()).hexdigest(),
                            "push_trigger": pushes or None}
    if "ci.yml" not in proof or "kv-poc-campaign.yml" not in proof:
        raise UnitFailure("expected push workflow disappeared")
    return {"literal_branch": "story/sc-26588-moss-recovery", "matches_push_trigger": False,
            "scope": "checked-in repository workflows, not external integrations", "workflows": proof}


def venue(backend: str) -> dict:
    head = fixture.git_head()
    if (os.environ.get("GITHUB_REPOSITORY") != "SceneWorks/inference"
            or os.environ.get("GITHUB_SHA") != head or not re.fullmatch(r"[0-9a-f]{40}", head)
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise UnitFailure("profile requires exact-repository/current-head manual GitHub-hosted execution")
    expected_os = "macOS" if backend == "mlx" else "Linux"
    if os.environ.get("RUNNER_OS") != expected_os:
        raise UnitFailure("backend does not match its fixed hosted OS row")
    pinned = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]["channel"]
    rustc = subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True, encoding="utf-8").strip()
    if pinned != "1.96.0" or not rustc.startswith(f"rustc {pinned} "):
        raise UnitFailure("actual compiler does not match the approved native toolchain")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT):
        raise UnitFailure("profile must start from unchanged checked-out source")
    return {"repository": os.environ["GITHUB_REPOSITORY"], "head": head,
            "ref": os.environ.get("GITHUB_REF"), "backend": backend, "runner_os": expected_os,
            "toolchain": rustc, "cache_hit": os.environ.get("MOSS_CPU_CACHE_HIT", "unknown"),
            "target": str(Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve()),
            "case_deadline_seconds": fixture.DEADLINE, "case_rss_cap_bytes": fixture.RSS_CAP,
            "push_route_proof": push_route_proof()}


def mutations(backend: str) -> list[dict]:
    memory = MLX_MEMORY if backend == "mlx" else "crates/media/candle-gen/candle-gen-moss-image/src/training_memory.rs"
    result = [{"id": "phase-isolation", "path": memory, "old": "            only.checkpoint = 0;\n",
               "new": "", "test": PHASE_TEST, "panic": "isolated phase retains another demand"},
              {"id": "ignored-export-factor-update",
               "path": "crates/contracts/gen-core-testkit/src/moss_training_probe/snapshot.rs",
               "old": "                let difference = f64::from(b) - f64::from(a);",
               "new": "                let difference = 0.0_f64;",
               "test": fixture.CASES[0], "panic": "native portable factors did not numerically update"}]
    if backend == "mlx":
        result.append({"id": "partial-window-cadence", "path": TRAINING,
                       "old": "    if c.sample_every > 0 && !c.sample_prompts.is_empty() && c.sample_steps == 0 {",
                       "new": "    if c.resume && c.save_every > 0 && !c.save_every.is_multiple_of(c.gradient_accumulation) {\n        return Err(Error::Unsupported(\"obsolete complete-window restriction\".into()).into());\n    }\n    if c.sample_every > 0 && !c.sample_prompts.is_empty() && c.sample_steps == 0 {",
                       "test": CADENCE_TEST, "panic": "complete recovery stores pending gradients"})
        for phase in ("backward", "checkpoint"):
            old = f'''        {phase}: sum(
            [
                {phase},
                r.audit_workspace_bytes,
                f.host_metadata_bytes,
                r.dataset_metadata_bytes,
            ],
            "{phase} audit staging",
        )?,'''
            for term in ("f.host_metadata_bytes", "r.dataset_metadata_bytes"):
                result.append({"id": f"{phase}-omit-{term.split('.')[1]}", "path": MLX_MEMORY,
                               "old": old, "new": old.replace(f"                {term},\n", ""),
                               "test": METADATA_TEST, "panic": f"{phase.capitalize()}, host={str(term.startswith('f.')).lower()}"})
        result.append({"id": "source-sized-staging", "path": "crates/media/mlx-gen/mlx-gen-moss-image/src/memory_strategy.rs",
                       "old": ".and_then(|n| n.checked_add(source_staging))",
                       "new": ".and_then(|n| n.checked_add(5_362_224_248))", "test": STAGING_TEST,
                       "panic": "private fixture source staging ignored"})
        result.append({"id": "positive-budget-error", "path": "crates/contracts/gen-core/src/train.rs",
                       "old": '"training memory budget must be positive"',
                       "new": '"mutated positive-budget refusal"', "test": ZERO_TEST,
                       "panic": "zero override must produce the typed positive-budget refusal"})
    return result


def verify_red(output: str, name: str, panic: str, returncode: int) -> None:
    names = re.findall(r"(?m)^test (\S+) \.\.\.", output)
    summaries = [line for line in output.splitlines() if line.startswith("test result:")]
    if (returncode != 101 or output.splitlines().count("running 1 test") != 1 or names != [name]
            or not re.search(rf"(?ms)^test {re.escape(name)} \.\.\.[ \t]*(?:FAILED$|\n.*?^FAILED$)", output)
            or len(summaries) != 1
            or not re.fullmatch(r"test result: FAILED\. 0 passed; 1 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*", summaries[0])
            or panic not in output or "panicked at" not in output):
        raise UnitFailure(f"mutation did not produce its exact runtime assertion failure: {name}")


def compile_binary(backend: str, path: Path) -> tuple[Path, dict]:
    with path.with_suffix(".compile.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/ci/moss_fixture_cpu.py"),
                                 "compile", backend, "--binary-file", str(path)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, timeout=2760)
    if result.returncode:
        # Preserve the complete compiler artifact log, and surface a bounded diagnostic tail.
        with path.with_suffix(".compile.log").open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 64 * 1024))
            print(log.read().decode("utf-8", "replace"), file=sys.stderr, flush=True)
        stage = "baseline" if path.stem == "baseline" else f"{path.parent.name}/{path.stem}"
        raise UnitFailure(f"{stage} compile failed; compile errors are NOT assertion RED evidence")
    record = json.loads(path.read_text(encoding="utf-8"))
    binary = Path(record["binary"])
    if (record["source"] != fixture.source_stamp() or record["binary_sha256"] != fixture.file_digest(binary)
            or record["cargo_json_sha256"] != fixture.file_digest(path.with_suffix(".cargo.jsonl"))):
        raise UnitFailure("mutation compiler artifact identity differs")
    return binary, record


def compiled_core_controls(directory: Path, package: str, lib_name: str, names: list[str], prefix: str, groups: dict) -> dict:
    argv = ["cargo", "test", "--locked", "--lib", "-j", "1", "-p", package,
            "--no-run", "--message-format=json"]
    stamp = fixture.source_stamp()
    with (directory / f"{prefix}-compile.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=log,
                                text=True, encoding="utf-8", timeout=2700)
    cargo_log = directory / f"{prefix}.cargo.jsonl"
    cargo_log.write_text(result.stdout, encoding="utf-8")
    if fixture.source_stamp() != stamp:
        raise UnitFailure("contract source changed during compilation")
    if result.returncode:
        fixture.render_compile_errors(result.stdout)
        if result.returncode != 101:
            raise UnitFailure("fraction compiler did not finish normally; independent readiness unknown")
        return {"passed": False, "compiler_exit": result.returncode, "source": stamp,
                "groups": groups,
                "cargo_json_sha256": fixture.file_digest(cargo_log),
                "tests": [{"test": name, "status": "blocked_compile"} for name in names]}
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    binary = fixture.select_binary(result.stdout, package, target, lib_name=lib_name)
    listing = supervise([str(binary), "--list"], fixture.CpuOwner(), directory / f"{prefix}-unfiltered-list.log",
                        deadline=fixture.DEADLINE, rss_cap=fixture.RSS_CAP)
    if not all(name + ": test" in listing.splitlines() for name in names):
        raise UnitFailure("contract binary lacks exact fraction tests")
    results = independent_tests(binary, names, directory, prefix)
    return {"tests": results, "passed": all(r["status"] == "passed" for r in results),
            "groups": groups,
            "source": fixture.source_stamp(), "binary_sha256": fixture.file_digest(binary),
            "cargo_json_sha256": fixture.file_digest(cargo_log)}


def contract_fraction_tests(directory: Path) -> dict:
    return compiled_core_controls(directory, "sceneworks-gen-core", "gen_core", [*FRACTION_TESTS, *MASK_TESTS],
                                  "contract", {"fraction": list(FRACTION_TESTS), "mask": list(MASK_TESTS)})


def snapshot_reader_tests(directory: Path) -> dict:
    return compiled_core_controls(directory, "sceneworks-gen-core-testkit", "gen_core_testkit", list(SNAPSHOT_TESTS),
                                  "snapshot", {"streamed_reader_only": list(SNAPSHOT_TESTS)})


def assertion_red(binary: Path, mutation: dict, log: Path) -> dict:
    environment = dict(os.environ, RUST_TEST_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    owner = fixture.CpuOwner()
    started = time.monotonic()
    try:
        supervise([str(binary), mutation["test"], "--exact", "--nocapture", "--color", "never", "--test-threads=1"],
                  owner, log, deadline=fixture.DEADLINE, rss_cap=fixture.RSS_CAP, env=environment)
    except UnitFailure as error:
        # Deadline, OOM, signal, zero/ignored tests, compile failure and unrelated panic never count.
        if str(error) != "test process exited with 101" or owner.child.returncode != 101:
            raise
    else:
        raise UnitFailure("mutation survived: exact test still passed")
    verify_red(log.read_text(encoding="utf-8"), mutation["test"], mutation["panic"], owner.child.returncode)
    if owner.collect():
        raise UnitFailure("mutation left an owned child alive")
    return {"exit_code": owner.child.returncode, "owned_released": True,
            "observed_peak_child_rss_bytes": owner.peak, "observed_elapsed_seconds": time.monotonic() - started,
            "configured_rss_cap_bytes": fixture.RSS_CAP, "configured_deadline_seconds": fixture.DEADLINE}


def independent_tests(binary: Path, names: list[str], directory: Path, prefix: str) -> list[dict]:
    results = []
    for index, name in enumerate(names):
        log = directory / f"{prefix}-{index}.log"
        try:
            fixture.execute(binary, name, log)
        except fixture.CaseAssertionFailure:
            pass  # Only exact independently released assertion failures are aggregatable.
        results.append(json.loads(log.with_suffix(".result.json").read_text(encoding="utf-8")))
    return results


def validate_batch(batch: dict, returncode: int, source: dict) -> None:
    rows = batch.get("cases")
    if (returncode not in (0, 1) or batch.get("fatal_error") is not None
            or batch.get("source") != source or batch.get("final_source") != source
            or batch.get("source_restored") is not True or batch.get("header_admitted") is not True
            or not isinstance(rows, list) or [r.get("test") for r in rows] != [fixture.TOKEN_TEST, *fixture.CASES]
            or any(r.get("status") not in ("passed", "assertion_failed")
                   or r.get("owned_released") is not True or r.get("source") != source for r in rows)):
        raise UnitFailure("fixture admission/source/cleanup/protocol incomplete; refusing independent phases")
    passed = all(r["status"] == "passed" for r in rows)
    if batch.get("passed") is not passed or returncode != (0 if passed else 1):
        raise UnitFailure("fixture batch exit/status differs")
    fixture.validate_demand("MOSS_FIXTURE_DEMAND=" + json.dumps(batch.get("actual_demand")))


def run(backend: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {"backend": backend, "head": None, "passed": False,
                "expected_fixture_tests": [fixture.DEMAND_TEST, fixture.TOKEN_TEST, *fixture.CASES],
                "expected_review_tests": sorted({m["test"] for m in mutations(backend)} | set(CAUSAL_TESTS[backend]) | set(DIAGNOSTIC_BASELINES)),
                "causal_controls": list(CAUSAL_TESTS[backend]), "diagnostic_admission_controls":list(DIAGNOSTIC_BASELINES), "mutations": []}
    baseline = None
    try:
        evidence.update(venue(backend))
        baseline = fixture.source_stamp()
        # The same reviewed compile/run commands used by ordinary package CI, not a copied trainer.
        binary, evidence["baseline_compiler"] = compile_binary(backend, directory / "baseline.json")
        with (directory / "fixture-run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/ci/moss_fixture_cpu.py"), "run", backend,
                                     "--binary-file", str(directory / "baseline.json")], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1500)
        batch = json.loads((directory / "baseline.batch.json").read_text(encoding="utf-8"))
        evidence["fixture_batch"] = batch
        validate_batch(batch, result.returncode, baseline)
        evidence["actual_demand"] = batch["actual_demand"]
        evidence["fraction_contract"] = contract_fraction_tests(directory)
        evidence["snapshot_reader_controls"] = snapshot_reader_tests(directory)
        evidence["review_baselines"] = independent_tests(binary, evidence["expected_review_tests"], directory, "review-baseline")
        green_baselines = {r["test"] for r in evidence["review_baselines"] if r["status"] == "passed"}
        for mutation in mutations(backend):
            if mutation["test"] not in green_baselines:
                evidence["mutations"].append({"id": mutation["id"], "test": mutation["test"],
                                               "status": "blocked_baseline_red", "runtime_assertion_red": False,
                                               "restored_green": False})
                continue
            if fixture.source_stamp() != baseline:
                raise UnitFailure("source changed before independent mutation")
            folder = directory / mutation["id"]
            folder.mkdir()
            path = ROOT / mutation["path"]
            original = path.read_bytes()
            source = original.decode()
            if source.count(mutation["old"]) != 1:
                raise UnitFailure(f"mutation source anchor changed: {mutation['id']}")
            try:
                path.write_text(source.replace(mutation["old"], mutation["new"], 1), encoding="utf-8")
                red_binary, red_record = compile_binary(backend, folder / "red.json")
                red_resources = assertion_red(red_binary, mutation, folder / "red.log")
            finally:
                path.write_bytes(original)
                os.utime(path, None)
            if fixture.source_stamp() != baseline:
                raise UnitFailure("mutation restore changed original source")
            green_binary, green_record = compile_binary(backend, folder / "green.json")
            fixture.execute(green_binary, mutation["test"], folder / "green.log")
            evidence["mutations"].append({"id": mutation["id"], "test": mutation["test"],
                                           "status": "killed_and_restored", "runtime_assertion_red": True, "restored_green": True,
                                           "red_compiler": red_record, "green_compiler": green_record,
                                           "red_resources": red_resources})
        evidence["passed"] = (batch["passed"] and evidence["fraction_contract"]["passed"]
                              and evidence["snapshot_reader_controls"]["passed"]
                              and all(r["status"] == "passed" for r in evidence["review_baselines"])
                              and all(m["status"] == "killed_and_restored" for m in evidence["mutations"]))
    except BaseException as error:
        evidence["fatal_error"] = str(error)
        raise
    finally:
        try:
            evidence["final_source"] = fixture.source_stamp()
            evidence["source_restored"] = evidence["final_source"] == baseline if baseline is not None else None
        except (OSError, subprocess.SubprocessError) as error:
            evidence["final_source"] = None
            evidence["source_restored"] = None
            evidence["source_check_error"] = str(error)
        if evidence["source_restored"] is not True:
            evidence["passed"] = False
        evidence["file_sha256"] = {str(p.relative_to(directory)): fixture.file_digest(p)
                                    for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "proof.json"}
        (directory / "proof.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if not evidence["source_restored"]:
        raise UnitFailure("profile source was not restored")
    print(json.dumps({"head": evidence["head"], "backend": backend, "passed": evidence["passed"],
                      "mutations": len(evidence["mutations"]), "proof_sha256": fixture.file_digest(directory / "proof.json")}))
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=tuple(fixture.PACKAGES))
    args = parser.parse_args()
    try:
        evidence = run(args.backend, Path(os.environ["RUNNER_TEMP"]) / f"moss-cpu-{args.backend}-evidence")
        if not evidence["passed"]:
            print("moss-cpu-profile: FAIL independent baseline/case/mutation failures remain", file=sys.stderr)
            return 1
    except (UnitFailure, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        print(f"moss-cpu-profile: FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
