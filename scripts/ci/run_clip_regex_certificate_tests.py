#!/usr/bin/env python3
"""Run the actual dependency-owned CLIP certificate tests and build-hasher vector on hosted CI.

The root workspace test selection does not run a dependency's unit tests. Cargo rustc builds
regex's OWN lib source as a test harness with its normal locked dependencies and actual build
provenance. No upstream dev-dependency resolution, copied assertions, synthetic source markers,
model/runtime work, or measurement gate. The original six rows, mandatory positive row and
hash-vector row must actually run.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_TESTS = (
    "literal_and_dependency_identity_are_exact",
    "overflow_is_not_a_saturating_certificate",
    "init_predicate_is_strict_and_checked_before_any_facts",
    "real_type_queries_produce_two_distinct_finite_terms_when_supported",
    "search_keeps_hybrid_and_both_fallback_stores",
    "arc_payload_is_not_the_arc_pointer_layout",
    "admitted_target_public_api_returns_checked_finite_envelopes",
)


def require_one_passed(output: str, name: str) -> None:
    rows = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", output)
    if rows != [("ok", "1", "0", "0")]:
        raise RuntimeError(f"{name}: expected one actual passed test, got {rows}")


def require_positive_facts(output: str) -> dict[str, int]:
    rows = re.findall(r"^CLIP_SOURCE_FACTS C=([0-9]{1,20}) S=([0-9]{1,20}) J=([0-9]{1,20}) J_limit=([0-9]{1,20}) M0=([0-9]{1,20}) M0_native=([0-9]{1,20}) status=checked$", output, re.MULTILINE)
    if len(rows) != 1:
        raise RuntimeError("expected one actual checked positive source-fact record")
    cold, search, j, limit, m0, native_m0 = map(int, rows[0])
    if not (cold > 0 and search > 0 and cold != search and limit == 2_097_152 and j < limit and m0 > 0 and m0 == native_m0):
        raise RuntimeError("positive source-fact record violates checked envelope/J/M0 predicates")
    return {"C": cold, "S": search, "J": j, "J_limit": limit, "M0": m0, "M0_native": native_m0}


def run(argv: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(argv, cwd=ROOT, env=env, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {argv}")
    return result.stdout


def exact(binary: Path, fqn: str, env: dict[str, str]) -> str:
    output = run([str(binary), "--exact", fqn, "--nocapture", "--test-threads=1"], env)
    require_one_passed(output, fqn)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(args.output_dir / "target")
    env["CARGO_PROFILE_DEV_DEBUG"] = "0"
    env["CARGO_PROFILE_TEST_DEBUG"] = "0"
    binary = args.output_dir / "regex-certificate-tests"
    # --test compiles the dependency's own #[cfg(test)] modules. Source audit: the only existing
    # lib-test module is regex/bytes.rs (alloc/std only); doc-comment is #[cfg(doctest)], inactive.
    # Cargo's artifact handling for this command must be demonstrated by this hosted step: an
    # absent executable or a build error is a real failure, never skipped or replaced by a DTO.
    # Excluded dependencies cannot take Cargo --features flags. The SDXL dependency table
    # opts in; require that actual inherited feature in the locked workspace graph first.
    features = run(["cargo", "tree", "--locked", "-p", "regex", "--depth", "0", "--format", "{f}"], env)
    if "clip-source-certificate" not in {part.strip() for line in features.splitlines() for part in line.split(",")}:
        raise RuntimeError("the actual locked regex dependency has no clip-source-certificate feature")
    # Cargo still regards this as a library build and may pass metadata-only externs to the
    # injected test harness. Build the real rlibs first, then substitute only their exact
    # same-hash artifact paths for the primary regex --test invocation. No source/feature/
    # compiler-provenance environment is fabricated and missing rlibs are a hard failure.
    env["RUSTC_WRAPPER"] = str(ROOT / "scripts/ci/clip_regex_test_rustc.py")
    run(["cargo", "build", "--locked", "-p", "regex", "--lib"], env)
    run(["cargo", "rustc", "--locked", "-p", "regex", "--lib", "--", "--test", "-o", str(binary)], env)
    if not binary.is_file():
        raise RuntimeError("cargo rustc did not produce the dependency-owned test executable")
    listed = run([str(binary), "--list"], env)
    for name in GENERATOR_TESTS:
        fqn = "clip_source_certificate::tests::" + name
        if fqn + ": test" not in listed.splitlines():
            raise RuntimeError(f"missing dependency-owned test {fqn}")
        output = exact(binary, fqn, env)
        if name == "admitted_target_public_api_returns_checked_finite_envelopes":
            require_positive_facts(output)
    hasher = args.output_dir / "source-hasher-tests"
    run(["rustc", "--edition", "2021", "--test", "vendor/clip-regex/build-verify.rs", "-o", str(hasher)], env)
    exact(hasher, "tests::source_hasher_known_vectors", env)
    print("CLIP source certificate: original six + mandatory positive dependency test and build-hasher vector passed")


if __name__ == "__main__":
    main()
