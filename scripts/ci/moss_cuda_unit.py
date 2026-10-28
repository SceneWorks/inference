#!/usr/bin/env python3
"""Compile and run exactly one MOSS CUDA unit test under an owned, bounded process tree.

`compile` builds the fixed package's lib test binary and records the one matching executable.
`run` starts that binary with the fixed libtest filter under a wall-clock and RSS ceiling, then
requires a positive `running 1 test` / `1 passed` result. Nothing here loads model weights or
measures VRAM: the caps are safety limits for a unit process, not calibration. A missing CUDA
device fails the test itself; this module never reports a pass it did not observe.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Callable

PACKAGE = "candle-gen-moss-image"
TEST_NAME = "tests::training_contract_tests::cuda_production_graph_four_cases_checkpoint_gradients_updates_and_strict_reload"
COMPILE_ARGV = [
    "cargo", "test", "--locked", "--lib", "-p", PACKAGE, "--features", "cuda",
    "--no-run", "--message-format=json",
]
RUN_ARGS = [TEST_NAME, "--exact", "--ignored", "--nocapture", "--color", "never", "--test-threads=1"]
DEADLINE_SECONDS = 300.0
RSS_CAP_BYTES = 6 * 1024**3
POLL_SECONDS = 0.25
GRACE_SECONDS = 10.0
OUTPUT_TAIL_BYTES = 1 << 20
REPO_ROOT = Path(__file__).resolve().parents[2]
EXE_FILE = "MOSS_CUDA_UNIT_EXE"
# Real Job Object smoke (fake Python children, no CUDA/Cargo/weights) that must run before the GPU test.
SMOKE_TESTS = (
    "scripts.tests.test_media_campaign_supervisor.WindowsSupervisorTests",
    "scripts.tests.test_moss_cuda_unit.WindowsOwnerTests",
)
# Variables that would redirect the test body into its export/import modes.
SCRUBBED_ENV = ("MOSS_PORTABLE_EXCHANGE_OUTPUT", "MOSS_PORTABLE_EXCHANGE_INPUT")


class UnitFailure(RuntimeError):
    pass


def select_artifact(stdout: str, target_dir: Path) -> Path:
    """The single lib test executable of PACKAGE inside target_dir; anything else fails closed."""
    found: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise UnitFailure(f"non-JSON cargo message: {line[:120]!r}") from error
        if not isinstance(message, dict) or message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target") or {}
        package_id = str(message.get("package_id", ""))
        named = re.search(rf"(?:/|#|^){re.escape(PACKAGE)}(?:[@#\s]|$)", package_id)
        if (
            named
            and target.get("name") == PACKAGE.replace("-", "_")
            and target.get("kind") == ["lib"]
            and (message.get("profile") or {}).get("test") is True
            and message.get("executable")
        ):
            found.append(message["executable"])
    if len(found) != 1:
        raise UnitFailure(f"expected exactly one {PACKAGE} lib test executable, found {len(found)}")
    exe = Path(found[0])
    if not exe.is_absolute() or not exe.is_file() or target_dir.resolve() not in exe.resolve().parents:
        raise UnitFailure(f"test executable {exe} is not a file inside {target_dir}")
    return exe


def canonical_gpu_uuid(value: str) -> str:
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise UnitFailure("malformed physical GPU UUID")
    return "GPU-" + value[4:].lower()


def verify_output(output: str, expected_uuid: str | None = None) -> None:
    """Positive evidence that the one exact test ran and passed (filtered-out count is unpinned)."""
    lines = output.splitlines()
    if lines.count("running 1 test") != 1:
        raise UnitFailure("libtest did not report exactly one `running 1 test`")
    identities = re.findall(r"MOSS_CUDA_CONTEXT_UUID=([^\s]+)", output)
    if expected_uuid is not None:
        if len(identities) != 1 or canonical_gpu_uuid(identities[0]) != canonical_gpu_uuid(expected_uuid):
            raise UnitFailure("missing, duplicate or mismatched actual CUDA context UUID")
    # --nocapture can print after the libtest prefix but before its final `ok`. Remove ONLY
    # the identity marker; retain exact FQN, one running test, and the strict result summary.
    normalized = re.sub(r"MOSS_CUDA_CONTEXT_UUID=[^\s]+", "", output).replace("\r\n", "\n")
    names = re.findall(r"(?m)^test (\S+) \.\.\.", normalized)
    if names != [TEST_NAME] or not re.search(rf"(?m)^test {re.escape(TEST_NAME)} \.\.\.[ \t\r\n]*ok$", normalized):
        raise UnitFailure(f"{TEST_NAME} was not reported ok")
    results = [line for line in lines if line.startswith("test result:")]
    if len(results) != 1 or not re.fullmatch(
        r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*",
        results[0],
    ):
        raise UnitFailure(f"unexpected libtest result line(s): {results}")


class PosixOwner:
    """Use the same session + birth-token containment as the native fixture owner."""

    def spawn(self, argv: list[str], env: dict[str, str], stdout, stderr) -> subprocess.Popen:
        # Lazy import avoids the helper's supervise import cycle. Never a second owner.
        from scripts.ci.moss_fixture_cpu import CpuOwner
        self.owner = CpuOwner()
        self.child = self.owner.spawn(argv, env, stdout, stderr)
        self.pgid = self.child.pid
        return self.child

    def footprint(self) -> int:
        return self.owner.footprint()

    def alive(self) -> bool:
        return bool(self.owner.collect())

    def terminate(self, grace: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self.owner.terminate(grace, sleep)

    def close(self) -> None:
        self.owner.close()


class WindowsOwner:
    """A non-breakaway Job Object, shared with the media campaign supervisor."""

    def spawn(self, argv: list[str], env: dict[str, str], stdout, stderr) -> subprocess.Popen:
        from scripts import media_campaign_windows as windows

        self.job = windows.WindowsJob()
        # CREATE_SUSPENDED: no descendant starts before Job ownership is installed.
        self.child = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=0x00000004
        )
        try:
            self.job.assign_and_resume(self.child)
        except BaseException:
            self.child.kill()
            self.child.wait(timeout=GRACE_SECONDS)
            self.job.close()
            raise
        return self.child

    def footprint(self) -> int:
        return self.job.footprint_bytes()

    def terminate(self, grace: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self.job.terminate_and_reap(self.child, grace)

    def close(self) -> None:
        self.job.close()


def supervise(
    argv: list[str],
    owner,
    out_path: Path,
    *,
    deadline: float = DEADLINE_SECONDS,
    rss_cap: int = RSS_CAP_BYTES,
    poll: float = POLL_SECONDS,
    grace: float = GRACE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    env: dict[str, str] | None = None,
) -> str:
    """Run argv once; return its combined output or raise UnitFailure. The owned tree never survives."""
    environment = {k: v for k, v in (os.environ if env is None else env).items() if k not in SCRUBBED_ENV}
    environment["RUST_TEST_THREADS"] = "1"
    started = clock()
    failure: str | None = None
    with out_path.open("wb") as sink:
        child = owner.spawn(argv, environment, sink, subprocess.STDOUT)
        try:
            while child.poll() is None:
                if clock() - started > deadline:
                    failure = f"exceeded the {deadline:g}s execution deadline"
                    break
                used = owner.footprint()
                if used > rss_cap:
                    failure = f"owned process tree used {used} bytes, above the {rss_cap} byte cap"
                    break
                sleep(poll)
        finally:
            # Also reached on KeyboardInterrupt / probe errors: the tree never outlives the supervisor.
            try:
                owner.terminate(grace, sleep)
            finally:
                owner.close()
    if failure:
        raise UnitFailure(failure)
    if child.returncode != 0:
        raise UnitFailure(f"test process exited with {child.returncode}")
    with out_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - OUTPUT_TAIL_BYTES))
        return handle.read().decode("utf-8", "replace")


def target_dir() -> Path:
    return Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")


def smi_path() -> str:
    from scripts import media_campaign_windows as windows

    return windows.trusted_nvidia_smi()


def smi_query(*args: str) -> str:
    result = subprocess.run(
        [smi_path(), *args, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    if result.returncode != 0:
        raise UnitFailure(f"nvidia-smi {args[0]} exited with {result.returncode}")
    return result.stdout


RESOURCE_FILE = REPO_ROOT / "crates/media/candle-gen/candle-gen-moss-image/src/tests/cuda_unit_resources.json"
EXPECTED_UUID_ENV = "MOSS_CUDA_UNIT_EXPECTED_GPU_UUID"
MIB = 1024**2


def positive_integer(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise UnitFailure(f"invalid {label}")
    return value


def read_unit_resources() -> tuple[dict, dict, dict]:
    try:
        resource = json.loads(RESOURCE_FILE.read_text(encoding="utf-8"))
        author = (RESOURCE_FILE.parents[2] / resource["author_fixture_root"]).resolve()
        if not author.is_relative_to(REPO_ROOT.resolve()):
            raise UnitFailure("unit author fixture must stay inside the repository")
        return (resource, json.loads((author / "fixture.json").read_text(encoding="utf-8")),
                json.loads((author / "dmd_fixture.json").read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as error:
        raise UnitFailure("invalid shared unit resource descriptor") from error


def derive_unit_budget(resource: dict, fixture: dict, dmd: dict) -> dict:
    """Structural overprovisioning, NOT measured consumption or a universal CUDA upper bound."""
    def elements(tensor: dict) -> int:
        shape = tensor["shape"]
        if not isinstance(shape, list) or not shape:
            raise UnitFailure("invalid fixture tensor shape")
        size = 1
        for axis in shape: size *= positive_integer(axis, "tensor axis")
        if len(tensor["values"]) != size:
            raise UnitFailure("fixture tensor shape/value population differs")
        return size
    try:
        weights = fixture["weights"]
        weight_elements = sum(elements(tensor) for tensor in weights.values())
        config = fixture["config"]
        dim, ffn, heads, layers = (positive_integer(config[key], key) for key in ("dim", "ffn_dim", "num_heads", "num_layers"))
        image = fixture["inputs"]["hidden_states"]
        context = fixture["inputs"]["encoder_hidden_states"]
        elements(image); elements(context)
        batch = positive_integer(resource["batch"], "unit batch")
        if len(image["shape"]) != 4 or len(context["shape"]) != 3 or batch > min(image["shape"][0], context["shape"][0]):
            raise UnitFailure("invalid unit fixture batch geometry")
        _, channels, height, width = image["shape"]
        _, context_tokens, context_dim = context["shape"]
        if channels != config["latent_channels"] or context_dim != config["text_dim"] or dim % heads:
            raise UnitFailure("unit fixture channels/heads differ from model")
        sequence = height * width + context_tokens
        # Largest FFN/QKV output or FP32 attention matrix of the actual selected fixture rows.
        largest_tensor = 4 * batch * max(sequence * max(3 * dim, 2 * ffn, dim), heads * sequence * sequence)
        text_layers = {key.split(".")[2] for key in weights if key.startswith("text_embedder.blocks.")}
        blocks = layers + len(text_layers)
        rank = positive_integer(resource["rank"], "unit rank")
        factor_elements = 0
        for layer in range(layers):
            for target in resource["target_modules"]:
                shape = weights[f"layers.{layer}.proj.{target}.weight"]["shape"]
                if len(shape) != 2: raise UnitFailure("unit adapter target is not a matrix")
                out_f, in_f = shape
                # Charge full LoRA plus a bound covering full/decomposed LoKr representations,
                # without constructing a Kronecker matrix or duplicating backend factorization.
                factor_elements += rank * (out_f + in_f) + 2 * out_f * in_f + rank * (out_f + in_f + 2)
        dmd_elements = sum(elements(t) for t in dmd["replacement_tensors"].values())
        for adapter in dmd["adapters"].values():
            dmd_elements += elements(adapter["down"]) + elements(adapter["up"]) + 1
        accounting = {key: positive_integer(value, key) for key, value in resource["accounting"].items()}
        cells = accounting["case_cache_allowance"]
        transient = (accounting["intermediates_per_block"] * accounting["retained_passes"] *
                     accounting["value_adjoint_slots"] * blocks * cells * largest_tensor)
        input_elements = batch * (channels * height * width + context_tokens * context_dim)
        persistent = cells * accounting["persistent_copies"] * 4 * (weight_elements + factor_elements + dmd_elements + input_elements)
        unrounded = transient + persistent
        rounded = 1 << (unrounded - 1).bit_length()
        vendor = accounting["vendor_reserve_bytes"]
        return {"largest_tensor_bytes": largest_tensor, "blocks": blocks, "transient_bytes": transient,
                "persistent_bytes": persistent, "unrounded_bytes": unrounded, "tensor_bytes": rounded,
                "vendor_reserve_bytes": vendor, "required_bytes": rounded + vendor}
    except (KeyError, TypeError, ValueError) as error:
        raise UnitFailure("invalid source fixture/resource accounting") from error


def require_gpu1_budget(query=None, budget: dict | None = None) -> dict:
    """WDDM desktop rows are NOT exclusion evidence. This is a fallible resource check, not a lease."""
    query = smi_query if query is None else query
    budget = derive_unit_budget(*read_unit_resources()) if budget is None else budget
    try:
        rows = list(csv.reader(query("--query-gpu=index,uuid,memory.free,memory.total").splitlines(), skipinitialspace=True, strict=True))
        if not rows or any(len(row) != 4 or not row[0].strip().isdigit() for row in rows):
            raise UnitFailure("malformed GPU resource inventory")
        uuids = [canonical_gpu_uuid(row[1].strip()) for row in rows]
        if len(set(uuids)) != len(rows) or len({int(row[0]) for row in rows}) != len(rows):
            raise UnitFailure("ambiguous GPU resource inventory")
        selected = [(row, uuid) for row, uuid in zip(rows, uuids) if int(row[0]) == 1]
        if len(selected) != 1: raise UnitFailure("GPU 1 is not present exactly once")
        row, uuid = selected[0]
        free, total = Decimal(row[2].strip()), Decimal(row[3].strip())
        if not free.is_finite() or not total.is_finite() or free < 0 or total <= 0 or free > total:
            raise UnitFailure("unknown or invalid GPU 1 free/total VRAM")
        free_bytes = int(free * MIB)  # Round fractional bytes DOWN; never inflate reported free VRAM.
        if free_bytes < budget["required_bytes"]:
            raise UnitFailure(f"GPU 1 has {free_bytes} free bytes, below unit requirement {budget['required_bytes']}")
        return {"uuid": uuid, "free_bytes": free_bytes, "budget": budget}
    except UnitFailure:
        raise
    except (OSError, subprocess.SubprocessError, InvalidOperation, ValueError, csv.Error, RuntimeError) as error:
        raise UnitFailure("GPU resource inventory unavailable or malformed") from error


def provenance() -> str:
    """Device and driver name for the log only; never an input to pass/fail."""
    try:
        return smi_query("--query-gpu=name,driver_version").strip()
    except Exception as error:  # informational; the test itself requires a CUDA device
        return f"unavailable ({error})"


def smoke_failure(result: unittest.TestResult) -> str | None:
    """Why a smoke run proves nothing: any failure, any skip, or nothing executed."""
    executed = result.testsRun - len(result.skipped)
    if not result.wasSuccessful():
        return "Windows ownership smoke failed"
    if result.skipped or executed <= 0:
        return f"Windows ownership smoke executed {executed} tests with {len(result.skipped)} skipped"
    return None


def smoke_command(_unused: Path | None = None) -> None:
    if os.name != "nt":
        raise UnitFailure("Windows process ownership is unavailable; refusing to run unguarded")
    suite = unittest.defaultTestLoader.loadTestsFromNames(SMOKE_TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    reason = smoke_failure(result)
    if reason:
        raise UnitFailure(reason)
    print(f"moss-cuda-unit: ownership smoke executed {result.testsRun} tests")


def compile_command(destination: Path) -> None:
    result = subprocess.run(
        COMPILE_ARGV, cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise UnitFailure(f"cargo test --no-run exited with {result.returncode}")
    exe = select_artifact(result.stdout, target_dir())
    destination.write_text(str(exe), encoding="utf-8")
    print(f"moss-cuda-unit: compiled {exe}")


def run_command(source: Path) -> None:
    if os.name != "nt":
        raise UnitFailure("Windows process ownership is unavailable; refusing to run unguarded")
    exe = Path(source.read_text(encoding="utf-8").strip())
    if not exe.is_file():
        raise UnitFailure(f"recorded test executable {exe} is missing")
    print(f"moss-cuda-unit: device {provenance()}")
    admission = require_gpu1_budget()
    print(f"moss-cuda-unit: GPU1 uuid={admission['uuid']} free_bytes={admission['free_bytes']} required_bytes={admission['budget']['required_bytes']}; resource check, not idle/exclusive")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = admission["uuid"]
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["SCENEWORKS_GPU_ID"] = "1"
    environment[EXPECTED_UUID_ENV] = admission["uuid"]
    owner = WindowsOwner()
    with tempfile.TemporaryDirectory() as scratch:
        output = supervise([str(exe), *RUN_ARGS], owner, Path(scratch) / "unit.log", env=environment)
    sys.stdout.write(output[-8192:])
    verify_output(output, admission["uuid"])
    print("moss-cuda-unit: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "run", "smoke"))
    parser.add_argument("--exe-file", type=Path, default=Path(os.environ.get("RUNNER_TEMP", ".")) / EXE_FILE)
    args = parser.parse_args(argv)
    try:
        {"compile": compile_command, "run": run_command, "smoke": smoke_command}[args.command](args.exe_file)
    except (UnitFailure, OSError, subprocess.SubprocessError) as error:
        print(f"moss-cuda-unit: FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
