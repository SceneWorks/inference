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
RUN_ARGS = [TEST_NAME, "--exact", "--nocapture", "--color", "never", "--test-threads=1"]
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


def verify_output(output: str) -> None:
    """Positive evidence that the one exact test ran and passed (filtered-out count is unpinned)."""
    lines = output.splitlines()
    if lines.count("running 1 test") != 1:
        raise UnitFailure("libtest did not report exactly one `running 1 test`")
    if f"test {TEST_NAME} ... ok" not in lines:
        raise UnitFailure(f"{TEST_NAME} was not reported ok")
    results = [line for line in lines if line.startswith("test result:")]
    if len(results) != 1 or not re.fullmatch(
        r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*",
        results[0],
    ):
        raise UnitFailure(f"unexpected libtest result line(s): {results}")


class PosixOwner:
    """A new process group; used by the unit tests and any non-Windows host."""

    def spawn(self, argv: list[str], env: dict[str, str], stdout, stderr) -> subprocess.Popen:
        self.child = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        self.pgid = self.child.pid
        return self.child

    def footprint(self) -> int:
        table = subprocess.run(
            ["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, check=True, timeout=10
        ).stdout
        total = 0
        for row in table.splitlines():
            fields = row.split()
            if len(fields) == 2 and fields[0] == str(self.pgid):
                total += int(fields[1]) * 1024
        return total

    def alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        return True

    def terminate(self, grace: float, sleep: Callable[[float], None] = time.sleep) -> None:
        """Kill only this group, reap the root, then require the group to be empty."""
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child.wait(timeout=grace)
        for _ in range(int(grace / 0.01)):
            if not self.alive():
                return
            sleep(0.01)
        raise UnitFailure("owned process group still has live processes after termination")

    def close(self) -> None:
        pass


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


def provenance() -> str:
    """Device and driver name for the log only; never an input to pass/fail."""
    try:
        from scripts import media_campaign_windows as windows

        smi = windows.trusted_nvidia_smi() if os.name == "nt" else "nvidia-smi"
        return subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
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
    owner = WindowsOwner()
    with tempfile.TemporaryDirectory() as scratch:
        output = supervise([str(exe), *RUN_ARGS], owner, Path(scratch) / "unit.log")
    sys.stdout.write(output[-8192:])
    verify_output(output)
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
