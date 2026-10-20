#!/usr/bin/env python3
"""Compile and run exactly one MOSS CUDA unit test under an owned, bounded process tree.

`compile` builds the fixed package's lib test binary and records the one matching executable.
`run` starts that binary with the fixed libtest filter under a wall-clock and RSS ceiling, then
requires a positive `running 1 test` / `1 passed` result. No command loads model weights.
`inventory` reads current NVML/CIM metadata only (including reported memory), never starts CUDA,
and is neither calibration nor an ownership lease. Unit caps remain safety limits. A missing CUDA
device fails the test itself; this module never reports a pass it did not observe.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import ntpath
import platform
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
            ["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, encoding="utf-8", check=True, timeout=10
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


def require_gpu0_idle(query: Callable[..., str] = smi_query) -> None:
    """Refuse to start while any compute process is on GPU 0, or if occupancy cannot be read.

    A pre-spawn check only: it narrows the window left by the compile step, it is not a lease
    and does not stop a process that starts later. Foreign processes are never signalled.
    """
    try:
        gpus = [[f.strip() for f in row.split(",")] for row in query("--query-gpu=index,uuid").splitlines() if row.strip()]
        apps = [[f.strip() for f in row.split(",")] for row in query("--query-compute-apps=gpu_uuid,pid").splitlines() if row.strip()]
    except (OSError, subprocess.SubprocessError) as error:
        raise UnitFailure(f"GPU occupancy unreadable: {error}") from error
    uuids = {row[1] for row in gpus if len(row) == 2 and row[0].isdigit() and row[1].startswith("GPU-")}
    if len(uuids) != len(gpus) or not gpus or len({row[0] for row in gpus}) != len(gpus):
        raise UnitFailure("GPU inventory is malformed or ambiguous")
    gpu0 = [row[1] for row in gpus if row[0] == "0"]
    if len(gpu0) != 1:
        raise UnitFailure("GPU 0 is not present exactly once")
    for row in apps:
        if len(row) != 2 or row[0] not in uuids or not row[1].isdigit():
            raise UnitFailure(f"malformed or unknown compute process row: {row}")
        if row[0] == gpu0[0]:
            raise UnitFailure(f"GPU 0 already has compute process {row[1]}; not starting")


INVENTORY_FILE = "moss-cuda-inventory.json"
INVENTORY_MACHINE = "MICHAEL-TRX50"
INVENTORY_TIMEOUT_SECONDS = 10
UNKNOWN = "UNKNOWN"


def inventory_query(*args: str) -> str:
    """Only the fixed read-only NVML queries below call this; never initializes CUDA."""
    try:
        executable = smi_path()
    except RuntimeError as error:
        raise UnitFailure("trusted NVML inventory tool unavailable") from error
    result = subprocess.run(
        [executable, *args, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, encoding="utf-8", timeout=INVENTORY_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise UnitFailure(f"inventory query exited with {result.returncode}")
    return result.stdout


def inventory_memory(value: str) -> float | str:
    try:
        number = float(value)
    except ValueError:
        return UNKNOWN
    return number if math.isfinite(number) and number >= 0 else UNKNOWN


def inventory_basename(value: str) -> str:
    return UNKNOWN if value.strip().upper() in {"", "N/A", "[N/A]", "[NOT SUPPORTED]", "[NOT FOUND]"} else ntpath.basename(value.strip()) or UNKNOWN


def inventory_processes(pids: list[int]) -> dict[int, dict]:
    """Restricted optional CIM metadata for NVML-observed PIDs, never a process dump."""
    if not pids:
        return {}
    if any(type(pid) is not int or not 0 < pid <= 0xFFFFFFFF for pid in pids):
        raise UnitFailure("invalid NVML PID for CIM query")
    ids = sorted(set(pids))
    where = " OR ".join(f"ProcessId = {pid}" for pid in ids)
    fields = "ProcessId,ParentProcessId,Name,CreationDate,SessionId"
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$rows=@(Get-CimInstance -ClassName Win32_Process -Filter '{where}' -Property {fields}); "
        f"ConvertTo-Json -InputObject @($rows | Select-Object {fields}) -Compress"
    )
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", timeout=INVENTORY_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise UnitFailure(f"restricted CIM query exited with {result.returncode}")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise UnitFailure("restricted CIM result is not an array")
    found = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("ProcessId")) is not int or row["ProcessId"] not in ids:
            raise UnitFailure("restricted CIM returned an unexpected PID")
        # Copy ONLY these fields; never log raw output, extra properties, paths or command lines.
        integer = lambda value: value if type(value) is int and value >= 0 else UNKNOWN
        created = row.get("CreationDate")
        found[row["ProcessId"]] = {
            "pid": row["ProcessId"], "parent_pid": integer(row.get("ParentProcessId")),
            "name": inventory_basename(row.get("Name")) if isinstance(row.get("Name"), str) else UNKNOWN,
            "creation_time": created if isinstance(created, str) and created else UNKNOWN,
            "session_id": integer(row.get("SessionId")),
        }
    return found


def inventory_document(query=inventory_query, process_query=inventory_processes) -> dict:
    """Current metadata only, not a lease, an idle assertion, or historical PID ownership proof."""
    machine = platform.node()
    document = {
        "schema": 1, "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": machine, "runner": os.environ.get("RUNNER_NAME") or UNKNOWN,
        "probe_pid": os.getpid(), "probe_parent_pid": os.getppid(),
        "expected_machine": INVENTORY_MACHINE, "source_sha": os.environ.get("GITHUB_SHA") or UNKNOWN,
        "status": "WRONG_MACHINE", "gpus": {"status": UNKNOWN, "rows": []},
        "compute_processes": {"status": UNKNOWN, "rows": []}, "windows_metadata": {"status": UNKNOWN},
    }
    if platform.system() != "Windows" or machine.upper() != INVENTORY_MACHINE:
        return document
    try:
        rows = list(csv.reader(query("--query-gpu=index,uuid,name,memory.free,memory.total").splitlines(), skipinitialspace=True, strict=True))
        gpus = []
        for row in rows:
            if len(row) != 5 or not row[0].strip().isdigit() or not row[1].strip().startswith("GPU-"):
                raise UnitFailure("malformed GPU inventory")
            gpus.append({"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip() or UNKNOWN,
                         "free_memory_mib": inventory_memory(row[3]), "total_memory_mib": inventory_memory(row[4])})
        if not gpus or len({g["index"] for g in gpus}) != len(gpus) or len({g["uuid"] for g in gpus}) != len(gpus):
            raise UnitFailure("empty or ambiguous GPU inventory")
        document["gpus"] = {"status": "KNOWN", "rows": gpus}
    except (OSError, subprocess.SubprocessError, UnitFailure, ValueError, csv.Error) as error:
        document["gpus"]["reason"] = type(error).__name__
    try:
        rows = list(csv.reader(query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory").splitlines(), skipinitialspace=True, strict=True))
        indices = {g["uuid"]: g["index"] for g in document["gpus"]["rows"]}
        apps = []
        for row in rows:
            if len(row) != 4 or not row[0].strip().startswith("GPU-") or not row[1].strip().isdigit() or not 0 < int(row[1]) <= 0xFFFFFFFF:
                raise UnitFailure("malformed compute inventory")
            apps.append({"gpu_uuid": row[0].strip(), "gpu_index": indices.get(row[0].strip(), UNKNOWN),
                         "pid": int(row[1]), "process_basename": inventory_basename(row[2]),
                         "used_memory_mib": inventory_memory(row[3]), "windows_process": UNKNOWN})
        document["compute_processes"] = {"status": "KNOWN", "rows": apps}
        try:
            metadata = process_query(sorted({a["pid"] for a in apps}))
            for app in apps:
                app["windows_process"] = metadata.get(app["pid"], UNKNOWN)
            document["windows_metadata"] = {"status": "KNOWN" if all(app["windows_process"] != UNKNOWN for app in apps) else UNKNOWN}
        except (OSError, subprocess.SubprocessError, UnitFailure, ValueError) as error:
            document["windows_metadata"]["reason"] = type(error).__name__
    except (OSError, subprocess.SubprocessError, UnitFailure, ValueError, csv.Error) as error:
        document["compute_processes"]["reason"] = type(error).__name__
    document["status"] = "CAPTURED" if all(document[key]["status"] == "KNOWN" for key in ("gpus", "compute_processes")) else "PARTIAL"
    return document


def inventory_command(_unused: Path | None = None) -> None:
    root = os.environ.get("RUNNER_TEMP")
    if not root:
        raise UnitFailure("inventory requires the runner temporary directory")
    document = inventory_document()
    text = json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)
    (Path(root) / INVENTORY_FILE).write_text(text + "\n", encoding="utf-8")
    print(text)
    if document["status"] != "CAPTURED":
        raise UnitFailure(f"inventory incomplete: {document['status']}; unknown is not idle")


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
    require_gpu0_idle()
    owner = WindowsOwner()
    with tempfile.TemporaryDirectory() as scratch:
        output = supervise([str(exe), *RUN_ARGS], owner, Path(scratch) / "unit.log")
    sys.stdout.write(output[-8192:])
    verify_output(output)
    print("moss-cuda-unit: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "run", "smoke", "inventory"))
    parser.add_argument("--exe-file", type=Path, default=Path(os.environ.get("RUNNER_TEMP", ".")) / EXE_FILE)
    args = parser.parse_args(argv)
    try:
        {"compile": compile_command, "run": run_command, "smoke": smoke_command, "inventory": inventory_command}[args.command](args.exe_file)
    except (UnitFailure, OSError, subprocess.SubprocessError) as error:
        print(f"moss-cuda-unit: FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
