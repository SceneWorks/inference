#!/usr/bin/env python3
"""Compile and supervise only the explicit native MOSS full-pipeline CPU fixture tests.

Compilation has its separate CI job budget. Each actual test runs alone under an owned process
session, 180-second deadline and 2-GiB RSS cap. No GPU inventory, models, caches or driver APIs.
"""
from __future__ import annotations
import argparse
import hashlib
import ctypes
import errno
import signal
import time
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.moss_cuda_unit import UnitFailure, supervise  # noqa: E402

PACKAGES = {"candle": "candle-gen-moss-image", "mlx": "mlx-gen-moss-image"}
CASES = (
    "test_support::tests::full_image_caption_base_lora_cached_adamw_resume_reentry",
    "test_support::tests::full_image_caption_base_lokr_uncached_text_adam_resume_reentry",
    "test_support::tests::full_image_caption_sft_lora_uncached_latents_prodigy_resume_reentry",
    "test_support::tests::full_image_caption_sft_lokr_uncached_both_rose_resume_reentry",
)
DEMAND_TEST = "test_support::tests::header_only_demand_profile"
TOKEN_TEST = "test_support::tests::actual_ragged_token_bound_short_long_mixed_preview_trigger_and_truncation"
RSS_CAP = 2 * 1024**3
DEADLINE = 180.0


class _BsdInfo(ctypes.Structure):
    # Verified against sys/proc_info.h PROC_PIDTBSDINFO. Command/name slots are never inspected.
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "reserved")]
    _fields_ += [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
    _fields_ += [(name, ctypes.c_uint32) for name in ("nfiles", "pgid", "jobc", "tty", "ttypgid")]
    _fields_ += [("nice", ctypes.c_int32), ("start_seconds", ctypes.c_uint64), ("start_microseconds", ctypes.c_uint64)]


def process_identity(pid: int) -> tuple[int, int] | None:
    """Only UID and OS birth token; never command lines, environment or process memory."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
        library.proc_pidinfo.restype = ctypes.c_int
        info = _BsdInfo()
        ctypes.set_errno(0)
        length = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if length == 0 and ctypes.get_errno() == errno.ESRCH:
            return None
        if length != ctypes.sizeof(info):
            raise UnitFailure(f"owned PID birth metadata unavailable for {pid}")
        if info.status == 5:  # zombie; Popen.wait reaps the owned root separately.
            return None
        return info.uid, info.start_seconds * 1_000_000 + info.start_microseconds
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return path.stat().st_uid, int(fields[19])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError) as error:
        raise UnitFailure(f"owned PID birth metadata unavailable for {pid}") from error


class CpuOwner:
    """PID+birth+UID descendant ownership, including descendants changing process groups.

    The child first waits at a private stdin gate, so identity is captured before its test binary
    executes. Cleanup rechecks each exact birth/UID immediately before signaling only owned PIDs.
    """
    def __init__(self):
        self.uid = os.geteuid()
        self.known = {}
        self.parents = {}
        self.peak = 0

    def spawn(self, argv, env, stdout, stderr):
        self.child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--owned-exec", *argv],
                                      cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=stdout,
                                      stderr=stderr, start_new_session=True)
        try:
            identity = process_identity(self.child.pid)
            if identity is None or identity[0] != self.uid:
                raise UnitFailure("new CPU child identity unavailable or UID differs")
            self.known[self.child.pid] = identity
            self.child.stdin.write(b"R")
            self.child.stdin.flush()
            self.child.stdin.close()
        except BaseException:
            if not self.child.stdin.closed:
                self.child.stdin.close()  # EOF refuses exec; no unverifiable PID is signaled.
            self.child.wait(timeout=10)
            raise
        return self.child

    def collect(self):
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,uid=,rss="], capture_output=True,
                                text=True, encoding="utf-8", check=True, timeout=10)
        table = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) != 4:
                raise UnitFailure("invalid process ownership metadata")
            pid, parent, uid, rss = map(int, fields)
            table[pid] = parent, uid, rss * 1024
        live = {}
        for pid, identity in self.known.items():
            if pid in table and table[pid][1] == self.uid and process_identity(pid) == identity:
                live[pid] = table[pid]
        changed = True
        while changed:
            changed = False
            for pid, (parent, uid, rss) in table.items():
                if pid not in live and parent in live:
                    identity = process_identity(pid)
                    if identity is None:
                        continue
                    if uid != self.uid or identity[0] != self.uid or identity[1] < self.known[parent][1]:
                        raise UnitFailure("descendant UID/birth ownership differs")
                    self.known[pid] = identity
                    self.parents[pid] = parent
                    live[pid] = parent, uid, rss
                    changed = True
        return live

    def footprint(self):
        total = sum(row[2] for row in self.collect().values())
        self.peak = max(self.peak, total)
        return total

    def _signal(self, signum):
        for pid in reversed(tuple(self.collect())):
            if process_identity(pid) == self.known[pid] and self.known[pid][0] == self.uid:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass

    def terminate(self, grace, sleep=time.sleep):
        self._signal(signal.SIGTERM)
        end = time.monotonic() + grace
        while time.monotonic() < end:
            self.child.poll()
            if not self.collect():
                break
            sleep(0.05)
        if self.collect():
            self._signal(signal.SIGKILL)
        self.child.wait(timeout=grace)
        end = time.monotonic() + grace
        while self.collect():
            if time.monotonic() >= end:
                raise UnitFailure("owned CPU descendants were not released")
            sleep(0.05)

    def close(self):
        pass


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_stamp() -> dict:
    delta = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    return {"head": git_head(), "tracked_delta_sha256": hashlib.sha256(delta).hexdigest()}


def compile_argv(backend: str) -> list[str]:
    return ["cargo", "test", "--locked", "--lib", "-j", "1", "-p", PACKAGES[backend],
            "--features", "test-support", "--no-run", "--message-format=json"]


def render_compile_errors(stdout: str) -> None:
    """Cargo JSON goes to the sealed artifact; also expose its actual errors in the job log."""
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("reason") == "compiler-message":
            message = record.get("message") or {}
            if message.get("level") == "error":
                print(message.get("rendered") or message.get("message", "compiler error"), file=sys.stderr, flush=True)


def select_binary(stdout: str, package: str, target: Path, *, lib_name: str | None = None) -> Path:
    found = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise UnitFailure("compiler stdout was not Cargo JSON") from error
        if not isinstance(message, dict) or message.get("reason") != "compiler-artifact":
            continue
        artifact = message.get("target") or {}
        if (re.search(rf"(?:/|#|^){re.escape(package)}(?:[@#\s]|$)", str(message.get("package_id", "")))
                and artifact.get("name") == (lib_name or package.replace("-", "_"))
                and artifact.get("kind") == ["lib"]
                and (message.get("profile") or {}).get("test") is True
                and message.get("executable")):
            found.append(Path(message["executable"]))
    if len(found) != 1:
        raise UnitFailure(f"expected one {package} lib test executable, got {len(found)}")
    binary = found[0]
    if not binary.is_absolute() or not binary.is_file() or target.resolve() not in binary.resolve().parents:
        raise UnitFailure("test executable is outside the owned Cargo target")
    return binary


def verify(output: str, name: str) -> None:
    lines = output.splitlines()
    normalized = re.sub(r"MOSS_FIXTURE_DEMAND=\{[^\n]*\}", "", output).replace("\r\n", "\n")
    names = re.findall(r"(?m)^test (\S+) \.\.\.", normalized)
    summaries = [line for line in lines if line.startswith("test result:")]
    if (lines.count("running 1 test") != 1 or names != [name]
            or not re.search(rf"(?m)^test {re.escape(name)} \.\.\.[ \t\n]*ok$", normalized)
            or len(summaries) != 1
            or not re.fullmatch(r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*", summaries[0])):
        raise UnitFailure(f"no positive exact one-test success for {name}")


def validate_demand(output: str) -> dict:
    records = re.findall(r"MOSS_FIXTURE_DEMAND=(\{[^\n]*\})", output)
    if len(records) != 1:
        raise UnitFailure("missing or duplicate actual Rust demand profile")
    try:
        record = json.loads(records[0])
        token_host=record["token_bound_case_host_envelope_bytes"]
        if record["cpu_reference"] is not True or type(token_host) is not int or not 0<token_host<=RSS_CAP:
            raise ValueError("actual token-shape case does not fit its CPU envelope")
        plans = record["cases"]
        if not isinstance(plans, list) or len(plans) != len(CASES):
            raise ValueError("fixture demand case population differs")
        for plan in plans:
            tensors = plan["tensor_peak_bytes"]
            host = plan["host_envelope_bytes"]
            if (type(tensors) is not int or type(host) is not int or tensors <= 0
                    or host < tensors or host > RSS_CAP):
                raise ValueError("derived fixture host demand does not fit its safety ceiling")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise UnitFailure(f"invalid fixture demand profile: {error}") from error
    return record


class CaseAssertionFailure(UnitFailure):
    """An exact, released libtest assertion failure; not admission/source/protocol failure."""
    def __init__(self, result: dict):
        self.result = result
        super().__init__(f"named baseline assertion failed: {result['test']}")


def log_tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - (1 << 20)))
        return handle.read().decode("utf-8", "replace")


def verify_assertion_failure(output: str, name: str) -> None:
    names = re.findall(r"(?m)^test (\S+) \.\.\.", output)
    summaries = [line for line in output.splitlines() if line.startswith("test result:")]
    if (output.splitlines().count("running 1 test") != 1 or names != [name]
            or not re.search(rf"(?ms)^test {re.escape(name)} \.\.\.[ \t]*(?:FAILED$|\n.*?^FAILED$)", output)
            or len(summaries) != 1
            or not re.fullmatch(r"test result: FAILED\. 0 passed; 1 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .*", summaries[0])
            or "panicked at" not in output or "Error:" in output):
        raise UnitFailure("unclassified native/source/admission failure; refusing further cases")


def read_observations(log: Path, head: str, required: bool) -> dict | None:
    path = log.with_suffix(".receipts.json")
    if not path.exists():
        if required:
            raise UnitFailure("training observations missing; admission/source readiness is not established")
        return None
    if path.is_symlink() or not path.is_file():
        raise UnitFailure("training observations are not an owned regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_head = head if os.environ.get("GITHUB_SHA") else None
    if value.get("version") != "moss-cpu-training-observations-v1" or value.get("source_head") != expected_head:
        raise UnitFailure("training observation source binding differs")
    evaluation = value.get("evaluation") or {}
    if (not re.fullmatch(r"[0-9a-f]{64}", evaluation.get("identity", ""))
            or any(type(evaluation.get(axis)) is not int or evaluation[axis] <= 0 or evaluation[axis] % 32
                   for axis in ("width", "height"))
            or type(evaluation.get("noise_seed")) is not int or not 0 <= evaluation["noise_seed"] < 2**64
            or type(evaluation.get("sigma")) not in (int, float) or not 0 <= evaluation["sigma"] <= 1):
        raise UnitFailure("fixed evaluation manifest metadata is missing or invalid")
    for field in ("image_sha256", "mask_sha256"):
        digest = evaluation.get(field)
        if field == "mask_sha256" and digest is None:
            continue
        if not isinstance(digest, list) or len(digest) != 32 or any(type(n) is not int or not 0 <= n <= 255 for n in digest):
            raise UnitFailure("fixed evaluation input digest differs")
    receipts = value.get("receipts")
    delta = value.get("parameter_hash_changed")
    if not isinstance(receipts, list) or not receipts or not isinstance(delta, list) or len(delta) != len(receipts):
        raise UnitFailure("training observations lack actual runs")
    workspace = value.get("admitted_audit_workspace_bytes")
    if type(workspace) is not int or workspace <= 0:
        raise UnitFailure("diagnostic workspace admission missing")
    def diagnostic_file(path, digest):
        path = Path(path)
        if (path.resolve().parent != log.parent.resolve() or path.is_symlink() or not path.is_file()
                or file_digest(path) != digest):
            raise UnitFailure("diagnostic native export path/content differs")
    def factor_delta(delta, initial_sha, layout):
        if (not isinstance(delta, dict) or delta.get("initial_sha256") != initial_sha or delta.get("layout_sha256") != layout
                or not re.fullmatch(r"[0-9a-f]{64}", delta.get("final_sha256", ""))
                or type(delta.get("elements")) is not int or delta["elements"] <= 0
                or type(delta.get("changed_elements")) is not int or not 0 <= delta["changed_elements"] <= delta["elements"]
                or type(delta.get("workspace_bytes")) is not int or not 0 < delta["workspace_bytes"] <= workspace
                or any(type(delta.get(key)) not in (int, float) or not 0 <= delta[key] < float("inf")
                       for key in ("max_abs_delta", "squared_l2_delta"))):
            raise UnitFailure("streamed actual factor delta is missing or malformed")
    for index, receipt in enumerate(receipts):
        identity = receipt.get("request_identity") or {}
        seed = receipt.get("resolved_seed")
        if (type(seed) is not int or not 0 <= seed < 2**64
                or not identity.get("source_checkpoint")
                or not re.fullmatch(r"[0-9a-f]{64}", identity.get("dataset_fingerprint", ""))
                or receipt.get("evaluation_manifest") != evaluation["identity"]
                or receipt.get("cpu_reference") is not True):
            raise UnitFailure("training observations lack actual request/source/seed/dataset/evaluation identity")
        initial = receipt.get("run_start_adapter") or {}
        if (type(initial.get("start_step")) is not int or initial["start_step"] < 0
                or type(initial.get("start_update")) is not int or initial["start_update"] < 0
                or type(initial.get("factor_tensors")) is not int or initial["factor_tensors"] <= 0
                or type(initial.get("workspace_bytes")) is not int or not 0 < initial["workspace_bytes"] <= workspace):
            raise UnitFailure("real native run-start diagnostic export missing")
        diagnostic_file(initial.get("path", ""), initial.get("sha256"))
        exported_identity = dict(initial.get("export_identity", []))
        if (not re.fullmatch(r"[0-9a-f]{64}", initial.get("layout_sha256", ""))
                or exported_identity.get("source_checkpoint_identity") != identity.get("source_checkpoint")
                or exported_identity.get("dataset_fingerprint") != identity.get("dataset_fingerprint")
                or exported_identity.get("effective_training_controls") != identity.get("effective_config")):
            raise UnitFailure("native start export parent/source/config/layout binding differs")
        if receipt.get("final_parameter_delta") is not None:
            exported = receipt["final_parameter_delta"]
            factor_delta(exported, initial["sha256"], initial["layout_sha256"])
            diagnostic_file(receipt.get("final_adapter_path", ""), exported["final_sha256"])
        if receipt.get("complete") is True and receipt.get("final_parameter_delta") is None:
            raise UnitFailure("completed run lacks streamed real final parameter delta")
        if len(receipt.get("checkpoints", [])) > 2:
            raise UnitFailure("unbounded retained checkpoint diagnostics")
        for checkpoint in receipt.get("checkpoints", []):
            if (checkpoint.get("state", {}).get("effective_identity") != identity
                    or type(checkpoint.get("workspace_bytes")) is not int or not 0 < checkpoint["workspace_bytes"] <= workspace
                    or any(type(n) not in (int, float) or not -float("inf") < n < float("inf")
                           for n in checkpoint.get("optimizer_scalars_as_stored", {}).values())):
                raise UnitFailure("checkpoint diagnostic identity/scalars differ")
            factor_delta(checkpoint.get("factor_delta"), initial["sha256"], initial["layout_sha256"])
        controls = json.loads(identity.get("effective_config", "null"))
        if not isinstance(controls, dict) or type(controls.get("seed")) is not int or controls.get("seed") != seed:
            raise UnitFailure("training observation effective seed/config differs")
        before, after = receipt.get("factor_before"), receipt.get("factor_after")
        expected_delta = (before != after) if before and after else None
        if delta[index] is not expected_delta:
            raise UnitFailure("training observation parameter-hash delta differs")
    generation = value.get("generation")
    if generation is not None:
        if not isinstance(generation, dict) or type(generation.get("adapter_applied")) is not bool:
            raise UnitFailure("generation observation is malformed")
        request = generation.get("request") or {}
        if request.get("seed") != receipts[0]["resolved_seed"]:
            raise UnitFailure("generation witness changed the fixed seed")
        for key in ("baseline_shape", "trained_shape", "zero_strength_shape"):
            shape = generation.get(key)
            if not isinstance(shape, list) or len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
                raise UnitFailure("generation witness shape missing")
        elements = generation.get("baseline_elements")
        if type(elements) is not int or elements <= 0:
            raise UnitFailure("generation baseline element count missing")
        control = generation.get("initial_native_export") or {}
        route = generation.get("route_matched_initial")
        binding = generation.get("native_export_binding") or {}
        if (not control or control.get("start_step") != 0
                or binding.get("layout_sha256") != control.get("layout_sha256")
                or binding.get("export_identity") != control.get("export_identity")
                or not binding.get("initial_factor_state_sha256") or not binding.get("trained_factor_state_sha256")):
            raise UnitFailure("generation lacks the real same-parent/source/layout untrained native export")
        diagnostic_file(control.get("path", ""), control.get("sha256"))
        comparisons = [generation.get("trained"), generation.get("zero_strength")]
        if route is not None:
            if not isinstance(route, dict) or type(route.get("adapter_applied")) is not bool:
                raise UnitFailure("route-matched initial control malformed")
            comparisons += [route.get("initial_vs_no_adapter"), route.get("trained_vs_initial")]
        for values in comparisons:
            values = values or {}
            count, nonzero, maximum = values.get("elements"), values.get("nonzero_elements_in_overlap"), values.get("max_abs_delta")
            if (type(count) is not int or count <= 0 or type(nonzero) is not int
                    or not 0 <= nonzero <= min(elements, count) or type(values.get("finite")) is not bool
                    or (values.get("finite") is True and count == elements and maximum is None)
                    or (maximum is not None and (type(maximum) not in (int, float) or not 0 <= maximum < float("inf")))):
                raise UnitFailure("generation delta metadata missing or invalid")
    return {"path": str(path), "sha256": file_digest(path), "runs": len(receipts),
            "generation_observed": generation is not None,
            "route_matched_observed": generation is not None and generation.get("route_matched_initial") is not None
                and generation["route_matched_initial"].get("adapter_applied") is True}


def execute(binary: Path, name: str, log: Path) -> str:
    environment = dict(os.environ)
    environment.update(RUST_TEST_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                       MOSS_CPU_RECEIPT_PATH=str(log.with_suffix(".receipts.json")))
    stamp, digest = source_stamp(), file_digest(binary)
    owner = CpuOwner()
    started = time.monotonic()
    failed = False
    try:
        output = supervise([str(binary), name, "--exact", "--nocapture", "--color", "never", "--test-threads=1"],
                           owner, log, deadline=DEADLINE, rss_cap=RSS_CAP, env=environment)
    except UnitFailure as error:
        # Resource/ownership/protocol failures are fatal, never an independently runnable red case.
        if str(error) != "test process exited with 101" or owner.child.returncode != 101:
            raise
        output = log_tail(log)
        verify_assertion_failure(output, name)
        failed = True
    if owner.collect():
        raise UnitFailure("owned CPU child/descendants not proven released")
    if source_stamp() != stamp or file_digest(binary) != digest:
        raise UnitFailure("source/executable changed during named case")
    if not failed:
        verify(output, name)
    observation = read_observations(log, stamp["head"], name in CASES)
    if name in CASES and not failed and not observation["route_matched_observed"]:
        raise UnitFailure("positive full case lacks actual route-matched FP32 generation observations")
    result = {"test": name, "status": "assertion_failed" if failed else "passed",
              "source": stamp, "binary_sha256": digest, "log_sha256": file_digest(log),
              "observations": observation, "observed_elapsed_seconds": time.monotonic() - started,
              "observed_peak_child_rss_bytes": owner.peak, "configured_rss_cap_bytes": RSS_CAP,
              "configured_deadline_seconds": DEADLINE, "owned_released": True}
    log.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    if failed:
        print(output, end="", file=sys.stderr, flush=True)
        raise CaseAssertionFailure(result)
    return output


def run_cases(binary: Path, backend: str, binary_file: Path, compiler: dict) -> dict:
    logs = binary_file.parent / f"moss-fixture-{backend}-logs"
    logs.mkdir(exist_ok=True)
    batch = {"source": compiler["source"], "passed": False, "header_admitted": False,
             "cases": [], "fatal_error": None}
    try:
        owner = CpuOwner()
        listing = supervise([str(binary), "--list"], owner, logs / "unfiltered-list.log",
                            deadline=DEADLINE, rss_cap=RSS_CAP)
        if owner.collect():
            raise UnitFailure("unfiltered-list child cleanup not proven")
        print(listing, end="", flush=True)
        names = [line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")]
        if len(names) != len(set(names)) or not set((DEMAND_TEST, TOKEN_TEST, *CASES)).issubset(names):
            raise UnitFailure("unfiltered current executable listing lacks required tests")
        output = execute(binary, DEMAND_TEST, logs / "demand.log")
        batch["actual_demand"] = validate_demand(output)
        batch["header_admitted"] = True
        print("MOSS_FIXTURE_DEMAND=" + json.dumps(batch["actual_demand"]), flush=True)
        for index, name in enumerate((TOKEN_TEST, *CASES)):
            if source_stamp() != compiler["source"] or file_digest(binary) != compiler["binary_sha256"]:
                raise UnitFailure("admitted source/executable changed before next independent case")
            log = logs / f"case-{index}.log"
            try:
                output = execute(binary, name, log)
                print(output, end="", flush=True)
            except CaseAssertionFailure:
                pass  # Only exact assertion failures with verified source and cleanup can continue.
            batch["cases"].append(json.loads(log.with_suffix(".result.json").read_text(encoding="utf-8")))
        batch["passed"] = all(row["status"] == "passed" for row in batch["cases"])
    except BaseException as error:
        batch["fatal_error"] = str(error)
        raise
    finally:
        try:
            batch["final_source"] = source_stamp()
            batch["source_restored"] = batch["final_source"] == compiler["source"]
        except (OSError, subprocess.SubprocessError) as error:
            batch["final_source"] = None
            batch["source_restored"] = None
            batch["source_check_error"] = str(error)
        if batch["source_restored"] is not True:
            batch["passed"] = False
            batch["fatal_error"] = batch["fatal_error"] or "final source identity is not proven unchanged"
        binary_file.with_suffix(".batch.json").write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    if batch["fatal_error"] is not None:
        raise UnitFailure(batch["fatal_error"])
    return batch


def main(argv: list[str] | None = None) -> int:
    arguments=sys.argv[1:] if argv is None else argv
    if arguments and arguments[0]=="--owned-exec":
        if len(arguments)<2 or sys.stdin.buffer.read(1)!=b"R":
            return 2
        os.execv(arguments[1],arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "run"))
    parser.add_argument("backend", choices=tuple(PACKAGES))
    parser.add_argument("--binary-file", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        if os.name != "posix" or (args.backend == "mlx" and sys.platform != "darwin"):
            raise UnitFailure("this CPU fixture profile requires its declared POSIX host")
        target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
        if args.command == "compile":
            stamp = source_stamp()
            result = subprocess.run(compile_argv(args.backend), cwd=ROOT, stdout=subprocess.PIPE,
                                    text=True, encoding="utf-8", timeout=2700, check=False)
            cargo_log = args.binary_file.with_suffix(".cargo.jsonl")
            cargo_log.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                render_compile_errors(result.stdout)
                raise UnitFailure(f"fixture compile exited {result.returncode}")
            binary = select_binary(result.stdout, PACKAGES[args.backend], target)
            if source_stamp() != stamp:
                raise UnitFailure("fixture source changed during compilation")
            args.binary_file.write_text(json.dumps({"backend": args.backend, "head": git_head(),
                                                   "source": stamp, "binary_sha256": file_digest(binary),
                                                   "cargo_json_sha256": file_digest(cargo_log),
                                                   "binary": str(binary), "target": str(target.resolve())}), encoding="utf-8")
        else:
            record = json.loads(args.binary_file.read_text(encoding="utf-8"))
            if record["backend"] != args.backend or record["head"] != git_head():
                raise UnitFailure("compiled fixture source/backend identity differs")
            binary = Path(record["binary"])
            if not binary.is_file() or Path(record["target"]) not in binary.resolve().parents:
                raise UnitFailure("compiled fixture executable missing or outside owned target")
            if record["source"] != source_stamp() or record["binary_sha256"] != file_digest(binary):
                raise UnitFailure("compiled fixture source or executable content changed")
            if record["cargo_json_sha256"] != file_digest(args.binary_file.with_suffix(".cargo.jsonl")):
                raise UnitFailure("compiled fixture Cargo artifact log changed")
            batch = run_cases(binary, args.backend, args.binary_file, record)
            if not batch["passed"]:
                print(f"moss-fixture-cpu: FAIL backend={args.backend}; independent assertion failures recorded", file=sys.stderr)
                return 1
            print(f"moss-fixture-cpu: PASS backend={args.backend}; CPU reference, not CUDA/Metal or real weights", flush=True)
    except (UnitFailure, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        print(f"moss-fixture-cpu: FAIL {error}", file=sys.stderr)
        return 2 if args.command == "run" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
