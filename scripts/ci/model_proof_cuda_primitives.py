"""Private exact published-App lease/ordinal primitives, NOT approval or memory-cap authority.

Only selected stdlib-only function bodies are copied unchanged from ba8/4d0b44.
Caller validates explicit approved roles/policies and composes the inference bounded supervisor.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

GUARD_SHA = "e4a000279a5da909dd9c074916dfe2bcdcf7751c849b0dad3080778f4b281a36"

def component(name: str, expected: str):
    file = Path(__file__).resolve().parent / "vendor" / name
    if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"pinned CUDA ownership component changed: {name}")
    spec = importlib.util.spec_from_file_location("nvfp4_" + file.stem, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unlock(handle, locker):
    try:
        handle.seek(0)
        locker.locking(handle.fileno(), locker.LK_UNLCK, 1)
    finally:
        handle.close()


def release_all(handles, locker):
    first_error = None
    for handle in reversed(handles):
        try:
            unlock(handle, locker)
        except BaseException as error:
            first_error = first_error or error
    if first_error is not None:
        raise first_error


def required_mib(required_free_bytes: int) -> int:
    if type(required_free_bytes) is not int or required_free_bytes <= 0:
        raise RuntimeError("positive per-UUID free-memory requirement is required")
    return -(-required_free_bytes // 1024**2)


WAIT_NOTICE_SECONDS = 60


def wait_notice(state: dict, reason: str, clock=time.monotonic) -> None:
    """Print the current refusal reason at most once per WAIT_NOTICE_SECONDS; never a silent wait."""
    now = clock()
    if now - state.get("last", float("-inf")) >= WAIT_NOTICE_SECONDS:
        state["last"] = now
        print(f"CUDA admission waiting: {reason}", file=sys.stderr, flush=True)


def shortfall(guard, device, required: int) -> str:
    if not guard.counters_valid(device):
        return f'{device["uuid"]}: invalid memory counters'
    return f'{device["uuid"]}: {device["free_mib"]} MiB free < {required} MiB required'


def claim_uuids(uuids, guard, out, deadline, public, locker, required_free_bytes,
                required_indices=None, phase_names=None, clock=time.monotonic, sleep=time.sleep):
    """Lock every UUID, then admit when each keeps its required free memory on a quiet card.

    Foreign contexts never refuse; the PUBLIC UUID lock is the CI-to-CI mutex and is released
    on every refusal so other CI jobs can use the card while this one waits.
    """
    expected = sorted(uuids, key=str.casefold)
    if not expected or len({value.casefold() for value in expected}) != len(expected):
        raise RuntimeError("explicit unique CUDA UUID set is required")
    needed = {identifier.casefold(): required_mib(required_free_bytes[identifier]) for identifier in expected}
    notice = {}
    if not public.is_absolute():
        raise RuntimeError("PUBLIC must name the host-global absolute lock namespace")
    started = clock()
    while clock() < deadline:
        devices, processes = guard.inventory(out, "uuid-admission", deadline)
        selected = []
        for identifier in expected:
            matches = [row for row in devices if row["uuid"].casefold() == identifier.casefold()]
            if len(matches) != 1:
                raise RuntimeError("declared GPU UUID changed or disappeared; no device fallback")
            device = matches[0]
            if required_indices and device["index"] != required_indices[identifier]:
                raise RuntimeError("declared physical GPU index/UUID binding changed")
            if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device["uuid"]):
                raise RuntimeError("unrecognized physical GPU UUID")
            selected.append(device)
        handles = []
        busy = False
        directory = public / "SceneWorks" / "gpu-locks"
        directory.mkdir(parents=True, exist_ok=True)
        handle = None
        try:
            for device in selected:
                handle = (directory / f'{device["uuid"]}.lock').open("a+b")
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    locker.locking(handle.fileno(), locker.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    busy = True
                    break
                handles.append(handle)
        except BaseException:
            if handle is not None and handle not in handles:
                handle.close()
            release_all(handles, locker)
            raise
        if busy:
            release_all(handles, locker)
            wait_notice(notice, "a selected UUID lock is held by another CI job", clock)
            sleep(min(1, max(0, deadline - clock())))
            continue
        # Judged after every lock is taken, but on the inventory read just before locking;
        # sample_headroom below re-reads each card under the lock before admission.
        short = [shortfall(guard, device, needed[device["uuid"].casefold()]) for device in selected
                 if not guard.has_headroom(device, needed[device["uuid"].casefold()])]
        if short:
            release_all(handles, locker)
            wait_notice(notice, "; ".join(short), clock)
            sleep(min(1, max(0, deadline - clock())))
            continue
        try:
            baselines = {}
            checked = []
            for device in selected:
                phase = (phase_names or {}).get(device["uuid"], f'claimed-device-{device["index"]}')
                current, baseline = guard.sample_headroom(device, out, phase, needed[device["uuid"].casefold()])
                checked.append(current)
                baselines[device["uuid"]] = baseline
            return checked, handles, baselines, clock() - started
        except RuntimeError as error:
            release_all(handles, locker)
            wait_notice(notice, f"card not stable across samples: {error}", clock)
            sleep(min(1, max(0, deadline - clock())))
        except BaseException:
            release_all(handles, locker)
            raise
    raise TimeoutError("exact GPU UUID lock/headroom wait exhausted the existing execution budget")


def integers(value):
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value or ""):
        raise RuntimeError("explicit nonnegative CUDA ordinal/index list is required")
    result = [int(part) for part in value.split(",")]
    if len(set(result)) != len(result):
        raise RuntimeError("duplicate CUDA selection")
    return result


def resolve_binding(guard, out, deadline, environment, physical_indices=None,
                    cuda_ordinals="0", probe_visible=None, run=subprocess.run):
    devices, _ = guard.inventory(out, "binding", deadline)
    if physical_indices is not None:
        selected = []
        for index in integers(physical_indices):
            matches = [row for row in devices if row["index"] == index]
            if len(matches) != 1:
                raise RuntimeError("actor's explicit NVML-to-UUID binding is unavailable")
            selected.append(matches[0])
        return {"kind": "actor-forced-nvml-uuid", "devices": selected,
                "key": [(row["index"], row["uuid"]) for row in selected]}
    probe_env = environment.copy()
    if probe_visible is not None:
        probe_env["CUDA_VISIBLE_DEVICES"] = probe_visible
    remaining = min(30, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("CUDA metadata mapping deadline exhausted")
    result = run([sys.executable, "-I", str(Path(__file__).with_name("windows_cuda_metadata.py"))],
                 env=probe_env, capture_output=True, text=True, timeout=remaining, check=True)
    packet = json.loads(result.stdout)
    if (packet.get("context_created") is not False
            or packet.get("abi_source", {}).get("checksum") != "1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc"):
        raise RuntimeError("unverified context-free CUDA ordinal metadata")
    visible = packet.get("devices", [])
    ordinals = [row["ordinal"] for row in visible] if cuda_ordinals == "all" else integers(cuda_ordinals)
    if not ordinals:
        raise RuntimeError("no declared CUDA-visible device; never skip/pass")
    selected = []
    mapping = []
    for ordinal in ordinals:
        logical = [row for row in visible if row["ordinal"] == ordinal]
        if len(logical) != 1:
            raise RuntimeError("declared actual CUDA ordinal is unavailable")
        matches = [row for row in devices if row["uuid"].casefold() == logical[0]["uuid"].casefold()]
        if len(matches) != 1:
            raise RuntimeError("actual CUDA ordinal UUID cannot be matched to NVML")
        selected.append(matches[0])
        mapping.append({"cuda_ordinal": ordinal, "uuid": matches[0]["uuid"], "nvml_index": matches[0]["index"]})
    if len({row["uuid"].casefold() for row in selected}) != len(selected):
        raise RuntimeError("actual CUDA ordinal mapping is ambiguous")
    return {"kind": "context-free-cuda-ordinal-uuid", "devices": selected,
            "probe": packet, "mapping": mapping,
            "key": [(row["cuda_ordinal"], row["uuid"]) for row in mapping]}
