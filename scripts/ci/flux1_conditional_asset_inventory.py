#!/usr/bin/env python3
"""Manual, CPU-only inventory of installed FLUX.1 conditional proof assets.

No download, model load, or inference takes place here. The output is a candidate
manifest for operator review; byte digests, not cache revision labels, are authoritative.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.flux1_pair.conditional import CASES, require_pid_gemma_footprint

RUNNERS = {"nax-macos-2", "cuda-windows", "cuda-windows-2"}
LANES = {"nax-macos-2": ["self-hosted", "macOS", "ARM64", "rw-llm"],
         "cuda-windows": ["self-hosted", "Windows", "X64", "cuda", "real-weights", "cuda-windows"],
         "cuda-windows-2": ["self-hosted", "Windows", "X64", "cuda", "real-weights", "cuda-windows-2"]}
MAX_ENTRIES = 250_000
MAX_CANDIDATES_PER_ROLE = 16
MAX_GEMMA_FILES = 5_000
MAX_HASH_BYTES = 120 * 1024**3
MAX_SECONDS = 1_200
SHA = re.compile(r"[0-9a-f]{40}")
SOURCE_MARKERS = ("flux.1", "flux1", "flux-ip", "pid-flux", "pulid", "controlnet-union-pro",
                  "clip-vit-large-patch14", "gemma-2-2b-it", "eva", "face", "antelope",
                  "facexlib", "insightface", "instantid")
LOOSE_ROOTS = {"flux1", "flux.1", "pulid", "pid", "face", "converted", "conditional-assets"}
GEMMA_REPOS = ("models--google--gemma-2-2b-it", "models--efficient-large-model--gemma-2-2b-it",
               "models--sceneworks--gemma-2-2b-it")


def progress(message):
    print("FLUX1_ASSET_INVENTORY " + message, file=sys.stderr, flush=True)


def role_for(path):
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]
    if name == "diffusion_pytorch_model.safetensors" and any("controlnet" in p for p in parts):
        return "control"
    if name == "control.safetensors" and any("flux" in p or "shakker" in p for p in parts):
        return "control"
    if name == "ip_adapter.safetensors" and any("flux" in p or "xlabs" in p for p in parts):
        return "ip_adapter"
    if name == "model.safetensors" and any("clip-vit-large-patch14" in p for p in parts):
        return "image_encoder"
    if name == "model.safetensors" and "image_encoder" in parts and any(
            "flux-ip-adapter" in p or "xlabs" in p for p in parts):
        return "image_encoder"
    if name in {"pulid_flux_v0.9.1.safetensors", "pulid.safetensors"}:
        return "pulid"
    if name.endswith(".safetensors") and name.startswith("eva"):
        return "eva"
    if name == "scrfd_10g.safetensors":
        return "scrfd"
    if name == "arcface_iresnet100.safetensors":
        return "arcface"
    if name == "bisenet_parsing.safetensors":
        return "bisenet"
    if name in {"flux_2k.safetensors", "pid_decoder.safetensors"}:
        return "pid_decoder"
    return None


def revision(path):
    parts = path.parts
    for index, part in enumerate(parts[:-2]):
        if part.startswith("models--") and parts[index + 1] == "snapshots" and SHA.fullmatch(parts[index + 2]):
            return parts[index + 2], "huggingface-cache-snapshot-path"
    return "local-unpinned", "local-path-label-only"


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.bytes = 0
        self.last_report = self.started
        self.stage = "discovery"

    def check(self, extra=0):
        now = time.monotonic()
        if now - self.started > MAX_SECONDS or self.bytes + extra > MAX_HASH_BYTES:
            raise ValueError(f"asset inventory time/byte budget exceeded at {self.stage}; "
                             f"elapsed_seconds={int(now - self.started)} hash_bytes={self.bytes}")
        if now - self.last_report >= 60:
            progress(f"elapsed_seconds={int(now - self.started)} hash_bytes={self.bytes} stage={self.stage}")
            self.last_report = now


def confined(path, hub):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(hub):
        raise ValueError("asset candidate escapes configured model hub: " + str(path))
    return resolved


def digest(path, hub, budget):
    budget.stage = "hash " + str(path.relative_to(hub))
    budget.check()
    resolved = confined(path, hub)
    if not resolved.is_file():
        raise ValueError("asset candidate is not a file: " + str(path))
    before = resolved.stat()
    budget.check(before.st_size)
    sha = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            budget.check(len(chunk))
            sha.update(chunk)
            budget.bytes += len(chunk)
    after = resolved.stat()
    if confined(path, hub) != resolved or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("asset candidate changed during inventory: " + str(path))
    return sha.hexdigest(), before.st_size


def gemma_files(directory, hub, budget):
    resolved = confined(directory, hub)
    if not resolved.is_dir():
        raise ValueError("PiD Gemma candidate is not a directory")
    files = {}
    def fail_walk(error):
        raise error

    for base, dirs, names in os.walk(directory, followlinks=False, onerror=fail_walk):
        dirs.sort()
        names.sort()
        for name in dirs[:]:
            if (Path(base) / name).is_symlink():
                raise ValueError("PiD Gemma has a symlink directory")
        for name in names:
            path = Path(base) / name
            relative = path.relative_to(directory).as_posix()
            if len(files) >= MAX_GEMMA_FILES:
                raise ValueError("PiD Gemma file budget exceeded")
            files[relative] = digest(path, hub, budget)[0]
    return files


def search_roots(hub, budget):
    """Inspect only FLUX.1-related HF snapshots and named loose asset directories.

    A Hub cache stores payloads under opaque `blobs/` names. Snapshot symlinks expose the
    candidate filenames, so walking every blob and unrelated model tree adds no discovery.
    """
    roots = []
    immediate = sorted(hub.iterdir())
    if len(immediate) > MAX_ENTRIES:
        raise ValueError("asset inventory entry budget exceeded")
    for child in immediate:
        budget.check()
        name = child.name.lower()
        if name.startswith("models--") and any(marker in name for marker in SOURCE_MARKERS):
            if child.is_symlink():
                raise ValueError("selected model cache directory is a symlink: " + str(child))
            snapshots = child / "snapshots"
            if snapshots.is_symlink():
                raise ValueError("selected snapshot directory is a symlink: " + str(snapshots))
            if snapshots.is_dir():
                roots.append(snapshots)
        elif name in LOOSE_ROOTS or name.startswith(("flux1-", "flux.1-", "pulid-", "pid-", "face-", "converted-")):
            if child.is_symlink():
                raise ValueError("selected loose asset directory is a symlink: " + str(child))
            if child.is_dir():
                roots.append(child)
    progress(f"selected_roots={len(roots)} hub_entries={len(immediate)}")
    return roots, immediate


def candidates(hub, budget):
    found = {role: [] for _, _, roles in CASES.values() for role in roles}
    roots, immediate = search_roots(hub, budget)
    entries = len(immediate)
    def fail_walk(error):
        raise error

    for path in immediate:
        if path.is_file():
            role = role_for(path)
            if role:
                found[role].append(path)
    for root in roots:
        budget.stage = "scan " + str(root.relative_to(hub))
        progress(f"scanning={root.relative_to(hub)}")
        if root.name == "snapshots" and root.parent.name.lower() in GEMMA_REPOS:
            for snapshot in sorted(root.iterdir()):
                budget.check()
                if snapshot.is_dir() and SHA.fullmatch(snapshot.name):
                    found["pid_gemma"].append(snapshot)
        for base, dirs, names in os.walk(root, followlinks=False, onerror=fail_walk):
            dirs.sort()
            names.sort()
            for name in dirs[:]:
                if (Path(base) / name).is_symlink():
                    dirs.remove(name)
            entries += len(dirs) + len(names)
            if entries > MAX_ENTRIES:
                raise ValueError("asset inventory entry budget exceeded")
            budget.check()
            for name in names:
                path = Path(base) / name
                role = role_for(path)
                if role:
                    found[role].append(path)
                    if len(found[role]) > MAX_CANDIDATES_PER_ROLE:
                        raise ValueError("too many candidates for role: " + role)
    if len(found["pid_gemma"]) > MAX_CANDIDATES_PER_ROLE:
        raise ValueError("too many candidates for role: pid_gemma")
    progress(f"discovered_entries={entries} candidates={sum(len(paths) for paths in found.values())}")
    return found, entries, roots


def inventory(hub, runner):
    if runner not in RUNNERS:
        raise ValueError("runner is not an approved conditional proof host")
    hub = Path(hub).expanduser().resolve(strict=True)
    if not hub.is_dir():
        raise ValueError("configured model hub is not a directory")
    budget = Budget()
    found, entries, roots = candidates(hub, budget)
    roles = {}
    for role, paths in sorted(found.items()):
        rows = []
        for path in paths:
            progress(f"role={role} candidate={path.relative_to(hub)}")
            resolved = confined(path, hub)
            rev, provenance = revision(path)
            row = {"path": str(path), "resolved": str(resolved), "revision": rev,
                   "revision_provenance": provenance}
            if role == "pid_gemma":
                row["files"] = gemma_files(path, hub, budget)
                try:
                    require_pid_gemma_footprint(path, row["files"])
                    row["complete"] = True
                except ValueError as error:
                    row["complete"] = False
                    row["incomplete_reason"] = str(error)
            else:
                row["sha256"], row["bytes"] = digest(path, hub, budget)
            rows.append(row)
        roles[role] = {"status": "missing" if not rows else "ambiguous" if len(rows) > 1 else
                       "incomplete" if role == "pid_gemma" and not rows[0]["complete"] else "unique",
                       "candidates": rows}
    cases = {}
    backend = "mlx" if runner == "nax-macos-2" else "candle-cuda"
    for case, (_, backends, required) in CASES.items():
        if backend in backends:
            cases[case] = {"ready": all(roles[role]["status"] == "unique" for role in required),
                           "roles": list(required)}
    return {"schema": 1, "protocol": "flux1-conditional-asset-inventory-v1", "runner": runner,
            "backend": backend, "hub": str(hub), "entries_scanned": entries,
            "search_roots": [str(root.relative_to(hub)) for root in roots],
            "hash_bytes": budget.bytes, "revision_note": "snapshot path is a provenance label; SHA-256 is authoritative",
            "missing_roles": sorted(role for role, row in roles.items() if row["status"] == "missing"),
            "ambiguous_roles": sorted(role for role, row in roles.items() if row["status"] == "ambiguous"),
            "incomplete_roles": sorted(role for role, row in roles.items() if row["status"] == "incomplete"),
            "roles": roles, "cases": cases}


def plan_request(backend, request, expected_sha, tier):
    if type(request) is not dict or set(request) != {"selection", "runner"} or request["selection"] != "flux1-asset-inventory":
        raise ValueError("inventory needs exact selection and runner")
    runner = request["runner"]
    if runner not in RUNNERS or backend != ("mlx" if runner == "nax-macos-2" else "candle-cuda"):
        raise ValueError("inventory backend and selected runner differ")
    if tier != "none":
        raise ValueError("asset inventory requires flux1_pair=none")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if expected_sha != head:
        raise ValueError("inventory source SHA differs from checked-out branch")
    return {"runner": LANES[runner], "asset_setup": "flux1-asset-inventory", "selected_runner": runner,
            "backend": backend, "source_sha": head}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--backend", choices=("mlx", "candle-cuda"), required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(os.environ[args.request_env])
        plan = plan_request(args.backend, request, args.expected_sha, os.environ.get("FLUX1_PAIR_TIER", "none"))
    except (KeyError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required for inventory")
    if os.environ.get("RUNNER_NAME") != plan["selected_runner"]:
        parser.error("selected runner differs from actual runner")
    root = os.environ.get("FLUX1_HUB")
    if not root:
        parser.error("FLUX1_HUB is required")
    result = inventory(root, plan["selected_runner"])
    result["source_sha"] = plan["source_sha"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
