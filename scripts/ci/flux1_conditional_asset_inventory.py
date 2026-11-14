#!/usr/bin/env python3
"""Manual, CPU-only inventory of installed FLUX.1 conditional proof assets.

No model load or inference takes place here. The output is a candidate manifest for operator
review; byte digests, not cache revision labels, are authoritative. The one network action is
the opt-in `--stage-pinned-assets` setup, which fetches exactly the product-pinned files the
inventory reports missing, off-pin or incomplete (stdlib HTTPS, no Hub client), never replacing
an existing file.

Discovery never lists a directory. On nax-macos-2, `os.scandir` blocked forever on the Hub root
(run 42463079723) and on `models--SceneWorks--flux1-dev-mlx/snapshots` (run 42474515680), while
`stat` probes returned in milliseconds. Each role therefore names the exact repository, file and
revision the SceneWorks product ships for that FLUX.1 route; the inventory builds the
`snapshots/<revision>/<file>` path and stats it in a child process with a per-call timeout. A
timeout is reported as `unreachable`, never as absent. Other repositories that hold a file of
the same name are ignored, so a role cannot be ambiguous.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.flux1_pair.conditional import CASES, require_pid_gemma_footprint
from scripts.ci.flux1_pair.fs_probe import Unreachable, probe

RUNNERS = {"nax-macos-2", "cuda-windows", "cuda-windows-2"}
LANES = {"nax-macos-2": ["self-hosted", "macOS", "ARM64", "rw-llm"],
         "cuda-windows": ["self-hosted", "Windows", "X64", "cuda", "real-weights", "cuda-windows"],
         "cuda-windows-2": ["self-hosted", "Windows", "X64", "cuda", "real-weights", "cuda-windows-2"]}
MAX_HASH_BYTES = 120 * 1024**3
MAX_SECONDS = 1_200
MAX_METADATA_SECONDS = 300
MAX_REF_BYTES = 256
SHA = re.compile(r"[0-9a-f]{40}")

# The exact artifact the SceneWorks product ships for each FLUX.1 conditional role, at its pinned
# revision: SceneWorks `crates/sceneworks-core/src/control_weights.rs` (SHIPPED_CONTROL_WEIGHTS
# `flux1_dev_control`; SHIPPED_AUXILIARY_SOURCES flux_ip_adapter, flux_ip_encoder, pulid_adapter,
# pulid_eva, pulid_bisenet, pulid_scrfd, pulid_arcface) and `config/manifests/registry/specs.json`
# (`pid_flux` downloads and resources.pidDecoders.flux). The base FLUX.1 weights are not inventoried:
# the comparison harness uses the frozen synthetic base fixture.
PRODUCT_SOURCES = {
    "control": ("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
                "diffusion_pytorch_model.safetensors", "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5"),
    "ip_adapter": ("XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors",
                   "18f6940238ab5dc3744df7a8e30315892279d5f9"),
    "image_encoder": ("openai/clip-vit-large-patch14", "model.safetensors",
                      "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    "pulid": ("guozinan/PuLID", "pulid_flux_v0.9.1.safetensors",
              "492b1451255dc9d9bc3c857259690b5f8b998d4a"),
    "eva": ("SceneWorks/pulid-flux-mlx", "eva02_clip_l_336.safetensors",
            "78ef91f977eae16d66fb191caf003154b7a0a0b8"),
    "bisenet": ("SceneWorks/pulid-flux-mlx", "bisenet_parsing.safetensors",
                "78ef91f977eae16d66fb191caf003154b7a0a0b8"),
    "scrfd": ("SceneWorks/instantid-mlx", "scrfd_10g.safetensors",
              "bca0cacf8e5e04529bb2b326a521361b02be84fd"),
    "arcface": ("SceneWorks/instantid-mlx", "arcface_iresnet100.safetensors",
                "bca0cacf8e5e04529bb2b326a521361b02be84fd"),
    "pid_decoder": ("SceneWorks/pid-flux", "pid_flux_2kto4k.safetensors",
                    "6d5c1f1049e863f1757f68fd81c6bc850a95609d"),
    # A directory role: its files are probed by the exact names below, never listed.
    "pid_gemma": ("SceneWorks/gemma-2-2b-it", None, "684c553b5b41a1c835989d89f62f585e6269a7de"),
}
# Every file name a Gemma 2 2B snapshot may carry; indexed shards come from the index itself.
GEMMA_FILES = (".gitattributes", "LICENSE", "NOTICE", "PROHIBITED_USE_POLICY.md", "README.md",
               "config.json", "generation_config.json", "special_tokens_map.json",
               "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "gemma-2-2b-it.safetensors", "model.safetensors", "model.safetensors.index.json")
MAX_GEMMA_INDEX_BYTES = 4 * 1024 * 1024
MAX_GEMMA_SHARDS = 64

def progress(message):
    print("FLUX1_ASSET_INVENTORY " + message, file=sys.stderr, flush=True)


class Budget:
    def __init__(self, max_seconds=None):
        self.started = time.monotonic()
        self.bytes = 0
        self.last_report = self.started
        self.stage = "discovery"
        self.max_seconds = MAX_SECONDS if max_seconds is None else max_seconds

    def check(self, extra=0):
        now = time.monotonic()
        if now - self.started > self.max_seconds or self.bytes + extra > MAX_HASH_BYTES:
            raise ValueError(f"asset inventory time/byte budget exceeded at {self.stage}; "
                             f"elapsed_seconds={int(now - self.started)} hash_bytes={self.bytes}")
        if now - self.last_report >= 60:
            progress(f"elapsed_seconds={int(now - self.started)} hash_bytes={self.bytes} stage={self.stage}")
            self.last_report = now


def cache_dir(hub, repo):
    return hub / ("models--" + repo.replace("/", "--"))


def confined(resolved, hub, path):
    resolved = Path(resolved)
    if not resolved.is_relative_to(hub):
        raise ValueError("asset candidate escapes configured model hub: " + str(path))
    return resolved


def digest(path, resolved, hub, budget, size):
    budget.stage = "hash " + str(path.relative_to(hub))
    budget.check(size)
    sha = hashlib.sha256()
    with Path(resolved).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            budget.check(len(chunk))
            sha.update(chunk)
            budget.bytes += len(chunk)
    if budget.bytes and Path(resolved).stat().st_size != size:
        raise ValueError("asset candidate changed during inventory: " + str(path))
    return sha.hexdigest()


def stat_file(path, hub, budget):
    budget.check()
    row = probe(path)
    if not row["exists"]:
        return None
    if not row["is_file"]:
        raise ValueError("asset candidate is not a file: " + str(path))
    return confined(row["resolved"], hub, path), row["size"]


def file_row(path, hub, budget, metadata_only, revision, provenance):
    found = stat_file(path, hub, budget)
    if found is None:
        return None
    resolved, size = found
    row = {"path": str(path), "resolved": str(resolved), "revision": revision,
           "revision_provenance": provenance, "bytes": size}
    if not metadata_only:
        row["sha256"] = digest(path, resolved, hub, budget, size)
    return row


def gemma_row(snapshot, hub, budget, metadata_only, revision, provenance):
    budget.check()
    directory = probe(snapshot)
    if not directory["exists"]:
        return None
    if not directory["is_dir"]:
        raise ValueError("PiD Gemma candidate is not a directory")
    names = list(GEMMA_FILES)
    index = probe(snapshot / "model.safetensors.index.json", op="read", limit=MAX_GEMMA_INDEX_BYTES)
    if index["exists"] and index.get("is_file"):
        if len(index["text"].encode("utf-8")) > MAX_GEMMA_INDEX_BYTES:
            raise ValueError("PiD Gemma shard index exceeds byte limit")
        try:
            shards = sorted(set(json.loads(index["text"])["weight_map"].values()))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError("PiD Gemma shard index is invalid") from error
        if len(shards) > MAX_GEMMA_SHARDS or not all(
                isinstance(name, str) and re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
                for name in shards):
            raise ValueError("PiD Gemma shard index is invalid")
        names += shards
    files = {}
    for name in names:
        found = stat_file(snapshot / name, hub, budget)
        if found is None:
            continue
        resolved, size = found
        files[name] = size if metadata_only else digest(snapshot / name, resolved, hub, budget, size)
    row = {"path": str(snapshot), "resolved": str(confined(directory["resolved"], hub, snapshot)),
           "revision": revision, "revision_provenance": provenance,
           "file_bytes" if metadata_only else "files": files}
    try:
        require_pid_gemma_footprint(snapshot, files)
        row["complete"] = True
    except ValueError as error:
        row["complete"] = False
        row["incomplete_reason"] = str(error)
    return row


def role_inventory(role, hub, budget, metadata_only):
    """Select the product-pinned artifact; report refs/main only as an off-pin diagnostic."""
    repo, name, pinned = PRODUCT_SOURCES[role]
    progress(f"role={role} repo={repo}")
    entry = {"repo": repo, "file": name, "pinned_revision": pinned}
    base = cache_dir(hub, repo)
    collect = gemma_row if name is None else file_row

    def candidate(revision, provenance):
        target = base / "snapshots" / revision
        return collect(target if name is None else target / name, hub, budget, metadata_only,
                       revision, provenance)

    try:
        if pinned is not None:
            row = candidate(pinned, "product-manifest-pin")
            if row is not None:
                status = "incomplete" if name is None and not row["complete"] else "unique"
                return {**entry, "status": status, "candidates": [row]}
        ref = probe(base / "refs" / "main", op="read", limit=MAX_REF_BYTES)
        revision = ref.get("text", "").strip() if ref["exists"] and ref.get("is_file") else ""
        if not SHA.fullmatch(revision) or revision == pinned:
            return {**entry, "status": "missing", "candidates": []}
        row = candidate(revision, "huggingface-refs-main")
        if row is None:
            return {**entry, "status": "missing", "candidates": []}
        if pinned is None:
            status = "incomplete" if name is None and not row["complete"] else "unique"
        else:
            status = "off_pin"
        return {**entry, "status": status, "candidates": [row]}
    except Unreachable as error:
        progress(f"role={role} unreachable: {error}")
        return {**entry, "status": "unreachable", "candidates": [], "unreachable": str(error)}


def inventory(hub, runner, metadata_only=False):
    if runner not in RUNNERS:
        raise ValueError("runner is not an approved conditional proof host")
    progress(f"resolve_hub mode={'metadata' if metadata_only else 'hash'}")
    hub = Path(hub).expanduser().resolve(strict=True)
    if not hub.is_dir():
        raise ValueError("configured model hub is not a directory")
    budget = Budget(MAX_METADATA_SECONDS if metadata_only else None)
    backend = "mlx" if runner == "nax-macos-2" else "candle-cuda"
    needed = sorted({role for _, backends, roles in CASES.values() if backend in backends
                     for role in roles})
    roles = {role: role_inventory(role, hub, budget, metadata_only) for role in needed}
    cases = {}
    for case, (_, backends, required) in CASES.items():
        if backend in backends:
            cases[case] = {"ready": not metadata_only and all(roles[role]["status"] == "unique"
                                                              for role in required),
                           "roles": list(required)}

    def with_status(status):
        return sorted(role for role, row in roles.items() if row["status"] == status)

    return {"schema": 2, "protocol": "flux1-conditional-asset-inventory-v2", "runner": runner,
            "backend": backend, "hub": str(hub), "mode": "metadata" if metadata_only else "hash",
            "hashes_complete": not metadata_only, "hash_bytes": budget.bytes,
            "selection_rule": "product-manifest repository/file/revision only; other repositories ignored",
            "revision_note": "snapshot path is a provenance label; SHA-256 is authoritative",
            "missing_roles": with_status("missing"), "off_pin_roles": with_status("off_pin"),
            "unreachable_roles": with_status("unreachable"),
            "incomplete_roles": with_status("incomplete"),
            "roles": roles, "cases": cases}


HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"
STAGE_STATUSES = {"missing", "off_pin", "incomplete"}
# Files staged for the PiD Gemma directory role; indexed shards are read from the staged index.
GEMMA_STAGE_FILES = ("config.json", "generation_config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                     "model.safetensors.index.json")
STAGE_CHUNK = 8 * 1024 * 1024
STAGE_SOCKET_TIMEOUT = 120
MAX_STAGE_SECONDS = 3_600


def pinned_path(hub, repo, revision, name):
    """`models--<org>--<name>/snapshots/<revision>/<file>` under the Hub root."""
    if not SHA.fullmatch(revision) or not name or "\\" in name or name.startswith("/") \
            or ".." in name.split("/"):
        raise ValueError(f"unsafe pinned asset path: {repo}@{revision}/{name}")
    return cache_dir(hub, repo) / "snapshots" / revision / name


class _RecordLinkedEtag(urllib.request.HTTPRedirectHandler):
    """Remember the Hub's `X-Linked-Etag` (the LFS SHA-256) from the resolve redirect."""

    def __init__(self):
        self.linked_etag = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.linked_etag = self.linked_etag or headers.get("X-Linked-Etag")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_pinned(url, destination, token=None, opener=None):
    """Stream `url` to a temporary file beside `destination`, then place it without overwriting.

    Returns (bytes, sha256, linked_etag). The temporary file is this process's own; an existing
    destination is never replaced (`os.link` on POSIX, `os.rename` on Windows both refuse it).
    """
    if probe(destination)["exists"]:
        raise FileExistsError(f"refusing to overwrite existing asset {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "flux1-asset-stager/1"})
    if token:
        # Unredirected: the presigned CDN URL the Hub redirects to never receives the token.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    recorder = _RecordLinkedEtag()
    opener = opener or urllib.request.build_opener(recorder)
    descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".",
                                             suffix=".partial", dir=destination.parent)
    temporary = Path(temporary)
    digest_state, size = hashlib.sha256(), 0
    # An HTTP request, not a file: bound by name so the text-mode file-open audit does not read it
    # as a locale-decoded `Path.open`. The body is consumed as bytes.
    request_url = opener.open
    try:
        with os.fdopen(descriptor, "wb") as sink, request_url(request, timeout=STAGE_SOCKET_TIMEOUT) as source:
            expected = source.headers.get("Content-Length")
            while chunk := source.read(STAGE_CHUNK):
                sink.write(chunk)
                digest_state.update(chunk)
                size += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if expected is not None and int(expected) != size:
            raise ValueError(f"short download for {url}: {size} of {expected} bytes")
        sha = digest_state.hexdigest()
        etag = (getattr(opener, "linked_etag", None) or recorder.linked_etag or "").strip('"')
        if re.fullmatch(r"[0-9a-f]{64}", etag) and etag != sha:
            raise ValueError(f"downloaded bytes for {url} differ from the Hub's LFS SHA-256")
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        return size, sha, etag or None
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_files(role, hub):
    """The exact files to stage for one role: its pinned file, or the PiD Gemma footprint."""
    repo, name, revision = PRODUCT_SOURCES[role]
    if name is not None:
        return [(repo, revision, name)]
    return [(repo, revision, file) for file in GEMMA_STAGE_FILES]


def stage_pinned_assets(hub, runner, token=None, fetch=fetch_pinned):
    """Fetch every product-pinned file the inventory reports missing, off-pin or incomplete.

    Existing files are kept and reported, never replaced; `refs/main` is never written; no
    directory is listed. Returns one receipt row per pinned file considered.
    """
    hub = Path(hub).expanduser().resolve(strict=True)
    before = inventory(hub, runner, metadata_only=True)
    receipts = []
    for role, row in sorted(before["roles"].items()):
        if row["status"] not in STAGE_STATUSES:
            continue
        pending = stage_files(role, hub)
        while pending:
            repo, revision, file = pending.pop(0)
            destination = pinned_path(hub, repo, revision, file)
            url = HF_RESOLVE.format(repo=repo, revision=revision,
                                    name=urllib.parse.quote(file))
            receipt = {"role": role, "repo": repo, "revision": revision, "file": file,
                       "path": str(destination)}
            if probe(destination)["exists"]:
                receipt["action"] = "kept-existing"
            else:
                progress(f"stage role={role} {repo}@{revision}/{file}")
                size, sha, etag = fetch(url, destination, token=token)
                receipt.update(action="fetched", bytes=size, sha256=sha,
                               hub_lfs_sha256_verified=etag is not None)
            receipts.append(receipt)
            if role == "pid_gemma" and file == "model.safetensors.index.json":
                index = probe(destination, op="read", limit=MAX_GEMMA_INDEX_BYTES)
                try:
                    shards = sorted(set(json.loads(index["text"])["weight_map"].values()))
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    raise ValueError("staged PiD Gemma shard index is invalid") from error
                if len(shards) > MAX_GEMMA_SHARDS or not all(
                        isinstance(shard, str)
                        and re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", shard)
                        for shard in shards):
                    raise ValueError("staged PiD Gemma shard index is invalid")
                pending.extend((repo, revision, shard) for shard in shards)
    return receipts


def plan_request(backend, request, expected_sha, tier):
    if (type(request) is not dict or not {"selection", "runner"} <= set(request)
            or set(request) - {"selection", "runner", "mode"}
            or request["selection"] != "flux1-asset-inventory"):
        raise ValueError("inventory needs exact selection, runner and optional mode")
    mode = request.get("mode", "hash")
    if mode not in {"hash", "metadata"}:
        raise ValueError("inventory mode must be hash or metadata")
    runner = request["runner"]
    if runner not in RUNNERS or backend != ("mlx" if runner == "nax-macos-2" else "candle-cuda"):
        raise ValueError("inventory backend and selected runner differ")
    if tier != "none":
        raise ValueError("asset inventory requires flux1_pair=none")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                   encoding="utf-8").strip()
    if expected_sha != head:
        raise ValueError("inventory source SHA differs from checked-out branch")
    return {"runner": LANES[runner], "asset_setup": "flux1-asset-inventory", "selected_runner": runner,
            "backend": backend, "source_sha": head, "mode": mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--stage-pinned-assets", action=argparse.BooleanOptionalAction, default=False,
                        help="before the inventory, fetch every product-pinned file it reports "
                             "missing, off-pin or incomplete into FLUX1_HUB (stdlib HTTPS only)")
    parser.add_argument("--backend", choices=("mlx", "candle-cuda"), required=True)
    parser.add_argument("--request-env", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.plan:
        progress("cli_start")
        faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    try:
        request = json.loads(os.environ[args.request_env])
        if not args.plan and hasattr(signal, "SIGALRM"):
            def interrupt(_signum, _frame):
                raise TimeoutError("asset inventory hard deadline reached")
            signal.signal(signal.SIGALRM, interrupt)
            mode = request.get("mode") if isinstance(request, dict) else None
            signal.alarm((MAX_METADATA_SECONDS if mode == "metadata" else MAX_SECONDS) + 60
                         + (MAX_STAGE_SECONDS if args.stage_pinned_assets else 0))
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
    progress(f"start runner={plan['selected_runner']} mode={plan['mode']} "
             f"stage_pinned_assets={args.stage_pinned_assets}")
    try:
        staged = (stage_pinned_assets(root, plan["selected_runner"],
                                      token=os.environ.get("HF_TOKEN") or None)
                  if args.stage_pinned_assets else None)
        result = inventory(root, plan["selected_runner"], metadata_only=plan["mode"] == "metadata")
        result["staging"] = {"requested": args.stage_pinned_assets, "files": staged or []}
    finally:
        faulthandler.cancel_dump_traceback_later()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    result["source_sha"] = plan["source_sha"]
    result["runner"] = plan["selected_runner"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
