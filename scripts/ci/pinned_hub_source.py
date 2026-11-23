"""Authenticate local snapshot bytes against a public, immutable Hugging Face revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request


MIB = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * MIB
MAX_FILE_BYTES = 64 * 1024**3


def file_inventory_seal(files: list[dict]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_file_inventory_receipt(receipt: dict | None, model: dict) -> None:
    """Replay a frozen selected-file manifest without querying the Hub or reading weights."""
    identity = {key: model[key] for key in ("repository", "revision", "files_sha256")}
    if (not isinstance(receipt, dict) or set(receipt) != identity.keys() | {"files"}
            or any(receipt[key] != value for key, value in identity.items())):
        raise ValueError("terminal asset receipt differs from the frozen snapshot identity")
    files = receipt["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("terminal asset receipt lacks the exact file inventory")
    total = 0
    previous = ""
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "kind", "digest"}:
            raise ValueError("invalid terminal asset inventory row")
        name, size, kind, digest = (row[key] for key in ("path", "bytes", "kind", "digest"))
        if (not isinstance(name, str) or name <= previous or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or type(size) is not int or not 0 <= size <= MAX_FILE_BYTES
                or not isinstance(digest, str)
                or not ((kind == "git_blob_sha1" and re.fullmatch(r"[0-9a-f]{40}", digest))
                        or (kind == "lfs_sha256" and re.fullmatch(r"[0-9a-f]{64}", digest)))):
            raise ValueError("invalid, duplicate or unordered terminal asset inventory row")
        total += size
        previous = name
    if total != model["asset_bytes"] or file_inventory_seal(files) != identity["files_sha256"]:
        raise ValueError("terminal asset bytes or file digests differ from the frozen inventory")


def _digest(path: Path, algorithm: str, size: int) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {size}\0".encode())
    read = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * MIB), b""):
            read += len(block)
            if read > size:
                raise ValueError(f"local snapshot file grew during verification: {path}")
            digest.update(block)
    if read != size:
        raise ValueError(f"local snapshot file changed during verification: {path}")
    return digest.hexdigest()


def pinned_manifest(repository: str, revision: str) -> tuple[str, bytes, dict[str, dict]]:
    """The Hub's expanded file inventory of one public, ungated repository revision."""
    if (not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("source must name one repository at an immutable revision")
    url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "SceneWorks-model-proof/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Hub pinned manifest exceeds the read bound")
    document = json.loads(raw)
    if (document.get("id") != repository or document.get("sha") != revision
            or document.get("private") is not False or document.get("gated") is not False):
        raise ValueError("Hub readback does not authenticate the public pinned source")
    siblings = document.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("Hub readback lacks expanded sibling inventory")
    inventory = {}
    for sibling in siblings:
        name = sibling.get("rfilename") if isinstance(sibling, dict) else None
        if (not isinstance(name, str) or not name or name.startswith("/")
                or "\\" in name or any(part in ("", ".", "..") for part in name.split("/"))
                or name in inventory):
            raise ValueError("Hub readback has an unsafe or repeated sibling path")
        inventory[name] = sibling
    return url, raw, inventory


def pinned_file_identity(relative: str, sibling: dict | None) -> dict:
    """Validate the exact length and digest from one expanded immutable-revision row."""
    size = sibling.get("size") if sibling is not None else None
    if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
        raise ValueError(f"Hub pinned inventory size/path mismatch: {relative}")
    lfs = sibling.get("lfs")
    if lfs is not None:
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        if (not isinstance(lfs, dict) or lfs.get("size") != size
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"Hub pinned LFS metadata invalid: {relative}")
        kind = "lfs_sha256"
    else:
        digest = sibling.get("blobId")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise ValueError(f"Hub pinned Git blob metadata invalid: {relative}")
        kind = "git_blob_sha1"
    return {"path": relative, "bytes": size, "kind": kind, "digest": digest}


def verify_pinned_hub_files(repository: str, revision: str, snapshot: Path,
                            relative_files: tuple[str, ...]) -> dict:
    """Return upstream-manifest and byte receipts; download no model payload."""
    if not relative_files or len(set(relative_files)) != len(relative_files):
        raise ValueError("source verification needs unique required files")
    url, raw, inventory = pinned_manifest(repository, revision)
    verified = []
    for relative in relative_files:
        if (not isinstance(relative, str) or not relative or relative.startswith("/")
                or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise ValueError("required snapshot file path is unsafe")
        file = snapshot / relative
        expected = pinned_file_identity(relative, inventory.get(relative))
        size = expected["bytes"]
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"Hub pinned inventory size/path mismatch: {relative}")
        algorithm = "sha256" if expected["kind"] == "lfs_sha256" else "sha1"
        if _digest(file, algorithm, size) != expected["digest"]:
            raise ValueError(f"Hub pinned content mismatch: {relative}")
        verified.append(expected)
    return {"status": "hub_content_verified", "repository": repository, "revision": revision,
            "readback_url": url, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": verified}
