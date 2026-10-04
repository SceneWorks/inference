"""Materialize immutable adapter transfer assets; never create or publish a release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.request


def validate_manifest(manifest):
    if manifest.get("repository") != "SceneWorks/inference":
        raise ValueError("adapter assets must come from SceneWorks/inference")
    if not isinstance(manifest.get("releaseId"), int) or manifest["releaseId"] <= 0:
        raise ValueError("a positive releaseId is required")
    entries = manifest.get("adapters", [])
    names, files = set(), set()
    if not entries:
        raise ValueError("adapter manifest is empty")
    for entry in entries:
        if not re.fullmatch(r"[a-z0-9_]+", entry.get("name", "")):
            raise ValueError("invalid adapter name")
        if entry["name"] in names or entry.get("file") in files:
            raise ValueError("duplicate adapter name or file")
        names.add(entry["name"])
        files.add(entry.get("file"))
        receipt = (manifest.get("purpose") == "DIAGNOSTIC_ONLY"
                   and entry.get("name") == "training_receipt"
                   and entry.get("kind") == "diagnostic_training_receipt")
        suffix = "json" if receipt else "safetensors"
        if not re.fullmatch(rf"[A-Za-z0-9_-]+\.{suffix}", entry.get("file", "")):
            raise ValueError("adapter file must be one safe safetensors basename")
        if entry.get("kind") not in ("lora", "lokr") and not receipt:
            raise ValueError("adapter kind must be lora or lokr")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError("adapter SHA-256 is required")
        if not isinstance(entry.get("assetId"), int) or entry["assetId"] <= 0:
            raise ValueError("a positive assetId is required")
        if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
            raise ValueError("expected byte length is required")
        if not re.fullmatch(rf"[A-Za-z0-9_-]+\.{suffix}", entry.get("assetName", "")):
            raise ValueError("expected release asset name is required")
    return entries


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(manifest, destination, opener=urllib.request.urlopen):
    entries = validate_manifest(manifest)
    headers = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    release_request = urllib.request.Request(
        f"https://api.github.com/repos/{manifest['repository']}/releases/{manifest['releaseId']}",
        headers=headers,
    )
    with opener(release_request) as response:
        release = json.load(response)
    assets = {asset["id"]: asset for asset in release["assets"]}
    for entry in entries:
        asset = assets.get(entry["assetId"])
        if asset is None or asset["name"] != entry["assetName"] or asset["size"] != entry["size"]:
            raise ValueError(f"release does not own the expected asset {entry['name']}")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = destination / entry["file"]
        if path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]:
            continue
        request = urllib.request.Request(
            f"https://api.github.com/repos/{manifest['repository']}/releases/assets/{entry['assetId']}",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/octet-stream",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        temporary = path.with_suffix(".download")
        try:
            with opener(request) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if temporary.stat().st_size != entry["size"] or sha256(temporary) != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {entry['name']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    resolved = {**manifest, "directory": str(destination.resolve())}
    output = destination / "adapter-imports-resolved.json"
    output.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(json.loads(args.manifest.read_text(encoding="utf-8")), args.destination))


if __name__ == "__main__":
    main()
