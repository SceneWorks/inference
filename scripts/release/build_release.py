#!/usr/bin/env python3
"""Build a deterministic runtime manifest, SPDX SBOM, and source archive."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote


TAG_PATTERN = re.compile(r"^runtime-(\d{4})\.(\d{2})\.(\d+)(?:-rc\.(\d+))?$")
REPOSITORY = "https://github.com/SceneWorks/inference"
TOOL_NAME = "SceneWorks inference release builder"


def run(*args: str, cwd: Path, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        list(args), check=True, cwd=cwd, capture_output=True, text=not binary
    )
    return result.stdout


def validate_tag(tag: str) -> re.Match[str]:
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(
            "release tag must match runtime-YYYY.MM.patch or "
            "runtime-YYYY.MM.patch-rc.N"
        )
    month = int(match.group(2))
    if month not in range(1, 13):
        raise ValueError("release tag month must be between 01 and 12")
    return match


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spdx_id(name: str, version: str, source: str | None) -> str:
    identity = f"{name}\0{version}\0{source or 'workspace'}".encode()
    suffix = hashlib.sha256(identity).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9.-]", "-", name)
    return f"SPDXRef-Package-{safe_name}-{suffix}"


def source_download_location(package: dict[str, Any]) -> str:
    source = package.get("source")
    if not source:
        return "NOASSERTION"
    if source.startswith("registry+"):
        return (
            f"https://crates.io/api/v1/crates/{quote(package['name'])}/"
            f"{quote(package['version'])}/download"
        )
    if source.startswith("git+"):
        return source.removeprefix("git+").split("#", 1)[0]
    return "NOASSERTION"


def package_key(package: dict[str, Any]) -> tuple[str, str, str | None]:
    return package["name"], package["version"], package.get("source")


def resolve_lock_dependency(
    dependency: str, packages_by_name: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    match = re.fullmatch(r"(\S+)(?: (\S+)(?: \((.+)\))?)?", dependency)
    if not match:
        raise RuntimeError(f"cannot parse Cargo.lock dependency {dependency!r}")
    name, version, source = match.groups()
    candidates = packages_by_name.get(name, [])
    if version:
        candidates = [package for package in candidates if package["version"] == version]
    if source:
        candidates = [package for package in candidates if package.get("source") == source]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cargo.lock dependency {dependency!r} resolved to {len(candidates)} packages"
        )
    return candidates[0]


def build_spdx(
    *,
    tag: str,
    revision: str,
    created: str,
    metadata: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    metadata_packages = {
        (package["name"], package["version"], package.get("source")): package
        for package in metadata["packages"]
    }
    packages = sorted(
        lock.get("package", []),
        key=lambda package: (
            package["name"],
            package["version"],
            package.get("source") or "",
        ),
    )
    metadata_ids = {
        package["id"]: spdx_id(*package_key(package)) for package in metadata["packages"]
    }

    spdx_packages: list[dict[str, Any]] = []
    for package in packages:
        key = (package["name"], package["version"], package.get("source"))
        metadata_package = metadata_packages.get(key, {})
        item: dict[str, Any] = {
            "name": package["name"],
            "SPDXID": spdx_id(*key),
            "versionInfo": package["version"],
            "downloadLocation": source_download_location(package),
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": metadata_package.get("license") or "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
        checksum = package.get("checksum")
        if checksum:
            item["checksums"] = [{"algorithm": "SHA256", "checksumValue": checksum}]
        spdx_packages.append(item)

    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": metadata_ids[member],
        }
        for member in sorted(metadata["workspace_members"])
    ]
    packages_by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        packages_by_name.setdefault(package["name"], []).append(package)
    for package in packages:
        for dependency in sorted(package.get("dependencies", [])):
            resolved = resolve_lock_dependency(dependency, packages_by_name)
            relationships.append(
                {
                    "spdxElementId": spdx_id(*package_key(package)),
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": spdx_id(*package_key(resolved)),
                }
            )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"SceneWorks inference {tag}",
        "documentNamespace": f"{REPOSITORY}/spdx/{tag}/{revision}",
        "creationInfo": {"created": created, "creators": [f"Tool: {TOOL_NAME}"]},
        "packages": spdx_packages,
        "relationships": relationships,
    }


def workspace_packages(metadata: dict[str, Any], root: Path) -> list[dict[str, str]]:
    members = set(metadata["workspace_members"])
    packages = []
    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        manifest = Path(package["manifest_path"])
        packages.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package.get("license") or "NOASSERTION",
                "manifest": manifest.relative_to(root).as_posix(),
            }
        )
    return sorted(packages, key=lambda package: package["name"])


def backend_pin(metadata: dict[str, Any], names: set[str]) -> dict[str, Any]:
    matches = [package for package in metadata["packages"] if package["name"] in names]
    sources = sorted({package.get("source") for package in matches if package.get("source")})
    if len(sources) != 1:
        raise RuntimeError(f"expected one source for {sorted(names)}, found {sources}")
    source = sources[0]
    revision = source.rsplit("#", 1)[1] if "#" in source else None
    return {
        "packages": sorted(package["name"] for package in matches),
        "source": source,
        "revision": revision,
    }


def gzip_archive(tar_content: bytes, timestamp: int) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=timestamp) as archive:
        archive.write(tar_content)
    return output.getvalue()


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_release(
    *,
    root: Path,
    output: Path,
    tag: str,
    source_ref: str,
    offline: bool,
    allow_dirty: bool,
    skip_archive: bool,
) -> dict[str, Any]:
    match = validate_tag(tag)
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    status = str(run("git", "status", "--porcelain", "--untracked-files=no", cwd=root))
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise RuntimeError("release inputs are dirty; commit them or pass --allow-dirty")

    revision = str(run("git", "rev-parse", f"{source_ref}^{{commit}}", cwd=root)).strip()
    timestamp = int(str(run("git", "show", "-s", "--format=%ct", revision, cwd=root)).strip())
    created = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    metadata_command = ["cargo", "metadata", "--locked", "--format-version", "1"]
    if offline:
        metadata_command.append("--offline")
    metadata = json.loads(str(run(*metadata_command, cwd=root)))
    lock_path = root / "Cargo.lock"
    lock_bytes = lock_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    toolchain = tomllib.loads((root / "rust-toolchain.toml").read_text(encoding="utf-8"))

    sbom_name = f"{tag}.spdx.json"
    sbom_path = output / sbom_name
    spdx = build_spdx(
        tag=tag, revision=revision, created=created, metadata=metadata, lock=lock
    )
    write_json(sbom_path, spdx)

    artifacts = [
        {
            "name": sbom_name,
            "kind": "spdx-2.3-json",
            "sha256": sha256_file(sbom_path),
            "bytes": sbom_path.stat().st_size,
        }
    ]
    if not skip_archive:
        archive_name = f"{tag}.tar.gz"
        archive_path = output / archive_name
        tar_content = run(
            "git",
            "archive",
            "--format=tar",
            f"--prefix=inference-{tag}/",
            revision,
            cwd=root,
            binary=True,
        )
        assert isinstance(tar_content, bytes)
        archive_path.write_bytes(gzip_archive(tar_content, timestamp))
        artifacts.insert(
            0,
            {
                "name": archive_name,
                "kind": "source-tar-gzip",
                "sha256": sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
            },
        )

    packages = workspace_packages(metadata, root)
    manifest = {
        "schema_version": 1,
        "release": {
            "tag": tag,
            "candidate": match.group(4) is not None,
            "repository": REPOSITORY,
            "revision": revision,
            "source_date": created,
            "dirty": dirty,
        },
        "toolchain": {
            "rust": toolchain["toolchain"]["channel"],
            "profile": toolchain["toolchain"].get("profile"),
            "components": sorted(toolchain["toolchain"].get("components", [])),
        },
        "lockfile": {
            "version": lock["version"],
            "sha256": sha256_bytes(lock_bytes),
            "package_count": len(lock.get("package", [])),
        },
        "workspace": {"package_count": len(packages), "packages": packages},
        "backends": {
            "mlx": backend_pin(metadata, {"pmetal-mlx-rs", "pmetal-mlx-sys"}),
            "candle": backend_pin(
                metadata,
                {"candle-core", "candle-nn", "candle-transformers", "candle-flash-attn"},
            ),
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "runtime-manifest.json"
    write_json(manifest_path, manifest)

    checksum_paths = [manifest_path, *[output / artifact["name"] for artifact in artifacts]]
    checksums = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(checksum_paths)
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("dist/release"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = build_release(
        root=root,
        output=args.output,
        tag=args.tag,
        source_ref=args.source_ref,
        offline=args.offline,
        allow_dirty=args.allow_dirty,
        skip_archive=args.skip_archive,
    )
    print(
        f"release bundle: {manifest['release']['tag']} "
        f"({manifest['workspace']['package_count']} workspace packages, "
        f"{manifest['lockfile']['package_count']} locked packages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
