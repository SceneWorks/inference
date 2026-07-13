#!/usr/bin/env python3
"""Verify release checksums, manifest/SBOM consistency, and source consumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

try:
    from scripts.release.build_release import validate_tag
except ModuleNotFoundError:  # Direct `python scripts/release/verify_release.py` invocation.
    from build_release import validate_tag


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(bundle: Path) -> None:
    checksums = bundle / "SHA256SUMS"
    if not checksums.is_file():
        raise RuntimeError("release bundle has no SHA256SUMS")
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, separator, name = line.partition("  ")
        if not separator or Path(name).name != name:
            raise RuntimeError(f"invalid checksum entry: {line!r}")
        artifact = bundle / name
        if not artifact.is_file():
            raise RuntimeError(f"checksum target is absent: {name}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {actual} != {expected}")


def verify_sbom(bundle: Path, manifest: dict) -> None:
    sbom_artifacts = [
        artifact for artifact in manifest["artifacts"] if artifact["kind"] == "spdx-2.3-json"
    ]
    if len(sbom_artifacts) != 1:
        raise RuntimeError("manifest must contain exactly one SPDX JSON artifact")
    sbom = json.loads((bundle / sbom_artifacts[0]["name"]).read_text(encoding="utf-8"))
    if sbom.get("spdxVersion") != "SPDX-2.3":
        raise RuntimeError("SBOM is not SPDX 2.3")
    packages = sbom.get("packages", [])
    if len(packages) != manifest["lockfile"]["package_count"]:
        raise RuntimeError("SBOM package count does not match runtime manifest")
    identifiers = {"SPDXRef-DOCUMENT", *(package["SPDXID"] for package in packages)}
    for relationship in sbom.get("relationships", []):
        if relationship["spdxElementId"] not in identifiers:
            raise RuntimeError(f"unknown relationship source: {relationship['spdxElementId']}")
        if relationship["relatedSpdxElement"] not in identifiers:
            raise RuntimeError(f"unknown relationship target: {relationship['relatedSpdxElement']}")


def safe_members(archive: tarfile.TarFile, expected_prefix: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    expected_root = expected_prefix.rstrip("/")
    for member in members:
        path = PurePosixPath(member.name)
        in_release_root = member.name == expected_root or member.name.startswith(expected_prefix)
        if path.is_absolute() or ".." in path.parts or not in_release_root:
            raise RuntimeError(f"unsafe or unexpected source path: {member.name}")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or ".." in link.parts:
                raise RuntimeError(f"unsafe source link: {member.name} -> {member.linkname}")
    return members


def smoke_source_archive(bundle: Path, manifest: dict, offline: bool) -> None:
    source_artifacts = [
        artifact for artifact in manifest["artifacts"] if artifact["kind"] == "source-tar-gzip"
    ]
    if len(source_artifacts) != 1:
        raise RuntimeError("manifest must contain exactly one source archive")
    tag = manifest["release"]["tag"]
    prefix = f"inference-{tag}/"
    with tempfile.TemporaryDirectory(prefix="inference-release-smoke-") as temporary:
        temp = Path(temporary)
        with tarfile.open(bundle / source_artifacts[0]["name"], "r:gz") as archive:
            members = safe_members(archive, prefix)
            archive.extractall(temp, members=members)
        source_root = temp / prefix.rstrip("/")
        consumer = temp / "consumer"
        (consumer / "src").mkdir(parents=True)
        dependency = (source_root / "crates/media/mlx-gen/gen-core").as_posix()
        (consumer / "Cargo.toml").write_text(
            "[package]\n"
            'name = "inference-release-smoke"\n'
            'version = "0.0.0"\n'
            'edition = "2021"\n\n'
            "[dependencies]\n"
            f'sceneworks-gen-core = {{ path = "{dependency}" }}\n',
            encoding="utf-8",
        )
        (consumer / "src/main.rs").write_text(
            "fn main() {\n"
            '    println!("gen={} llm={}", gen_core::VERSION, gen_core::core_llm::VERSION);\n'
            "}\n",
            encoding="utf-8",
        )
        command = ["cargo", "check", "--manifest-path", str(consumer / "Cargo.toml")]
        if offline:
            command.append("--offline")
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(temp / "target")
        subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()

    verify_checksums(bundle)
    manifest = json.loads((bundle / "runtime-manifest.json").read_text(encoding="utf-8"))
    validate_tag(manifest["release"]["tag"])
    if manifest["release"]["dirty"] and not args.allow_dirty:
        raise RuntimeError("release manifest records dirty inputs")
    if manifest["workspace"]["package_count"] != 69:
        raise RuntimeError("release manifest does not contain the expected 69 workspace packages")
    verify_sbom(bundle, manifest)
    if not args.skip_smoke:
        smoke_source_archive(bundle, manifest, args.offline)
    print("release verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
