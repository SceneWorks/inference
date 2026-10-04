"""Record the archives/source Cargo actually linked, instead of a remembered MLX version."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def collect_identity(messages, lock):
    dependency = next(p for p in lock["package"] if p["name"] == "pmetal-mlx-sys")
    revision = dependency["source"].rsplit("#", 1)[1]
    records = [m for m in messages if m.get("reason") == "build-script-executed"
               and "pmetal-mlx-sys" in m.get("package_id", "")
               and revision in m["package_id"]]
    if len(records) != 1:
        raise ValueError("expected exactly one locked mlx-sys build identity")
    record = records[0]
    libraries = [Path(p.removeprefix("native=")) for p in record["linked_paths"]]
    libraries = [p for p in libraries if (p / "libmlx.a").is_file() and (p / "libmlxc.a").is_file()]
    if len(libraries) != 1:
        raise ValueError("expected exactly one actual MLX library directory")
    directory = libraries[0]
    manifest = directory / "pmetal-mlx-prebuilt.txt"
    build_manifest = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
    if not build_manifest.get("fingerprint"):
        raise ValueError("linked MLX archives lack their source fingerprint")
    staged = Path(record["out_dir"]) / "mlx-c-staged" / "CMakeLists.txt"
    tag = None
    if staged.is_file():
        match = re.search(r"GIT_TAG\s+(v[0-9.]+)", staged.read_text())
        if match is None or match[1] != "v0.32.0":
            raise ValueError("actual staged MLX source must select v0.32.0")
        tag = match[1]
    return {
        "mlxSysPackageId": record["package_id"], "lockedMlxRsRevision": revision,
        "expectedCoreTag": "v0.32.0", "actualStagedCoreTag": tag,
        "sourceIdentity": file_identity(staged) if staged.is_file() else None,
        "buildManifest": build_manifest, "buildManifestIdentity": file_identity(manifest),
        "archives": [file_identity(directory / name) for name in ("libmlx.a", "libmlxc.a")],
        "metallib": file_identity(directory / "mlx.metallib") if (directory / "mlx.metallib").is_file() else None,
        "linkMode": "source_with_staged_tag" if tag else "prebuilt_verified_by_mlx_sys_source_fingerprint",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("Cargo.lock"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    messages = []
    for line in args.messages.read_text().splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # cargo stderr diagnostics are preserved in the same log
    identity = collect_identity(messages, tomllib.loads(args.lock.read_text()))
    args.out.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
