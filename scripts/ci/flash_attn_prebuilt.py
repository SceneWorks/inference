#!/usr/bin/env python3
"""Package the exact vendored FlashAttention archive for a per-revision release.

This only handles build output. It never launches a CUDA kernel or model.
"""

import argparse
import hashlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tarfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates/media/candle-gen/vendor/candle-flash-attn"
NOTICES = ("LICENSE-MIT", "LICENSE-APACHE", "LICENSE-FLASH-ATTENTION", "LICENSE-CUTLASS.txt", "AUTHORS-FLASH-ATTENTION")


def source_sha256(crate=CRATE):
    digest = hashlib.sha256()
    files = [crate / "build.rs", crate / "Cargo.toml", *(crate / "kernels").iterdir(),
             *(crate / name for name in NOTICES)]
    for path in sorted(files, key=lambda path: path.relative_to(crate).as_posix()):
        if not path.is_file():
            continue
        digest.update(path.relative_to(crate).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cuda_version():
    nvcc = os.environ.get("NVCC")
    if not nvcc:
        for key in ("CUDA_HOME", "CUDA_PATH"):
            if key in os.environ:
                candidate = pathlib.Path(os.environ[key]) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
                if candidate.is_file():
                    nvcc = str(candidate)
                    break
    output = subprocess.check_output([nvcc or "nvcc", "--version"], text=True)
    match = re.search(r"release ([0-9]+\.[0-9]+)", output)
    if not match:
        raise ValueError("cannot determine nvcc release")
    return match.group(1)


def host_compiler_version(target):
    if target.endswith("-msvc"):
        result = subprocess.run(["cl"], capture_output=True, text=True, check=False)
        match = re.search(r"Version ([0-9.]+)", result.stdout + result.stderr)
        if not match:
            raise ValueError("cannot determine MSVC version; run vcvars64 first")
        return match.group(1)
    return subprocess.check_output(["g++", "-dumpfullversion"], text=True).strip().split()[0]


def manifest_text(crate, archive, target, revision):
    if target not in ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"):
        raise ValueError("unsupported prebuilt target")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an exact commit")
    values = {
        "format": "1",
        "inference_revision": revision,
        "source_sha256": source_sha256(crate),
        "archive_sha256": file_sha256(archive),
        "target": target,
        "cuda_compute_cap": "80",
        "cuda_version": cuda_version(),
        "host_compiler_version": host_compiler_version(target),
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())


def pack(crate, directory, output, target, revision):
    archive = directory / "libflashattention.a"
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError("source-built libflashattention.a is missing or empty")
    manifest = manifest_text(crate, archive, target, revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as bundle:
        bundle.add(archive, arcname=archive.name)
        data = manifest.encode()
        info = tarfile.TarInfo("flash-attn-prebuilt.txt")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
        for name in NOTICES:
            bundle.add(crate / name, arcname=name)
    print(f"packaged {output} ({archive.stat().st_size} archive bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    pack(CRATE, args.directory, args.output, args.target, args.revision)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"flash-attn-prebuilt: {error}", file=sys.stderr)
        sys.exit(1)
