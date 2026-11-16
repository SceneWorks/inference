#!/usr/bin/env python3
"""Fetch a verified FlashAttention archive for this source tree, or build from source."""

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

from flash_attn_prebuilt import CRATE, LICENSES, source_sha256


NAMES = {"libflashattention.a", "flash-attn-prebuilt.txt", *LICENSES}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(directory, source_hash, target):
    if {p.name for p in directory.iterdir()} != NAMES:
        raise ValueError("FlashAttention package has missing or unexpected files")
    if any(not p.is_file() or p.is_symlink() for p in directory.iterdir()):
        raise ValueError("FlashAttention package contains a non-file")
    fields = {}
    for line in (directory / "flash-attn-prebuilt.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("invalid FlashAttention manifest")
        fields[key] = value
    if fields.get("format") != "1" or fields.get("source_sha256") != source_hash or fields.get("target") != target:
        raise ValueError("FlashAttention manifest source or target mismatch")
    if digest_file(directory / "libflashattention.a") != fields.get("archive_sha256"):
        raise ValueError("FlashAttention archive checksum mismatch")


def fetch(target, cache):
    if target not in ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"):
        return None
    source_hash = source_sha256(CRATE)
    destination = cache / source_hash / target
    if destination.exists():
        validate(destination, source_hash, target)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = f"flash-attn-{target}.tar.gz"
    url = f"https://github.com/SceneWorks/inference/releases/download/flash-attn-{source_hash}/{asset}"
    with tempfile.TemporaryDirectory(prefix="flash-attn-", dir=destination.parent) as temporary_text:
        temporary = pathlib.Path(temporary_text)
        archive = temporary / asset
        try:
            with urllib.request.urlopen(url, timeout=300) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        unpacked = temporary / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(NAMES) or {m.name for m in members} != NAMES or any(not m.isfile() for m in members):
                raise ValueError("FlashAttention package has unsafe archive entries")
            for member in members:
                with bundle.extractfile(member) as source, (unpacked / member.name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        validate(unpacked, source_hash, target)
        try:
            unpacked.rename(destination)
        except FileExistsError:
            validate(destination, source_hash, target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/sceneworks/flash-attn")
    parser.add_argument("--github-env", action="store_true")
    args, command = parser.parse_known_args()
    if command and command[0] != "--":
        parser.error("command must follow --")
    directory = fetch(args.target, args.cache)
    if directory:
        print(f"FlashAttention prebuilt: {directory}", file=sys.stderr)
        if args.github_env:
            with pathlib.Path(os.environ["GITHUB_ENV"]).open("a") as output:
                output.write(f"CANDLE_FLASH_ATTN_PREBUILT_DIR={directory}\n")
    else:
        print("FlashAttention prebuilt unavailable; compiling from source", file=sys.stderr)
    if command:
        env = os.environ.copy()
        if directory:
            env["CANDLE_FLASH_ATTN_PREBUILT_DIR"] = str(directory)
        return subprocess.call(command[1:], env=env)
    return 0 if directory else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as error:
        print(f"FlashAttention prebuilt: {error}", file=sys.stderr)
        sys.exit(2)
