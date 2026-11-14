#!/usr/bin/env python3
"""Link one identical catalog consumer against both product release partitions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


TARGET = "aarch64-apple-ios"
BIN = "sga_ios_catalog_size"
ENTRY_POINT = """fn main() {
    println!("{:?}", runtime_ios::catalog().unwrap().snapshot());
}
"""


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def measure(media):
    features = ["--no-default-features"] + (["--features", "media"] if media else [])
    common = ["--locked", "--target", TARGET, "-p", "runtime-ios", *features]
    # Normal edges only: the test-only ios-catalog must not skew the product graph.
    tree = output("cargo", "tree", "-e", "normal", "--prefix", "none", *common)
    packages = {line.split()[0] for line in tree.splitlines()}
    if ("mlx-gen-ios-catalog" in packages) != media or "mlx-gen-catalog" in packages:
        raise ValueError("unexpected product catalog partition")
    subprocess.run(["cargo", "build", *common, "--release", "--bin", BIN], check=True)
    binary = Path(os.environ["CARGO_TARGET_DIR"]) / TARGET / "release" / BIN
    arches = output("xcrun", "lipo", "-archs", str(binary))
    load_command = output("xcrun", "vtool", "-show-build", str(binary))
    if arches != "arm64" or "platform IOS" not in load_command:
        raise ValueError("expected a physical iOS arm64 Mach-O")
    return {"bytes": binary.stat().st_size, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "features": features, "architecture": arches, "load_command": load_command,
            "catalog_packages": sorted(p for p in packages if "catalog" in p)}


def main():
    if len(sys.argv) != 2 or output("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("run in a clean source checkout with one receipt output path")
    source = output("git", "rev-parse", "HEAD")
    fixture = Path("crates/bundles/runtime-ios/src/bin") / (BIN + ".rs")
    fixture.parent.mkdir(exist_ok=True)
    # Register the same temporary binary in the measured package on old and new
    # revisions. Avoid example/dev edges (runtime-ios has a media dev dependency).
    with fixture.open("x", encoding="utf-8") as f:
        f.write(ENTRY_POINT)
    try:
        partitions = {name: measure(media) for name, media in (("llm-only", False), ("media", True))}
    finally:
        fixture.unlink()
    receipt = {"source": source, "measurement_revision": os.environ.get("GITHUB_SHA"),
               "entry_point": ENTRY_POINT, "target": TARGET, "profile": "release",
               "rustc": output("rustc", "-vV"), "xcode": output("xcodebuild", "-version"),
               "iphoneos_sdk": output("xcrun", "--sdk", "iphoneos", "--show-sdk-version"),
               "macos": output("sw_vers"), "macos_deployment_target": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
               "release_strip": os.environ.get("CARGO_PROFILE_RELEASE_STRIP"),
               "partitions": partitions}
    Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
