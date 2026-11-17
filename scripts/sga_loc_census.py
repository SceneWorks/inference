#!/usr/bin/env python3
"""Report tracked source/test physical lines by area at immutable revisions."""
import argparse
import json
from pathlib import Path
import subprocess


# Group first-party executable source, including shaders and the checked-in tooling.
# Keep area boundaries stable so additions and deletions are visible between runs.
AREAS = {
    "inference": ("crates/media/mlx-gen", "crates/media/candle-gen", "crates/llm",
                  "crates/audio", "crates/kernels", "crates/contracts", "crates/bundles",
                  "ios-host", "scripts", "tests"),
    "SceneWorks": ("crates/sceneworks-worker", "crates/sceneworks-core",
                   "crates/sceneworks-memory-adapter", "crates/sceneworks-image-quality",
                   "crates/sceneworks-mcp", "apps/rust-api", "apps/rust-worker",
                   "apps/web", "apps/desktop", "packages", "scripts", "tests"),
}
SUFFIXES = set(".rs .py .sh .ps1 .cmd .bat .mjs .cjs .js .jsx .ts .tsx .css .scss .html "
               ".swift .c .cc .cpp .h .hpp .m .mm .metal .cu .cuh .wgsl .glsl".split())


def bucket(path):
    if (set(path.parts) & {"tests", "test", "__tests__", "fixtures", "testdata"}
            or path.stem.startswith("test_")
            or path.stem.endswith(("_test", "_tests", ".test", ".spec"))):
        return "test"
    return "source"


def census(name, root, revision):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    commit = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    areas = AREAS[name]
    totals = {area: {kind: {"files": 0, "lines": 0} for kind in ("source", "test")}
              for area in areas}
    # ls-tree is the git ls-files population of the named commit. Read committed
    # blobs so a dirty working tree or the caller's current branch cannot taint it.
    entries = git("ls-tree", "-rz", commit, "--", *areas).split(b"\0")
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as blobs:
        for entry in filter(None, entries):
            header, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = header.split()
            path = Path(raw_path.decode())
            if (kind != b"blob" or mode == b"120000" or path.suffix.lower() not in SUFFIXES
                    or set(path.parts) & {"vendor", "node_modules"}):
                continue
            blobs.stdin.write(oid + b"\n")
            blobs.stdin.flush()
            size = int(blobs.stdout.readline().split()[2])
            data = blobs.stdout.read(size)
            if blobs.stdout.read(1) != b"\n":
                raise ValueError("truncated Git blob stream")
            area = next(area for area in areas if path.is_relative_to(area))
            count = totals[area][bucket(path)]
            count["files"] += 1
            count["lines"] += data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))
        blobs.stdin.close()
        if blobs.wait():
            raise RuntimeError("git cat-file failed")
    return {"commit": commit, "areas": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inference", "sceneworks"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-rev", required=True)
    args = parser.parse_args()
    print(json.dumps({
        "inference": census("inference", args.inference, args.inference_rev),
        "SceneWorks": census("SceneWorks", args.sceneworks, args.sceneworks_rev),
    }, indent=2))


if __name__ == "__main__":
    main()
