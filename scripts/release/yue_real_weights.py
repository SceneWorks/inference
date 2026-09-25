#!/usr/bin/env python3
"""Stage the inputs of YuE's CUDA real-weight smoke (sc-19387, `real-weights-yue.yml`).

`crates/audio/candle-audio-yue/tests/registered_loader_real_weights.rs` reads
`$YUE_SNAPSHOT_ROOT/<repo>/...` and `$YUE_REF_DIR/sceneworks-derived/pop.00001.f32le`.

``stage-root`` lays out ``YUE_SNAPSHOT_ROOT`` from hub-cache snapshot directories. It cannot use
junctions: a hub-cache snapshot's files are relative symlinks into ``../../blobs``, and Windows
resolves those against the path they were opened through, so behind a junction every file is
"not found". Each file is instead hard-linked to its resolved blob, which needs ``--root`` on the
cache's volume; across volumes each file is COPIED instead (tens of GB), and the printed per-repo
counts say which happened.

``stage-reference`` does only the decode step of ``scripts/reference/yue_icl_reference.py`` (which
needs the whole torch reference environment): fetch upstream's ``prompt_egs/pop.00001.mp3`` at the
pinned YuE commit, verify its SHA-256, decode it with ``soundfile`` at float32 exactly as the
producer does, and write the ``.f32le``. The decoded PCM's SHA-256 is compared with the committed
fixture's ``pcm_sha256`` and printed: a mismatch (a different libsndfile/mpg123 build) is reported,
not fatal, because the smoke asserts output shape and finiteness, not reference-exact ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "crates/audio/candle-audio-yue/tests/fixtures/yue_icl_reference.json"
YUE_COMMIT = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
CLIP = "pop.00001"
CLIP_URL = (
    "https://raw.githubusercontent.com/multimodal-art-projection/YuE/"
    f"{YUE_COMMIT}/prompt_egs/{CLIP}.mp3"
)
CLIP_SHA256 = "27760a9be58c03258d749f31d23e848ad17a85e22cb2b56594ee87df84fbe4b8"


def stage_root(root: Path, repos: list[str]) -> None:
    """`repos` are `NAME=SNAPSHOT_DIR`; each becomes `root/NAME` holding real files."""
    if root.exists():
        shutil.rmtree(root)
    for spec in repos:
        name, _, source = spec.partition("=")
        source_dir = Path(source)
        if not name or not source_dir.is_dir():
            raise SystemExit(f"bad --repo {spec!r}: expected NAME=<existing snapshot dir>")
        linked = copied = 0
        for directory, _, files in os.walk(source_dir):
            for file in files:
                src = Path(directory) / file
                dst = root / name / src.relative_to(source_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                blob = os.path.realpath(src)
                try:
                    os.link(blob, dst)
                    linked += 1
                except OSError:
                    shutil.copyfile(blob, dst)
                    copied += 1
        print(f"{name}: {linked} hard-linked, {copied} copied from {source_dir}")


def stage_reference(ref_dir: Path) -> None:
    with urllib.request.urlopen(CLIP_URL, timeout=120) as response:
        mp3 = response.read()
    actual = hashlib.sha256(mp3).hexdigest()
    if actual != CLIP_SHA256:
        raise SystemExit(f"{CLIP_URL}: sha256 {actual} != pinned {CLIP_SHA256}")
    raw_dir = ref_dir / "prompt_egs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = raw_dir / f"{CLIP}.mp3"
    mp3_path.write_bytes(mp3)

    import numpy as np
    import soundfile

    data, rate = soundfile.read(str(mp3_path), dtype="float32", always_2d=True)
    raw = np.ascontiguousarray(data).astype("<f4").tobytes()
    derived = ref_dir / "sceneworks-derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / f"{CLIP}.f32le").write_bytes(raw)

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["pop"]["clips"][CLIP]
    decoded = {
        "rate": rate,
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "pcm_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if (decoded["rate"], decoded["channels"]) != (expected["rate"], expected["channels"]):
        raise SystemExit(f"{CLIP}: decoded {decoded}, fixture expects {expected}")
    verdict = "matches" if decoded == expected else "DIFFERS from"
    print(f"{CLIP}: decoded {decoded} ({verdict} the committed fixture {expected})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    root = commands.add_parser("stage-root", help="lay out YUE_SNAPSHOT_ROOT")
    root.add_argument("--root", required=True, type=Path)
    root.add_argument("--repo", required=True, action="append", help="NAME=SNAPSHOT_DIR")
    reference = commands.add_parser("stage-reference", help="populate YUE_REF_DIR")
    reference.add_argument("--ref-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "stage-root":
        stage_root(args.root, args.repo)
    else:
        stage_reference(args.ref_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
