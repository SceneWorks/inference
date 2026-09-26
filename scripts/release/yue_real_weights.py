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
``--with-stems`` also stages the song's ``Vocals`` / ``Instrumental`` stems (SHA-256-pinned, each
with a ``<clip>.json`` rate/channels sidecar) for ``tests/cuda_memory_real_weights.rs``.

``summarize-memory`` folds the memory mode's per-case JSONs and nvidia-smi CSVs into
``memory-summary.json`` / ``memory-summary.md`` in the evidence directory.
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
CLIP_SHA256 = "27760a9be58c03258d749f31d23e848ad17a85e22cb2b56594ee87df84fbe4b8"
# The same song's separated stems, which `tests/cuda_memory_real_weights.rs` loops into the dual
# ICL references of the memory-measurement cases (`stage-reference --with-stems`).
STEM_SHA256 = {
    f"{CLIP}.Vocals": "bcc751ffb3d9cb281eeb219f8eb78b7340a5e6d2cd0dfb687d081325c8f01c88",
    f"{CLIP}.Instrumental": "fba52a706c76c5e2ed30203a922c3a5ece82f4b9fa41e4d32f492b3898bff81b",
}


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


def fetch_and_decode(ref_dir: Path, clip: str, sha256: str):
    """Fetch `prompt_egs/<clip>.mp3` at the pinned commit, verify it, decode it to `.f32le`."""
    url = (
        "https://raw.githubusercontent.com/multimodal-art-projection/YuE/"
        f"{YUE_COMMIT}/prompt_egs/{clip}.mp3"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        mp3 = response.read()
    actual = hashlib.sha256(mp3).hexdigest()
    if actual != sha256:
        raise SystemExit(f"{url}: sha256 {actual} != pinned {sha256}")
    raw_dir = ref_dir / "prompt_egs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = raw_dir / f"{clip}.mp3"
    mp3_path.write_bytes(mp3)

    import numpy as np
    import soundfile

    data, rate = soundfile.read(str(mp3_path), dtype="float32", always_2d=True)
    raw = np.ascontiguousarray(data).astype("<f4").tobytes()
    derived = ref_dir / "sceneworks-derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / f"{clip}.f32le").write_bytes(raw)
    return data, rate, raw


def stage_reference(ref_dir: Path, with_stems: bool = False) -> None:
    data, rate, raw = fetch_and_decode(ref_dir, CLIP, CLIP_SHA256)

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
    if not with_stems:
        return
    for clip, sha256 in STEM_SHA256.items():
        data, rate, raw = fetch_and_decode(ref_dir, clip, sha256)
        meta = {"rate": rate, "channels": int(data.shape[1]), "frames": int(data.shape[0])}
        (ref_dir / "sceneworks-derived" / f"{clip}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        print(f"{clip}: decoded {meta}, pcm sha256 {hashlib.sha256(raw).hexdigest()}")


GIB = 1024**3


def nvidia_smi_peak(csv_path: Path, pci: str | None) -> dict | None:
    """First sample and max `memory.used` (MiB) of the GPU at `pci` in a sampler CSV.

    Rows are `timestamp, index, pci.bus_id, memory.used` every 500 ms from before the case's
    process starts, so the first row is the pre-process baseline.
    """
    if not csv_path.is_file():
        return None
    used = []
    for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 4 or not cells[3].isdigit():
            continue
        if pci is None or cells[2].upper() == pci.upper():
            used.append(int(cells[3]))
    if not used:
        return None
    return {
        "baseline_mib": used[0],
        "peak_mib": max(used),
        "peak_above_baseline_gib": (max(used) - used[0]) / 1024,
        "samples": len(used),
    }


def summarize_memory(evidence: Path) -> None:
    """Fold the per-case JSONs and nvidia-smi CSVs into `memory-summary.{json,md}`."""
    cases = []
    for path in sorted(evidence.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "case" not in record or "phases" not in record:
            continue
        cuda = record.get("cuda") or {}
        csv_path = evidence / f"{record['case']}-{record['tier']}-vram.csv"
        cases.append(
            {
                "case": record["case"],
                "tier": record["tier"],
                "wall_s": record["wall_s"],
                "song_s": record["song_s"],
                "device_peak_above_baseline_gib": cuda.get("device_peak_above_baseline_gib"),
                "pool_reserved_high_gib": cuda.get("pool_reserved_high_gib"),
                "pool_used_high_gib": cuda.get("pool_used_high_gib"),
                "nvidia_smi": nvidia_smi_peak(csv_path, cuda.get("pci_bus_id")),
                "phases": {
                    p["phase"]: {
                        "wall_s": p["wall_s"],
                        "device_peak_above_baseline_gib": p.get("device_peak_above_baseline_gib"),
                        "pool_reserved_high_gib": p.get("pool_reserved_high_gib"),
                        "pool_used_high_gib": p.get("pool_used_high_gib"),
                    }
                    for p in record["phases"]
                },
            }
        )
    (evidence / "memory-summary.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    def gib(value) -> str:
        return "-" if value is None else f"{value:.2f}"

    phases = ["stage1_load", "stage1_decode", "stage2_load", "stage2", "decode"]
    lines = [
        "| case | tier | device peak GiB (above baseline) | nvidia-smi peak GiB | pool reserved GiB "
        "| pool used GiB | " + " | ".join(f"{p} dev/pool GiB" for p in phases) + " | wall s | song s |",
        "|" + "---|" * (8 + len(phases)),
    ]
    for c in cases:
        smi = c["nvidia_smi"] or {}
        per_phase = [
            f"{gib(c['phases'].get(p, {}).get('device_peak_above_baseline_gib'))}/"
            f"{gib(c['phases'].get(p, {}).get('pool_reserved_high_gib'))}"
            for p in phases
        ]
        lines.append(
            f"| {c['case']} | {c['tier']} | {gib(c['device_peak_above_baseline_gib'])} "
            f"| {gib(smi.get('peak_above_baseline_gib'))} | {gib(c['pool_reserved_high_gib'])} "
            f"| {gib(c['pool_used_high_gib'])} | " + " | ".join(per_phase)
            + f" | {c['wall_s']:.1f} | {c['song_s']:.1f} |"
        )
    table = "\n".join(lines) + "\n"
    (evidence / "memory-summary.md").write_text(table, encoding="utf-8")
    print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    root = commands.add_parser("stage-root", help="lay out YUE_SNAPSHOT_ROOT")
    root.add_argument("--root", required=True, type=Path)
    root.add_argument("--repo", required=True, action="append", help="NAME=SNAPSHOT_DIR")
    reference = commands.add_parser("stage-reference", help="populate YUE_REF_DIR")
    reference.add_argument("--ref-dir", required=True, type=Path)
    reference.add_argument(
        "--with-stems",
        action="store_true",
        help="also stage the vocal + instrumental stems the memory-measurement cases loop",
    )
    summary = commands.add_parser(
        "summarize-memory", help="fold the memory-mode case records into a summary table"
    )
    summary.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "stage-root":
        stage_root(args.root, args.repo)
    elif args.command == "stage-reference":
        stage_reference(args.ref_dir, args.with_stems)
    else:
        summarize_memory(args.evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
