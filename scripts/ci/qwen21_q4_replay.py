"""Materialize only the two immutable replay PNGs, through authenticated release assets."""

import argparse
import json
from pathlib import Path
import urllib.request

from scripts.ci.qwen21_adapter_imports import materialize
from scripts.ci.qwen21_diagnostic_adapter import ADAPTER_SHA, RUN, SOURCE


def validate_replay(manifest):
    expected = {
        "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False,
        "repository": "SceneWorks/inference", "releaseId": 402708057,
        "protocolSha256": "1e42ed9cd80f45712cdb75f6ee63d93ae4afe7b310f6bfbc9f645f8b2c634fab",
        "replayRun": 37223595546, "replaySource": "83bd4c53ce9abb3f2c755415c0a92d39ed2096c7",
        "trainingSourceMain": SOURCE, "trainingRun": RUN, "adapterSha256": ADAPTER_SHA,
        "adapters": [
            {"name": "q4_edit_base", "kind": "q4_replay_png", "assetId": 610462396,
             "assetName": "qwen21_q4_replay_37223595546_base.png", "size": 232879,
             "sha256": "ba8baf8aa439a0c00e952b9615669dbde38844ec92bb3a7aa48fe79d89708235",
             "file": "q4_edit_base.png"},
            {"name": "q4_edit_mlx_lokr", "kind": "q4_replay_png", "assetId": 610462398,
             "assetName": "qwen21_q4_replay_37223595546_structured.png", "size": 306523,
             "sha256": "b8ffa6b98d4f2ddbf855060630c15d258544628fb2a1b5f0244ad880f4e7ee55",
             "file": "q4_edit_mlx_lokr.png"},
        ],
    }
    if manifest != expected:
        raise ValueError("immutable Q4 replay identities, original donor and diagnostic-only scope required")


def prepare(manifest, destination, opener=urllib.request.urlopen):
    validate_replay(manifest)
    return materialize(manifest, destination, opener)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.destination))


if __name__ == "__main__":
    main()
