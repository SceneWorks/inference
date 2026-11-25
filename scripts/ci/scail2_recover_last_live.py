#!/usr/bin/env python3
"""Replay the exact GPU 0 three-reference capture through completed RGB."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.scail2_recover_completed_job import replay_archive
from scripts.ci.scail2_recover_owned_zero import derive_certificates
from scripts.ci.scail_plural import contract as c
from scripts.ci.scail_plural.completed_job import LAST_LIVE_CAPTURE


COUNTS = [6, 2, 3, 4, 5]


def replay_files(dest, policy):
    return derive_certificates(dest, policy, LAST_LIVE_CAPTURE, COUNTS, [3],
                               "scail2-quiet-gpu0-last-live-recovery-v1", fresh_cuda=True)["3"]


def recover(archive):
    request = {"selection": c.SELECTION, "policy": c.policy("candle-cuda", cuda_uuid=c.GPU0)}
    return replay_archive(archive, LAST_LIVE_CAPTURE, 396263, COUNTS, request, replay_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(recover(parser.parse_args().archive), sort_keys=True, separators=(",", ":")))
