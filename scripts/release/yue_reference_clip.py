#!/usr/bin/env python3
"""Stage YuE's upstream ICL reference clip for the CUDA real-weight smoke (sc-19387).

`crates/audio/candle-audio-yue/tests/registered_loader_real_weights.rs` reads
`$YUE_REF_DIR/sceneworks-derived/pop.00001.f32le` -- interleaved little-endian float32 PCM that
`scripts/reference/yue_icl_reference.py` decodes on the dev box from upstream's
`prompt_egs/pop.00001.mp3`. That producer needs the whole torch reference environment; this
script does only its decode step: fetch the MP3 at the pinned YuE commit, verify its SHA-256,
decode it with `soundfile` at float32 exactly as the producer does, and write the `.f32le`.

The decoded PCM's SHA-256 is compared with the committed fixture's `pcm_sha256` and the result is
printed: a match proves the runner fed the smoke the same samples the fixture was built from; a
mismatch (a different libsndfile/mpg123 build) is reported, not fatal, because the smoke asserts
output shape and finiteness, not reference-exact ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", required=True, type=Path, help="the YUE_REF_DIR to populate")
    args = parser.parse_args()

    with urllib.request.urlopen(CLIP_URL, timeout=120) as response:
        mp3 = response.read()
    actual = hashlib.sha256(mp3).hexdigest()
    if actual != CLIP_SHA256:
        raise SystemExit(f"{CLIP_URL}: sha256 {actual} != pinned {CLIP_SHA256}")
    raw_dir = args.ref_dir / "prompt_egs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = raw_dir / f"{CLIP}.mp3"
    mp3_path.write_bytes(mp3)

    import numpy as np
    import soundfile

    data, rate = soundfile.read(str(mp3_path), dtype="float32", always_2d=True)
    raw = np.ascontiguousarray(data).astype("<f4").tobytes()
    derived = args.ref_dir / "sceneworks-derived"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
