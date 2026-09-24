#!/usr/bin/env python3
"""Regenerate the YuE xcodec decode reference-parity fixture (sc-19377).

`crates/audio/candle-audio-yue/tests/xcodec_parity.rs` holds the native candle xcodec decoder
(`candle_audio_yue::codec`) to the upstream PyTorch `SoundStream` on the same 8-codebook grid:

* ``get_embed(codes)`` — the RVQ dequant (the Vocos upsampler's input), ``[1, 1024, T]``;
* ``decode(codes)`` — RVQ dequant → ``fc_post2`` → DAC ``decoder_2``, a 16 kHz waveform.

The grid is **in-distribution**: a deterministic, stdlib-synthesized musical clip (no third-party
audio, no licence to clear) is *encoded* by the upstream codec at 4 kb/s — exactly
``NUM_CODEBOOKS = 8`` quantizers — and the first ``FRAMES`` frames are kept. Random codes would
decode too, but a grid the codec's own encoder produced exercises the decoder where the model runs.

The reference implementation is **not** vendored. It is the YuE-v1 clone (with the
``m-a-p/xcodec_mini_infer`` snapshot at ``yue/inference/xcodec_mini_infer``) that
``YUE_REFERENCE_INFERENCE_DIR`` names, run in float32 on the CPU. The checkpoint is the upstream
``final_ckpt/ckpt_00360000.pth`` (the SceneWorks safetensors rehost is tensor-equal to it).
This script never resolves repository ids or derives a Hugging Face cache location (epic 13657).

Regenerating (dev box, inside the YuE reference venv)::

    export YUE_REFERENCE_INFERENCE_DIR=/path/to/YuE/inference   # YuE-v1 clone, xcodec inside
    python scripts/reference/yue_xcodec_reference.py

It rewrites ``tests/fixtures/xcodec_decode_reference.safetensors`` and its metadata JSON in place.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "crates" / "audio" / "candle-audio-yue" / "tests" / "fixtures"
FIXTURE = FIXTURE_DIR / "xcodec_decode_reference.safetensors"
METADATA = FIXTURE_DIR / "xcodec_decode_reference.json"

INFERENCE_DIR_ENV = "YUE_REFERENCE_INFERENCE_DIR"

#: The codec's pinned upstream revisions (epic sc-19373 reference environment).
YUE_REVISION = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
XCODEC_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"

SAMPLE_RATE = 16_000
#: 4 kb/s at 0.5 kb/s per quantizer = the 8 codebooks YuE's stage 2 emits.
TARGET_BANDWIDTH = 4.0
NUM_CODEBOOKS = 8
#: Frames committed (0.5 s at 50 Hz) — keeps the fixture small while spanning every conv's
#: receptive field many times over.
FRAMES = 25
#: Seconds of synthetic clip encoded (the codec needs a whole second for the HuBERT framing).
CLIP_SECONDS = 1


def synth_clip() -> list[float]:
    """A deterministic chord with a plucked envelope and a vibrato lead — stdlib only."""
    n = SAMPLE_RATE * CLIP_SECONDS
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-3.0 * (t % 0.25))
        chord = sum(math.sin(2 * math.pi * f * t) for f in (220.0, 277.18, 329.63)) / 3
        lead = math.sin(2 * math.pi * 440.0 * t + 3.0 * math.sin(2 * math.pi * 5.0 * t))
        out.append(0.45 * env * chord + 0.2 * lead)
    return out


def main() -> None:
    inference_dir = os.environ.get(INFERENCE_DIR_ENV)
    if not inference_dir:
        sys.exit(f"set {INFERENCE_DIR_ENV} to the YuE-v1 clone's inference/ directory")
    inf = Path(inference_dir).resolve()
    # Upstream bakes `./xcodec_mini_infer/...` relative paths into SoundStream.__init__.
    os.chdir(inf)
    sys.path[:0] = [str(inf), str(inf / "xcodec_mini_infer"),
                    str(inf / "xcodec_mini_infer" / "descriptaudiocodec")]

    import torch
    from omegaconf import OmegaConf
    from models.soundstream_hubert_new import SoundStream
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = OmegaConf.load("./xcodec_mini_infer/final_ckpt/config.yaml")
    codec = SoundStream(**cfg.generator.config)
    state = torch.load("./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
                       map_location="cpu", weights_only=False)["codec_model"]
    codec.load_state_dict(state)
    codec.eval()

    clip = torch.tensor(synth_clip(), dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        codes = codec.encode(clip, target_bw=TARGET_BANDWIDTH)  # [n_q, B, T]
        assert codes.shape[0] == NUM_CODEBOOKS, codes.shape
        codes = codes[:, :, :FRAMES].contiguous()
        embed = codec.get_embed(codes)  # [1, 1024, T]
        wave = codec.decode(codes)  # [1, 1, T*320]

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "codes": codes[:, 0, :].to(torch.int64).contiguous(),  # [8, T]
            "embed": embed[0].contiguous(),  # [1024, T]
            "wave": wave.reshape(-1).contiguous(),  # [T*320]
        },
        str(FIXTURE),
    )
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=inf, capture_output=True,
                         text=True).stdout.strip()
    METADATA.write_text(json.dumps({
        "story": "sc-19377",
        "producer": "scripts/reference/yue_xcodec_reference.py",
        "yue_revision": YUE_REVISION,
        "yue_clone_head": rev,
        "xcodec_mini_infer_revision": XCODEC_REVISION,
        "checkpoint": "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth [codec_model]",
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "clip": "synthetic stdlib chord + vibrato lead, 1 s @ 16 kHz (synth_clip)",
        "target_bandwidth_kbps": TARGET_BANDWIDTH,
        "frames": FRAMES,
        "tensors": {"codes": list(codes[:, 0, :].shape), "embed": list(embed[0].shape),
                    "wave": [wave.numel()]},
        "wave_abs_max": float(wave.abs().max()),
        "sha256": digest,
    }, indent=2) + "\n")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, sha256 {digest})")
    if rev != YUE_REVISION:
        sys.exit(f"YuE clone HEAD {rev} != pinned {YUE_REVISION}")


if __name__ == "__main__":
    main()
