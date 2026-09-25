#!/usr/bin/env python3
"""Regenerate the YuE Vocos upsampler + low-band splice reference-parity fixture (sc-19378).

Two Rust tests read the fixture this writes:

* ``crates/audio/candle-audio-yue/tests/vocos_parity.rs`` (real weights, ``--ignored``) holds the
  native candle Vocos decoders (``candle_audio_yue::vocoder``) to the upstream PyTorch
  ``VocosDecoder`` — the vocal checkpoint ``decoder_131000`` and the instrumental checkpoint
  ``decoder_151000`` — on the embedding the upstream pipeline actually feeds them:
  ``SoundStream.get_embed(codes)`` (``xcodec_mini_infer/vocoder.py`` ``process_audio``).
* ``crates/audio/candle-audio-yue/src/splice.rs`` (weights-free, always runs) holds the native
  post-process to the upstream one on the same two stems.

The upstream post-process (``yue/inference/infer.py`` after stage 2) round-trips through files;
this script runs the same arithmetic in memory, calling the upstream functions themselves:

1. ``recons``: each track's 16 kHz ``codec.decode`` goes through ``save_audio(…, 16000)`` —
   ``rescale`` is never passed there, so it is always the ``clamp(±0.99)`` branch — and the two
   stems are summed into the 16 kHz mix.
2. ``vocoder``: each track's Vocos output is a 44.1 kHz stem (saved through
   ``save_audio(…, rescale)``); the mix is ``instrumental + vocal``, also saved through
   ``save_audio(…, rescale)`` — ``rescale`` false = ``clamp(±0.99)``, true = ``× min(0.99/peak, 1)``.
3. ``post_process_audio.replace_low_freq_with_energy_matched(recons_mix, vocoder_mix,
   cutoff_freq=5500)`` — executed verbatim, with ``torchaudio.load`` / ``torchaudio.save`` swapped
   for an in-memory store, so no lossy (mp3) encode sits between the stages.

Both ``rescale`` settings are recorded. The grids are **in-distribution**: two deterministic,
stdlib-synthesized clips (a vibrato lead for the vocal track, a chord + bass + clicks bed for the
instrumental one; no third-party audio) are *encoded* by the upstream codec at 4 kb/s — exactly 8
quantizers — and the first ``FRAMES`` frames of each are kept. The clips are loud enough that the
44.1 kHz mix exceeds the 0.99 limit, so the clamp and the rescale both act.

The reference implementation is **not** vendored. It is the YuE-v1 clone (with the
``m-a-p/xcodec_mini_infer`` snapshot at ``yue/inference/xcodec_mini_infer``) that
``YUE_REFERENCE_INFERENCE_DIR`` names, run in float32 on the CPU from the upstream ``.pth``
checkpoints (the SceneWorks safetensors rehost is tensor-equal to them). This script never resolves
repository ids or derives a Hugging Face cache location (epic 13657).

Regenerating (dev box, inside the YuE reference venv)::

    export YUE_REFERENCE_INFERENCE_DIR=/path/to/YuE/inference   # YuE-v1 clone, xcodec inside
    python scripts/reference/yue_vocos_reference.py

It rewrites ``tests/fixtures/vocos_splice_reference.safetensors`` and its metadata JSON in place.
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
FIXTURE = FIXTURE_DIR / "vocos_splice_reference.safetensors"
METADATA = FIXTURE_DIR / "vocos_splice_reference.json"

INFERENCE_DIR_ENV = "YUE_REFERENCE_INFERENCE_DIR"

#: The pinned upstream revisions (epic sc-19373 reference environment).
YUE_REVISION = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
XCODEC_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"

CODEC_RATE = 16_000
VOCODER_RATE = 44_100
TARGET_BANDWIDTH = 4.0
NUM_CODEBOOKS = 8
#: Frames committed per track (0.5 s at 50 Hz) — spans the Vocos backbone's 55-frame receptive
#: field's worth of context on both edges and the splice filters' settling many times over.
FRAMES = 25
CLIP_SECONDS = 1
CUTOFF_HZ = 5500.0


def lcg(seed: int):
    """A deterministic stdlib noise source in [-1, 1)."""
    state = seed
    while True:
        state = (state * 1103515245 + 12345) % (1 << 31)
        yield state / (1 << 30) - 1.0


def vocal_clip() -> list[float]:
    """A loud vibrato lead with a formant-ish second partial — stdlib only."""
    out = []
    for i in range(CODEC_RATE * CLIP_SECONDS):
        t = i / CODEC_RATE
        f = 330.0 * (1.0 + 0.02 * math.sin(2 * math.pi * 5.5 * t))
        ph = 2 * math.pi * f * t
        env = 0.6 + 0.4 * math.sin(2 * math.pi * 2.0 * t) ** 2
        out.append(env * (0.95 * math.sin(ph) + 0.25 * math.sin(3 * ph + 0.3)))
    return out


def instrumental_clip() -> list[float]:
    """A loud chord + bass + decaying noise clicks every 125 ms — stdlib only."""
    noise = lcg(19378)
    out = []
    for i in range(CODEC_RATE * CLIP_SECONDS):
        t = i / CODEC_RATE
        chord = sum(math.sin(2 * math.pi * f * t) for f in (261.63, 329.63, 392.0)) / 3
        bass = math.sin(2 * math.pi * 65.41 * t)
        click = math.exp(-60.0 * (t % 0.125)) * next(noise)
        out.append(0.5 * chord + 0.4 * bass + 0.35 * click)
    return out


def main() -> None:
    inference_dir = os.environ.get(INFERENCE_DIR_ENV)
    if not inference_dir:
        sys.exit(f"set {INFERENCE_DIR_ENV} to the YuE-v1 clone's inference/ directory")
    inf = Path(inference_dir).resolve()
    os.chdir(inf)
    sys.path[:0] = [str(inf), str(inf / "xcodec_mini_infer"),
                    str(inf / "xcodec_mini_infer" / "descriptaudiocodec")]

    import torch
    import torchaudio
    from omegaconf import OmegaConf
    from models.soundstream_hubert_new import SoundStream
    from vocos import VocosDecoder
    import post_process_audio
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = OmegaConf.load("./xcodec_mini_infer/final_ckpt/config.yaml")
    codec = SoundStream(**cfg.generator.config)
    codec.load_state_dict(torch.load("./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
                                     map_location="cpu", weights_only=False)["codec_model"])
    codec.eval()

    def vocos(path: str):
        # Upstream `vocoder.build_codec_model`, minus its CUDA-only `torch.load` (no map_location).
        d = VocosDecoder.from_hparams(config_path="./xcodec_mini_infer/decoders/config.yaml")
        d.load_state_dict(torch.load(path, map_location="cpu"))
        return d.eval()

    decoders = {
        "vocal": vocos("./xcodec_mini_infer/decoders/decoder_131000.pth"),
        "inst": vocos("./xcodec_mini_infer/decoders/decoder_151000.pth"),
    }

    # `infer.py`'s save_audio limiter, as arithmetic (the file write is the only thing dropped).
    def limit(wav: torch.Tensor, rescale: bool) -> torch.Tensor:
        lim = 0.99
        max_val = wav.abs().max()
        return wav * min(lim / max_val, 1) if rescale else wav.clamp(-lim, lim)

    tensors = {}
    tracks = {}
    for name, clip in (("vocal", vocal_clip()), ("inst", instrumental_clip())):
        x = torch.tensor(clip, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            codes = codec.encode(x, target_bw=TARGET_BANDWIDTH)  # [n_q, B, T]
            assert codes.shape[0] == NUM_CODEBOOKS, codes.shape
            codes = codes[:, :, :FRAMES].contiguous()
            wave16 = codec.decode(codes).reshape(1, -1)  # [1, T*320]
            embed = codec.get_embed(codes)  # [1, 1024, T] — what process_audio feeds Vocos
            wave44 = decoders[name](embed)  # [1, T*882]
        tracks[name] = (wave16, wave44)
        tensors[f"{name}_codes"] = codes[:, 0, :].to(torch.int64).contiguous()
        tensors[f"{name}_wave16"] = wave16.reshape(-1).contiguous()
        tensors[f"{name}_wave44"] = wave44.reshape(-1).contiguous()

    # The recons (16 kHz) mix: clamp each stem (save_audio's default branch), then sum.
    recons_mix = limit(tracks["vocal"][0], False) + limit(tracks["inst"][0], False)

    # Swap the upstream post-process's file I/O for an in-memory store and run it verbatim.
    store = {}
    post_process_audio.torchaudio.load = lambda path: store[path]
    post_process_audio.torchaudio.save = lambda path, wav, sample_rate: store.__setitem__(
        path, (wav, sample_rate))

    stats = {}
    for rescale in (False, True):
        mode = "rescale" if rescale else "clamp"
        vocal44, inst44 = tracks["vocal"][1], tracks["inst"][1]
        mix = limit(inst44 + vocal44, rescale)
        store["a"] = (recons_mix, CODEC_RATE)
        store["b"] = (mix, VOCODER_RATE)
        post_process_audio.replace_low_freq_with_energy_matched(
            a_file="a", b_file="b", c_file="c", cutoff_freq=CUTOFF_HZ)
        final, sr = store["c"]
        assert sr == VOCODER_RATE
        tensors[f"{mode}_mix"] = final.reshape(-1).contiguous()
        tensors[f"{mode}_vocal"] = limit(vocal44, rescale).reshape(-1).contiguous()
        tensors[f"{mode}_inst"] = limit(inst44, rescale).reshape(-1).contiguous()
        stats[f"{mode}_vocoder_mix_peak_before_limit"] = float((inst44 + vocal44).abs().max())
        stats[f"{mode}_final_peak"] = float(final.abs().max())
    stats["recons_mix_peak"] = float(recons_mix.abs().max())
    for name, (w16, w44) in tracks.items():
        stats[f"{name}_wave16_peak"] = float(w16.abs().max())
        stats[f"{name}_wave44_peak"] = float(w44.abs().max())

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(FIXTURE))
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=inf, capture_output=True,
                         text=True).stdout.strip()
    METADATA.write_text(json.dumps({
        "story": "sc-19378",
        "producer": "scripts/reference/yue_vocos_reference.py",
        "yue_revision": YUE_REVISION,
        "yue_clone_head": rev,
        "xcodec_mini_infer_revision": XCODEC_REVISION,
        "checkpoints": {
            "codec": "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth [codec_model]",
            "vocal": "xcodec_mini_infer/decoders/decoder_131000.pth",
            "inst": "xcodec_mini_infer/decoders/decoder_151000.pth",
        },
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "device": "cpu",
        "dtype": "float32",
        "clips": "synthetic stdlib vocal lead / instrumental bed, 1 s @ 16 kHz",
        "target_bandwidth_kbps": TARGET_BANDWIDTH,
        "frames": FRAMES,
        "cutoff_hz": CUTOFF_HZ,
        "tensors": {k: list(v.shape) for k, v in sorted(tensors.items())},
        "stats": stats,
        "sha256": digest,
    }, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, sha256 {digest})")
    if rev != YUE_REVISION:
        sys.exit(f"YuE clone HEAD {rev} != pinned {YUE_REVISION}")


if __name__ == "__main__":
    main()
