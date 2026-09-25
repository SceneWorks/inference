#!/usr/bin/env python3
"""Regenerate the YuE ICL reference-encoder parity fixture (sc-19379, epic sc-19373).

`crates/audio/candle-audio-yue/src/icl.rs` ports YuE-v1's in-context-learning reference path:
`load_audio_mono` (channel mean + torchaudio `Resample` to 16 kHz) → `encode_audio` (xcodec
`SoundStream.encode` at 0.5 kb/s — the DAC acoustic encoder, the HuBERT semantic branch through
the RepCodec semantic encoder, `fc_prior`, and codebook 0 of the RVQ) → `CodecManipulator.npy2ids`
→ the prompt window (single-track `[start·50 : end·50]`, dual-track vocal/instrumental interleave
`[start·100 : end·100]`). This script runs the **reference Python itself** and writes what it
produces to `crates/audio/candle-audio-yue/tests/fixtures/yue_icl_reference.json`:

* ``single`` / ``dual`` — the windowed codebook-0 mm ids (`audio_prompt_codec`), and the full
  first-segment stage-1 prompt (`prompt_ids` at `i == 1` in `infer.py`) for fixed genres/lyrics,
  so the native encoder + prompt builder are held to the reference token for token.
* per track, the SHA-256 of the int16 PCM the clip is built from. The clips are synthesized from
  integer-exact recipes (`synth_*`, an LCG plus `math.sin` quantized to int16) that the Rust test
  re-synthesizes, so the fixture carries no audio — the test fails loudly if its PCM ever differs.
* ``resample`` — torchaudio `Resample(sr, 16000)` (the resampler `load_audio_mono` builds) on a
  short int16 clip at several source rates, down- and up-sampling. Weights-free, so the Rust
  resampler is held to it on every CI run.
* per track, the RVQ nearest-codeword **margin** (`d₂ − d₁`, the gap between the best and
  second-best squared distance) at every frame — how far each reference token is from a float-noise
  flip. Diagnostics only; the Rust gate is exact ids.

Where the code comes from: the YuE-v1 clone (`multimodal-art-projection/YuE` @ `YuE-v1`, commit
``6d4f0b1f8ce6a55fb2392e959394c46e07ee334d``) with the ``m-a-p/xcodec_mini_infer`` snapshot at
``inference/xcodec_mini_infer`` (revision ``fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5``) from the
shared reference environment (``YUE_REF_DIR``, default ``~/.cache/sceneworks-yue-ref``; see its
README). `load_audio_mono` and `encode_audio` are **extracted from `infer.py` with `ast` and
executed as-is** (only `torchaudio.load` is redirected to the synthesized clip — it would return
the same `int16 / 32768` float32 tensor for a PCM-16 WAV); the ICL windowing and wrapping lines of
`infer.py`'s stage-1 loop are reproduced verbatim below, and `infer.py`'s SHA-256 is asserted so a
changed upstream cannot silently drift. The codec is the upstream `.pth` (`codec_model` state dict;
the SceneWorks safetensors rehost is tensor-equal), run in float32 on the CPU.

Regenerating (dev box, inside the YuE reference venv; loads the ~1.3 GB codec on CPU)::

    ~/.cache/sceneworks-yue-ref/venv/bin/python scripts/reference/yue_icl_reference.py \
        [--dump /tmp/yue_icl_intermediates]

``--dump`` also writes the intermediate tensors (resampled waves, HuBERT layer mean, acoustic and
semantic encodings, the `fc_prior` output) as safetensors for debugging a port; they are never
committed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "crates/audio/candle-audio-yue/tests/fixtures/yue_icl_reference.json"

YUE_COMMIT = "6d4f0b1f8ce6a55fb2392e959394c46e07ee334d"
XCODEC_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
INFER_SHA256 = "9e2fa795b1556d73e9a8c7a1c6c32bcb8b053d70ba7cf67b2c5ca9a4f510d47f"

GENRES = "inspiring female uplifting pop airy vocal electronic bright vocal"
LYRICS = "[verse]\nStaring at the sunset, colors paint the sky\n\n[chorus]\nDon't let this moment fade\n"

#: Source rates for the weights-free resampler golden (441:160, 3:1, 441:320 and 1:2 ratios).
RESAMPLE_RATES = [44_100, 48_000, 22_050, 8_000]
#: Length of the resampler golden's input clip (int16 PCM at each source rate).
RESAMPLE_SAMPLES = 499


#: Single-track reference: 44.1 kHz stereo (exercises the channel mean and the 441:160 resample).
#: The frame counts are deliberately not whole 20 ms multiples: at 16 kHz the DAC encoder then
#: yields one frame fewer than HuBERT, so `SoundStream.encode` takes its re-encode-padded branch
#: (the common case for real clips).
SINGLE = {"rate": 44_100, "channels": 2, "frames": 44_100 * 4 + 1_234, "start": 0.5, "end": 3.5}
#: Dual-track reference: two 48 kHz mono stems (3:1 resample), windowed 1.0–2.5 s.
DUAL = {"rate": 48_000, "channels": 1, "frames": 48_000 * 3 + 777, "start": 1.0, "end": 2.5}


# --------------------------------------------------------------------------------------------
# Integer-exact synthetic clips (mirrored by `synth_*` in tests/icl_parity.rs)
# --------------------------------------------------------------------------------------------


def lcg(state: int) -> int:
    """Numerical Recipes 32-bit LCG."""
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def to_i16(x: float) -> int:
    v = math.floor(x * 32767.0 + 0.5)
    return max(-32768, min(32767, v))


def synth_single(rate: int, frames: int) -> list[int]:
    """Interleaved stereo: L = plucked triad + noise hat, R = vibrato lead with harmonics."""
    out = []
    state = 12345
    for i in range(frames):
        t = i / rate
        state = lcg(state)
        noise = (state >> 8) / 16777216.0 - 0.5
        env = math.exp(-4.0 * (t % 0.5))
        hat = math.exp(-40.0 * (t % 0.25))
        chord = sum(math.sin(2.0 * math.pi * f * t) for f in (196.0, 246.94, 293.66)) / 3.0
        left = 0.5 * env * chord + 0.15 * hat * noise
        ph = 2.0 * math.pi * 392.0 * t + 4.0 * math.sin(2.0 * math.pi * 5.5 * t)
        right = 0.3 * (math.sin(ph) + 0.5 * math.sin(2.0 * ph) + 0.25 * math.sin(3.0 * ph))
        out.append(to_i16(left))
        out.append(to_i16(right))
    return out


def synth_vocals(rate: int, frames: int) -> list[int]:
    """A gliding, vibrato 'sung' line with vowel-like harmonic weights and syllable gating."""
    out = []
    for i in range(frames):
        t = i / rate
        f0 = 220.0 + 110.0 * (t / 3.0)
        ph = 2.0 * math.pi * f0 * t + 3.0 * math.sin(2.0 * math.pi * 6.0 * t)
        gate = 0.5 - 0.5 * math.cos(2.0 * math.pi * 2.0 * t)
        v = math.sin(ph) + 0.6 * math.sin(2.0 * ph) + 0.4 * math.sin(3.0 * ph) + 0.2 * math.sin(5.0 * ph)
        out.append(to_i16(0.35 * gate * v))
    return out


def synth_instrumental(rate: int, frames: int) -> list[int]:
    """A bass + chord bed with a kick and an LCG noise snare."""
    out = []
    state = 777
    for i in range(frames):
        t = i / rate
        state = lcg(state)
        noise = (state >> 8) / 16777216.0 - 0.5
        beat = t % 0.5
        kick = math.exp(-30.0 * beat) * math.sin(2.0 * math.pi * 60.0 * beat)
        snare = math.exp(-25.0 * ((t + 0.25) % 0.5)) * noise
        bass = math.sin(2.0 * math.pi * 55.0 * t)
        chord = sum(math.sin(2.0 * math.pi * f * t) for f in (261.63, 329.63, 392.0)) / 3.0
        out.append(to_i16(0.4 * kick + 0.3 * snare + 0.25 * bass + 0.2 * chord))
    return out


def synth_resample_input(n: int) -> list[int]:
    """LCG noise plus a slow sine: broadband, so every kernel phase is exercised."""
    out = []
    state = 4242
    for i in range(n):
        state = lcg(state)
        out.append(to_i16(0.4 * ((state >> 8) / 16777216.0 - 0.5) + 0.5 * math.sin(0.05 * i)))
    return out


def pcm_sha256(pcm: list[int]) -> str:
    return hashlib.sha256(b"".join(v.to_bytes(2, "little", signed=True) for v in pcm)).hexdigest()


# --------------------------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_functions(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in picked} == names, names
    return "\n\n".join(ast.get_source_segment(source, n) for n in picked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dump", type=Path, default=None, help="write intermediates here")
    args = parser.parse_args()

    ref_dir = Path(os.environ.get("YUE_REF_DIR", Path.home() / ".cache/sceneworks-yue-ref"))
    inf = (ref_dir / "yue" / "inference").resolve()
    infer_py = inf / "infer.py"
    got = sha256(infer_py)
    if got != INFER_SHA256:
        sys.exit(f"{infer_py} sha256 {got} != pinned {INFER_SHA256}")
    os.chdir(inf)  # upstream bakes `./xcodec_mini_infer/...` relative paths in
    sys.path[:0] = [str(inf), str(inf / "xcodec_mini_infer"),
                    str(inf / "xcodec_mini_infer" / "descriptaudiocodec")]

    import numpy as np
    import torch
    import torchaudio
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchaudio.transforms import Resample
    from codecmanipulator import CodecManipulator
    from mmtokenizer import _MMSentencePieceTokenizer
    from models.soundstream_hubert_new import SoundStream

    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = OmegaConf.load("./xcodec_mini_infer/final_ckpt/config.yaml")
    codec_model = SoundStream(**cfg.generator.config)
    state = torch.load("./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
                       map_location="cpu", weights_only=False)["codec_model"]
    codec_model.load_state_dict(state)
    codec_model.eval()
    mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
    codectool = CodecManipulator("xcodec", 0, 1)

    # `infer.py`'s own `load_audio_mono` / `encode_audio`, verbatim. `torchaudio.load` is
    # redirected to the synthesized clip (channels-first float32 = int16 / 32768, which is what it
    # returns for a PCM-16 WAV).
    clips: dict[str, tuple] = {}

    class _Torchaudio:
        @staticmethod
        def load(path):
            pcm, rate, channels = clips[path]
            data = torch.tensor(pcm, dtype=torch.int16).view(-1, channels).t().contiguous()
            return data.to(torch.float32) / 32768.0, rate

    namespace = {"torch": torch, "torchaudio": _Torchaudio, "Resample": Resample, "np": np}
    exec(extract_functions(infer_py.read_text(encoding="utf-8"), {"load_audio_mono", "encode_audio"}), namespace)
    load_audio_mono, encode_audio = namespace["load_audio_mono"], namespace["encode_audio"]
    del torchaudio  # only the redirected loader is used

    dumps = {}
    margins = {}

    def trace(name: str, path: str):
        """Re-run the encode path for `path` capturing intermediates and codeword margins."""
        audio = load_audio_mono(path).unsqueeze(0)
        with torch.no_grad():
            m = codec_model
            sem_in = m.get_regress_target(audio)
            sem = m.encoder_semantic(sem_in.transpose(1, 2))
            ac = m.encoder(audio)
            if ac.shape[2] != sem.shape[2]:
                ac = m.encoder(torch.transpose(torch.nn.functional.pad(audio[:, 0, :], (160, 160)).unsqueeze(0), 0, 1))
            e = m.fc_prior(torch.cat([ac, sem], dim=1).transpose(1, 2)).transpose(1, 2)
            embed = m.quantizer.vq.layers[0]._codebook.embed  # [1024, 1024]
            x = e[0].t()  # [T, D]
            dist = x.pow(2).sum(1, keepdim=True) - 2 * x @ embed.t() + embed.pow(2).sum(1)[None]
            top2 = dist.topk(2, dim=1, largest=False).values
            margins[name] = [float(v) for v in (top2[:, 1] - top2[:, 0])]
        dumps.update({
            f"{name}.wave16k": audio.reshape(-1).contiguous(),
            f"{name}.hubert_mean": sem_in[0].contiguous(),
            f"{name}.semantic": sem[0].contiguous(),
            f"{name}.acoustic": ac[0].contiguous(),
            f"{name}.fc_prior": e[0].contiguous(),
        })

    # ---- single-track (`--use_audio_prompt`) ----
    single_pcm = synth_single(SINGLE["rate"], SINGLE["frames"])
    clips["single.wav"] = (single_pcm, SINGLE["rate"], SINGLE["channels"])

    class A:  # the CLI args the ICL lines read
        prompt_start_time = SINGLE["start"]
        prompt_end_time = SINGLE["end"]

    audio_prompt = load_audio_mono("single.wav")
    raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
    code_ids = codectool.npy2ids(raw_codes[0])
    single_frames = len(code_ids)
    audio_prompt_codec = code_ids[int(A.prompt_start_time * 50): int(A.prompt_end_time * 50)]
    single_icl = [int(v) for v in audio_prompt_codec]
    single_prompt = first_segment_prompt(mmtokenizer, codectool, audio_prompt_codec)
    trace("single", "single.wav")

    # ---- dual-track (`--use_dual_tracks_prompt`) ----
    vocals_pcm = synth_vocals(DUAL["rate"], DUAL["frames"])
    inst_pcm = synth_instrumental(DUAL["rate"], DUAL["frames"])
    clips["vocals.wav"] = (vocals_pcm, DUAL["rate"], 1)
    clips["instrumental.wav"] = (inst_pcm, DUAL["rate"], 1)

    class B:
        prompt_start_time = DUAL["start"]
        prompt_end_time = DUAL["end"]

    args_ = B
    vocals_ids = load_audio_mono("vocals.wav")
    instrumental_ids = load_audio_mono("instrumental.wav")
    vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
    instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
    vocals_ids = codectool.npy2ids(vocals_ids[0])
    instrumental_ids = codectool.npy2ids(instrumental_ids[0])
    dual_frames = len(vocals_ids)
    ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
    audio_prompt_codec = ids_segment_interleaved[int(args_.prompt_start_time*50*2): int(args_.prompt_end_time*50*2)]
    audio_prompt_codec = audio_prompt_codec.tolist()
    dual_icl = [int(v) for v in audio_prompt_codec]
    dual_prompt = first_segment_prompt(mmtokenizer, codectool, audio_prompt_codec)
    trace("vocals", "vocals.wav")
    trace("instrumental", "instrumental.wav")

    resample_cases = []
    for rate in RESAMPLE_RATES:
        pcm = synth_resample_input(RESAMPLE_SAMPLES)
        x = torch.tensor(pcm, dtype=torch.int16).to(torch.float32).view(1, -1) / 32768.0
        y = Resample(orig_freq=rate, new_freq=16000)(x)
        resample_cases.append({"rate": rate, "output": [float(v) for v in y.reshape(-1)]})

    fixture = {
        "story": "sc-19379",
        "producer": "scripts/reference/yue_icl_reference.py",
        "yue_revision": YUE_COMMIT,
        "xcodec_mini_infer_revision": XCODEC_REVISION,
        "checkpoint": "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth [codec_model]",
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "target_bandwidth_kbps": 0.5,
        "genres": GENRES,
        "lyrics": LYRICS,
        "resample": {
            "input_samples": RESAMPLE_SAMPLES,
            "input_pcm_sha256": pcm_sha256(synth_resample_input(RESAMPLE_SAMPLES)),
            "cases": resample_cases,
        },
        "single": {
            **SINGLE,
            "pcm_sha256": pcm_sha256(single_pcm),
            "codec_frames": single_frames,
            "icl_ids": single_icl,
            "segment0_prompt_ids": single_prompt,
            "min_margin": min(margins["single"]),
        },
        "dual": {
            **DUAL,
            "vocals_pcm_sha256": pcm_sha256(vocals_pcm),
            "instrumental_pcm_sha256": pcm_sha256(inst_pcm),
            "codec_frames": dual_frames,
            "icl_ids": dual_icl,
            "segment0_prompt_ids": dual_prompt,
            "min_margin_vocals": min(margins["vocals"]),
            "min_margin_instrumental": min(margins["instrumental"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    for k, v in margins.items():
        s = sorted(v)
        print(f"{k}: frames {len(v)}, smallest margins {[f'{x:.3e}' for x in s[:5]]}")

    if args.dump:
        from safetensors.torch import save_file
        args.dump.mkdir(parents=True, exist_ok=True)
        save_file(dumps, str(args.dump / "icl_intermediates.safetensors"))
        (args.dump / "margins.json").write_text(json.dumps(margins), encoding="utf-8")
        print(f"dumped intermediates to {args.dump}")


def first_segment_prompt(mmtokenizer, codectool, audio_prompt_codec) -> list[int]:
    """`infer.py`'s stage-1 loop at `i == 1` for an ICL render, verbatim (lyrics/genres fixed)."""
    import re

    def split_lyrics(lyrics):
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

    genres = GENRES.strip()
    lyrics = split_lyrics(LYRICS.strip())
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    p = prompt_texts[1]
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
    sentence_ids = mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
    head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
    prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    return [int(v) for v in prompt_ids]


if __name__ == "__main__":
    main()
