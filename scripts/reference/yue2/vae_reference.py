#!/usr/bin/env python3
"""Regenerate the YuE2 VAE parity fixtures (sc-22993, epic sc-22988).

The native Candle VAE (``candle_audio_yue2::vae`` + ``candle_audio_yue2::decode``) is held to the
pinned upstream ``yue2.modeling_vae.YuE2VAE`` and to upstream's own post-processing in
``yue2.pipeline.YuE2Pipeline.decode`` (finite check, ``clamp(-1, 1)``, ``[S, 2]`` channel-last).
The reference is the installed pinned package (``import yue2``), never a copy.

Two modes:

``tiny`` (default) — the always-run, weights-free CI fixture, committed under
``crates/audio/candle-audio-yue2/tests/fixtures/``. It builds two upstream ``YuE2VAE`` models with
the **released topology** (strides ``[2, 2, 4, 4, 5, 6]``, 64 latent channels, SnakeBeta,
weight-norm, no final tanh, 2 audio channels) at toy widths, randomizes every parameter (weight-norm
``g``/``v``, biases, SnakeBeta ``alpha``/``beta`` — none at identity), scales the final conv so ~10% of
the raw waveform overshoots ±1 (so clamping is exercised) and saves each with upstream's own
``save_pretrained`` as ``vae_tiny/<variant>/{config.json,model.safetensors}`` (``release_variant``
``standard`` / ``legacy``, different seeds). ``vae_tiny_reference.safetensors`` holds, for one
shared ``[frames, 64]`` latent (the ``latent.npy`` layout ``synthesize`` returns):

* ``<variant>.full_raw`` — ``YuE2VAE.decode`` (unclamped ``[1, 2, S]``);
* ``<variant>.pipeline_tiled`` — ``YuE2Pipeline.decode`` itself (called unbound on a stub carrying
  only the attributes it reads) with ``vae_core_frames=TINY_CORE``, i.e. the clamped ``[S, 2]``
  production waveform. Python's own ``decode_tiled``-vs-``decode`` and tiled-vs-full pipeline
  differences are recorded in the JSON rather than stored (both measure 0.0 at this size);
* ``<variant>.encode_mean`` / ``encode_scale`` / ``encode_stdev`` / ``encode_sampled`` for an
  asymmetric stereo ``audio`` and injected ``noise`` (upstream draws the noise from a
  ``torch.Generator``; the native port takes it as input, epic E9).

``vae_tiny_reference.json`` records the shapes, natural lengths, ``required_halo``, the SHA-256 of
``np.save(latent)`` (the native ``.npy`` writer must reproduce it byte for byte) and provenance.

``real`` — the pinned-weight reference for the ``#[ignore]``d real-weight tests. Needs
``YUE2_HF_HUB`` (a hub directory holding the pinned ``m-a-p/YuE2-Vae`` and ``m-a-p/YuE2-Vae-legacy``
revisions; both are integrity-checked by upstream's ``model_identity(verify=True)`` first). A
deterministic asymmetric stereo clip (LEFT: a near-full-scale 220 Hz square wave whose
reconstruction overshoots ±1; RIGHT: a quiet 659.25 Hz sine) is encoded by the standard VAE's
encoder (posterior mean) into ``REAL_FRAMES`` latent frames; those SAME latents are decoded by both
pinned decoders through ``YuE2Pipeline.decode`` (tiled at the default core, and full) and by
``decode``/``decode_tiled(core_frames=REAL_CORE)``. Both encoders' ``(mean, scale)`` are stored too.
Because these waveforms are derived from CC BY-NC 4.0 weights they are written OUTSIDE the
repository (``--out``, default ``~/.cache/sceneworks-yue2-fixtures/vae``); only
``vae_real_reference.json`` (their SHA-256, shapes, statistics and Python's own measured
tiled-vs-full error) is committed, and the Rust test refuses reference files whose hash differs.

Run with the pinned reference environment (``setup_reference_env.sh``)::

    ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/vae_reference.py tiny
    HF_HUB_OFFLINE=1 YUE2_HF_HUB=/path/to/hub \\
        ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/vae_reference.py real

Expected peak RSS: tiny < 0.5 GB; real ~2.5 GB (one 530 MB FP32 VAE resident at a time, loaded
twice over by ``safe_open`` + ``load_state_dict``, plus torch) — CPU only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "crates/audio/candle-audio-yue2/tests/fixtures"
YUE2_COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"

VAE = ("m-a-p/YuE2-Vae", "95535e72a97bc0f09b8ada125d26b4009428c0e8")
VAE_LEGACY = ("m-a-p/YuE2-Vae-legacy", "b54118f0fc462f08999d1ec07e88817f4ee3f770")

#: Tiny-mode geometry: released strides and latent width, toy channel widths.
TINY_CHANNELS, TINY_C_MULTS = 3, [1, 2, 2, 3, 3, 4]
TINY_FRAMES, TINY_CORE, TINY_AUDIO = 7, 3, 3 * 1920 + 37
TINY_SEEDS = {"standard": 22993, "legacy": 22994}
#: The final-conv rescale puts this quantile of |raw| at 1.0, so ~10% of the raw waveform
#: exceeds ±1 and clamping is exercised on both sides of the threshold.
TINY_CLAMP_QUANTILE = 0.9

#: Real-mode clip length in latent frames (0.64 s) and the tile core used for seam coverage.
REAL_FRAMES, REAL_CORE = 16, 4
#: A longer latent (3 s) for Python's own measured tiled-vs-full boundary error (not stored).
REAL_LONG_FRAMES, REAL_LONG_CORE = 75, 16


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def environment() -> dict:
    env = Path(os.environ.get("YUE2_REF_DIR", Path.home() / ".cache/sceneworks-yue2-ref"))
    data = json.loads((env / "ENVIRONMENT.json").read_text())
    if data["yue2_commit"] != YUE2_COMMIT:
        sys.exit(f"reference environment is at {data['yue2_commit']}, expected {YUE2_COMMIT}")
    packages = data["packages"]
    return {
        "yue2_commit": YUE2_COMMIT,
        "python": data["python"],
        "torch": packages["torch"],
        "numpy": packages["numpy"],
        "safetensors": packages["safetensors"],
        "transformers": packages["transformers"],
        "device": "cpu",
        "dtype": "float32",
    }


def pipeline_decode(vae, latents_t64: np.ndarray, core_frames: int, full: bool) -> np.ndarray:
    """Run upstream ``YuE2Pipeline.decode`` itself on a stub holding only what it reads."""
    from yue2.pipeline import YuE2Pipeline

    stage = types.SimpleNamespace(update=lambda *a, **k: None)

    @contextlib.contextmanager
    def status(*_a, **_k):
        yield stage

    stub = types.SimpleNamespace(
        _status=status,
        _model=None,
        _vae=vae,
        device=torch.device("cpu"),
        vae_core_frames=core_frames,
        progress=False,
    )
    return YuE2Pipeline.decode(stub, latents_t64, full=full)


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def tiny_config(variant: str):
    from yue2.modeling_vae import YuE2VAEConfig

    common = dict(channels=TINY_CHANNELS, c_mults=TINY_C_MULTS, strides=[2, 2, 4, 4, 5, 6],
                  use_snake=True)
    return YuE2VAEConfig(
        encoder_config=dict(common, in_channels=2, latent_dim=128),
        decoder_config=dict(common, out_channels=2, latent_dim=64, snake_type="vanilla",
                            use_filter=False, final_tanh=False),
        latent_dim=64, release_variant=variant)


def randomize(model, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            # Magnitudes keep activations O(1) through the stack: with trained-VAE-like
            # conditioning, FP32 rounding differences between runtimes stay ~1e-6 instead of
            # being amplified by steep random SnakeBeta slopes (exp(α)·exp(-β) ≫ 1).
            if name.endswith("weight_g"):
                param.copy_(0.3 + 0.2 * noise.abs())
            elif name.endswith("weight_v"):
                param.copy_(noise)
            elif name.endswith("bias"):
                param.copy_(0.1 * noise)
            elif name.endswith(("alpha", "beta")):
                param.copy_(0.2 * noise)
            else:
                raise ValueError(f"unexpected parameter {name}")


def tiny() -> None:
    from yue2.modeling_vae import YuE2VAE

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2299301)
    latent_t64 = torch.randn((TINY_FRAMES, 64), generator=generator)
    latent = latent_t64.T.unsqueeze(0).contiguous()
    t = torch.arange(TINY_AUDIO, dtype=torch.float32) / 48000.0
    audio = torch.stack([
        0.6 * torch.sin(2 * math.pi * 330.0 * t) + 0.1 * torch.randn(TINY_AUDIO, generator=generator),
        0.2 * torch.sin(2 * math.pi * 1210.0 * t),
    ]).unsqueeze(0)

    tensors = {"latent": latent_t64.contiguous(), "audio": audio.contiguous()}
    meta_variants = {}
    root = FIXTURE_DIR / "vae_tiny"
    for variant, seed in TINY_SEEDS.items():
        model = YuE2VAE(tiny_config(variant))
        randomize(model, seed)
        # Scale the final (bias-free, linear) conv so ~10% of |raw| exceeds 1: clamping bites.
        final = model.decoder.layers[-2]
        level = float(torch.quantile(model.decode(latent).abs().flatten(), TINY_CLAMP_QUANTILE))
        with torch.no_grad():
            final.weight_g.mul_(1.0 / level)
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        (out / "modeling_vae.py").unlink(missing_ok=True)  # remote code is never committed/run
        # Reload through upstream's own loader so the reference runs on the saved bytes.
        decoder = YuE2VAE.from_pretrained(out, decoder_only=True)
        full_model = YuE2VAE.from_pretrained(out)
        full_raw = decoder.decode(latent)
        tiled_raw = decoder.decode_tiled(latent, core_frames=TINY_CORE, halo_frames=16)
        pipe_tiled = pipeline_decode(decoder, latent_t64.numpy(), TINY_CORE, full=False)
        pipe_full = pipeline_decode(decoder, latent_t64.numpy(), TINY_CORE, full=True)
        mean, info = full_model.encode(audio, return_info=True)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed + 7))
        sampled = noise * info["stdev"] + mean
        over = int((full_raw.abs() > 1).sum())
        if not 0 < over < full_raw.numel() // 2:
            sys.exit(f"{variant}: {over} samples beyond ±1 — clamp coverage is degenerate")
        tensors.update({
            f"{variant}.full_raw": full_raw.contiguous(),
            f"{variant}.pipeline_tiled": torch.from_numpy(pipe_tiled).contiguous(),
            f"{variant}.encode_mean": mean.contiguous(),
            f"{variant}.encode_scale": info["scale"].contiguous(),
            f"{variant}.encode_stdev": info["stdev"].contiguous(),
            f"{variant}.encode_noise": noise.contiguous(),
            f"{variant}.encode_sampled": sampled.contiguous(),
        })
        meta_variants[variant] = {
            "seed": seed,
            "config_sha256": sha256_file(out / "config.json"),
            "weights_sha256": sha256_file(out / "model.safetensors"),
            "natural_output_length": decoder.natural_output_length(TINY_FRAMES),
            "required_halo": decoder.required_halo(TINY_CORE),
            "raw_samples_beyond_unit": over,
            "raw_abs_max": float(full_raw.abs().max()),
            "python_tiled_vs_full_max_abs": max_abs(tiled_raw, full_raw),
            "python_pipeline_tiled_vs_full_max_abs": float(np.abs(pipe_tiled - pipe_full).max()),
            "encode_frames": int(mean.shape[-1]),
        }

    save_file(tensors, str(FIXTURE_DIR / "vae_tiny_reference.safetensors"))
    meta = {
        "story": "sc-22993",
        "producer": "scripts/reference/yue2/vae_reference.py tiny",
        "reference": environment(),
        "topology": {"channels": TINY_CHANNELS, "c_mults": TINY_C_MULTS,
                     "strides": [2, 2, 4, 4, 5, 6], "latent_dim": 64},
        "frames": TINY_FRAMES,
        "core_frames": TINY_CORE,
        "halo_frames": 16,
        "audio_samples": TINY_AUDIO,
        "latent_npy_sha256": sha256_bytes(npy_bytes(latent_t64.numpy())),
        "latent_sha256": sha256_bytes(latent_t64.numpy().astype("<f4").tobytes()),
        "variants": meta_variants,
    }
    (FIXTURE_DIR / "vae_tiny_reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def snapshot(hub: Path, repo: str, revision: str) -> Path:
    from yue2.storage import model_identity

    path = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        sys.exit(f"offline cache miss: {repo}@{revision} is not in {hub}")
    model_identity(path, verify=True)  # upstream's own integrity routine
    return path


def real_clip() -> torch.Tensor:
    n = REAL_FRAMES * 1920
    t = torch.arange(n, dtype=torch.float64) / 48000.0
    left = 0.97 * torch.sign(torch.sin(2 * math.pi * 220.0 * t))
    right = 0.3 * torch.sin(2 * math.pi * 659.25 * t)
    return torch.stack([left, right]).to(torch.float32).unsqueeze(0).contiguous()


def real(hub: Path, out: Path) -> None:
    from yue2.modeling_vae import YuE2VAE

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    out.mkdir(parents=True, exist_ok=True)
    clip = real_clip()
    tensors = {"clip": clip}
    record = {}
    latent = None
    long_latent = None
    for key, (repo, revision) in (("standard", VAE), ("legacy", VAE_LEGACY)):
        path = snapshot(hub, repo, revision)
        model = YuE2VAE.from_pretrained(path, local_files_only=True)
        mean, info = model.encode(clip, return_info=True)
        tensors[f"{key}.encode_mean"] = mean.contiguous()
        tensors[f"{key}.encode_scale"] = info["scale"].contiguous()
        if latent is None:
            # The shared cached latent: the standard encoder's posterior mean, [frames, 64].
            latent = mean.contiguous()
            tensors["latent"] = latent[0].T.contiguous()
            g = torch.Generator().manual_seed(2299375)
            long_latent = torch.randn((1, 64, REAL_LONG_FRAMES), generator=g) * mean.std()
        latent_t64 = latent[0].T.contiguous().numpy()
        full_raw = model.decode(latent)
        tiled_raw = model.decode_tiled(latent, core_frames=REAL_CORE, halo_frames=16)
        pipe_default = pipeline_decode(model, latent_t64, 1024, full=False)
        pipe_full = pipeline_decode(model, latent_t64, 1024, full=True)
        long_full = model.decode(long_latent)
        long_tiled = model.decode_tiled(long_latent, core_frames=REAL_LONG_CORE, halo_frames=16)
        tensors[f"{key}.pipeline_default"] = torch.from_numpy(pipe_default).contiguous()
        tensors[f"{key}.full_raw"] = full_raw.contiguous()
        left, right = full_raw[0, 0], full_raw[0, 1]
        record[key] = {
            "repo": repo,
            "revision": revision,
            "weights_sha256": sha256_file(path / "model.safetensors"),
            "config_sha256": sha256_file(path / "config.json"),
            "natural_output_length": model.natural_output_length(REAL_FRAMES),
            "required_halo_core_4": model.required_halo(REAL_CORE),
            "required_halo_default": model.required_halo(),
            "raw_abs_max": float(full_raw.abs().max()),
            "raw_samples_beyond_unit": int((full_raw.abs() > 1).sum()),
            "left_rms": float(left.pow(2).mean().sqrt()),
            "right_rms": float(right.pow(2).mean().sqrt()),
            "python_tiled_core4_vs_full_max_abs": max_abs(tiled_raw, full_raw),
            "python_long_tiled_core16_vs_full_max_abs": max_abs(long_tiled, long_full),
            "python_pipeline_default_vs_full_max_abs": float(
                np.abs(pipe_default - pipe_full).max()),
        }
        del model
    ref = out / "vae_real_reference.safetensors"
    save_file(tensors, str(ref))
    meta = {
        "story": "sc-22993",
        "producer": "scripts/reference/yue2/vae_reference.py real",
        "reference": environment(),
        "frames": REAL_FRAMES,
        "core_frames": REAL_CORE,
        "long_frames": REAL_LONG_FRAMES,
        "long_core_frames": REAL_LONG_CORE,
        "long_latent": "torch.randn((1,64,75), Generator.manual_seed(2299375)) * std(latent)",
        "clip": "LEFT 0.97*sign(sin(2pi*220t)), RIGHT 0.3*sin(2pi*659.25t), 48 kHz, "
                f"{REAL_FRAMES * 1920} samples",
        "reference_file": ref.name,
        "reference_sha256": sha256_file(ref),
        "latent_sha256": sha256_bytes(tensors["latent"].numpy().astype("<f4").tobytes()),
        "decoders": record,
    }
    (FIXTURE_DIR / "vae_real_reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", nargs="?", default="tiny", choices=["tiny", "real"])
    parser.add_argument("--hub", type=Path, default=os.environ.get("YUE2_HF_HUB"))
    parser.add_argument("--out", type=Path,
                        default=Path.home() / ".cache/sceneworks-yue2-fixtures/vae")
    args = parser.parse_args()
    if args.mode == "tiny":
        tiny()
    else:
        if args.hub is None:
            sys.exit("set YUE2_HF_HUB or pass --hub")
        real(args.hub, args.out)


if __name__ == "__main__":
    main()
