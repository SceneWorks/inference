#!/usr/bin/env python3
"""Regenerate the YuE2 acoustic (NAR) parity fixtures (sc-22992, epic sc-22988).

`crates/audio/candle-audio-yue2`'s `nar` module is checked against this script's output. Every
latent here is produced by the **pinned upstream** code imported from the reference environment
(``import yue2`` — github.com/multimodal-art-projection/YuE @
92a73cc7652fcc1f937855e4b765e0a0edd7ff2e): ``yue2.nar.synthesize`` / ``song_chunks`` /
``CachedNAR`` and ``yue2.modeling_yue2.YuE2ForCausalLM.nar_velocity``. Nothing here
re-implements the reference; the only instrumentation is a wrapper around
``CachedNAR.velocity`` that records every evaluation's input state, raw timestep and output.

Subcommands
-----------
``synthetic``  (no weights; seconds) writes ``tests/fixtures/nar_synthetic.json`` (cases, token
ids, chunks) and ``tests/fixtures/nar_synthetic.safetensors`` (the injected noise, every
evaluation's input / raw ``t`` / velocity, and the final latents) for the tiny-width synthetic
MoT of ``ar_fixtures.synthetic_model`` plus NAR heads whose weights are the same integer hash
(``synthetic_heads``; the Rust side rebuilds them bit-identically in ``nar::synthetic``).

``real``  (YuE2-3B, F32 on the CPU; minutes) writes the upstream reference for the real weights to
``$YUE2_NAR_REFERENCE_DIR`` (default ``~/.cache/sceneworks-yue2-fixtures/nar``) — **outside the
repository**, because latents computed by the CC BY-NC 4.0 weights are derived from them — and
commits only ``tests/fixtures/nar_real_reference.json`` (cases, token ids, and the SHA-256 of the
reference file). Expected peak RSS ~21 GB (both MoT paths in F32 plus the mapped checkpoint); run
it alone, never beside the native real-weight test.

Usage (the shared reference environment, see README.md)::

    ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/nar_fixtures.py synthetic
    HF_HUB_OFFLINE=1 YUE2_HF_HUB=/Volumes/Models/huggingface/hub \\
        ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/nar_fixtures.py real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar_fixtures as af  # noqa: E402  (the sibling generator: synthetic MoT, env record, snapshot)

import yue2.nar as ynar  # noqa: E402
from yue2.modeling_yue2 import YuE2ForCausalLM  # noqa: E402
from yue2.protocol import (  # noqa: E402
    ABC_END, ABC_START, CODEC_OFFSET, CODEC_SIZE, CONTEXT, EOD, LATENT_END, LATENT_PAD,
    LATENT_START, MUSIC_END, MUSIC_START, chunk_ranges,
)

FIXTURES = af.FIXTURES
SEED = 831001
# The synthetic song prefix: EOD, text ids, a framed score, MUSIC_START (11 tokens).
SYNTHETIC_PREFIX = [EOD, 40, 1234, 99, 5021, ABC_START, 5, 777, 3000, ABC_END, MUSIC_START]


def codes(n: int, name: str = "codes") -> list[int]:
    """Deterministic codec indices in ``[0, CODEC_SIZE)`` (the fixture hash)."""
    return [int(c) for c in af._hash(name, n) % np.uint64(CODEC_SIZE)]


def synthetic_heads(hidden: int, max_latent_frames: int) -> dict[str, tuple[list[int], float, float]]:
    """``(shape, scale, offset)`` of every NAR head tensor — `nar::synthetic::heads_state_dict`."""
    d, f = 64, 256
    return {
        "vae2llm.weight": ([hidden, d], af.proj(d), 0.0),
        "vae2llm.bias": ([hidden], 0.1, 0.0),
        "llm2vae.weight": ([d, hidden], af.proj(hidden), 0.0),
        "llm2vae.bias": ([d], 0.1, 0.0),
        "time_embedder.mlp.0.weight": ([hidden, f], af.proj(f), 0.0),
        "time_embedder.mlp.0.bias": ([hidden], 0.1, 0.0),
        "time_embedder.mlp.2.weight": ([hidden, hidden], af.proj(hidden), 0.0),
        "time_embedder.mlp.2.bias": ([hidden], 0.1, 0.0),
        "latent_pos_embed.pe": ([max_latent_frames, hidden], 1.0, 0.0),
    }


def synthetic_nar_model() -> YuE2ForCausalLM:
    model = af.synthetic_model()
    cfg = model.config
    state = model.state_dict()
    for name, (shape, scale, offset) in synthetic_heads(cfg.hidden_size, cfg.max_latent_frames).items():
        if tuple(state[name].shape) != tuple(shape):
            raise SystemExit(f"{name}: upstream shape {tuple(state[name].shape)} != {shape}")
        state[name] = torch.from_numpy(af.values(name, int(np.prod(shape)), scale, offset).reshape(shape))
    model.load_state_dict(state)
    return model.eval()


class Recorder:
    """Wraps ``CachedNAR.velocity`` while a synthesis runs: every evaluation, grouped per engine
    (one engine per original chunk, in creation order)."""

    def __init__(self):
        self.chunks: list[dict] = []

    def __enter__(self):
        self.original = ynar.CachedNAR.velocity
        recorder = self

        def velocity(engine, state, raw_t):
            out = recorder.original(engine, state, raw_t)
            # Tag the engine itself: `id()` values are reused once a chunk's engine is freed.
            if not hasattr(engine, "_fixture_index"):
                engine._fixture_index = len(recorder.chunks)
                recorder.chunks.append({"inputs": [], "raw": [], "velocity": [],
                                        "ar_tokens": list(engine.chunk.ar_tokens),
                                        "visible": engine.visible_length})
            rec = recorder.chunks[engine._fixture_index]
            rec["inputs"].append(state.detach().float().clone())
            rec["raw"].append(float(raw_t))
            rec["velocity"].append(out.detach().float().clone())
            return out

        ynar.CachedNAR.velocity = velocity
        return self

    def __exit__(self, *exc):
        ynar.CachedNAR.velocity = self.original


def song_noise(prefix, codec, seed, context) -> torch.Tensor:
    """The full-song noise upstream draws: the concatenation of ``song_chunks``' chunk views
    (contiguous ranges covering every frame)."""
    return torch.cat([c.noise for c in ynar.song_chunks(prefix, codec, seed, context)], 0)


def record_case(tensors: dict, name: str, rec: Recorder, final: torch.Tensor, noise: torch.Tensor,
                evaluations: bool = True):
    tensors[f"{name}/noise"] = noise.contiguous()
    tensors[f"{name}/final"] = final.float().contiguous()
    for k, c in enumerate(rec.chunks if evaluations else []):
        tensors[f"{name}/c{k}/input"] = torch.stack(c["inputs"]).contiguous()
        tensors[f"{name}/c{k}/raw"] = torch.tensor(c["raw"], dtype=torch.float64)
        tensors[f"{name}/c{k}/velocity"] = torch.stack(c["velocity"]).contiguous()


def run_synthesis(model, tensors, name, prefix, codec, *, steps, context=CONTEXT,
                  query_chunk_size=None, offload_ar=False, evaluations=True) -> dict:
    progress = []
    noise = song_noise(prefix, codec, SEED, context)
    with Recorder() as rec:
        final = ynar.synthesize(model, prefix, codec, SEED, steps=steps, context=context,
                                query_chunk_size=query_chunk_size, offload_ar=offload_ar,
                                on_progress=lambda done, total: progress.append([done, total]))
    record_case(tensors, name, rec, final, noise, evaluations)
    chunks = [[a, b] for a, b in chunk_ranges(len(codec), len(prefix), context)]
    return {"name": name, "prefix": prefix, "codes": codec, "steps": steps, "context": context,
            "chunks": chunks, "ar_tokens": [c["ar_tokens"] for c in rec.chunks],
            "visible": [c["visible"] for c in rec.chunks], "progress": progress,
            "query_chunk_size": query_chunk_size, "offload_ar": offload_ar,
            "timestep_shift": float(model.config.timestep_shift)}


def synthetic(_args) -> None:
    torch.set_num_threads(1)  # a deterministic reduction order for the committed reference
    model = synthetic_nar_model()
    cfg = model.config
    p = SYNTHETIC_PREFIX
    # (context − prefix − 3) // 2 = 8 frames per chunk.
    small = len(p) + 3 + 16
    tensors: dict[str, torch.Tensor] = {}
    cases = []
    with torch.inference_mode():
        cases.append(run_synthesis(model, tensors, "default_32", p, codes(6), steps=32))
        cases.append(run_synthesis(model, tensors, "steps_7", p, codes(6), steps=7))
        cases.append(run_synthesis(model, tensors, "steps_1", p, codes(6), steps=1))
        cases.append(run_synthesis(model, tensors, "multi_chunk", p, codes(21), steps=4,
                                   context=small))
        cases.append(run_synthesis(model, tensors, "chunk_edge_exact", p, codes(16), steps=3,
                                   context=small))
        cases.append(run_synthesis(model, tensors, "chunk_edge_plus_one", p, codes(17), steps=3,
                                   context=small))
        cases.append(run_synthesis(model, tensors, "position_clamp", p, codes(70), steps=1))
        cases.append(run_synthesis(model, tensors, "tiled_q5_offload", p, codes(6), steps=32,
                                   query_chunk_size=5, offload_ar=True, evaluations=False))
        tiled = tensors["tiled_q5_offload/final"]
        untiled = tensors["default_32/final"]
        cases[-1]["upstream_tiled_vs_untiled_max_abs"] = float((tiled - untiled).abs().max())

        # Text-only visibility (Chunk.nar_cond_end): one original chunk solved by CachedNAR.
        chunk = ynar.song_chunks(p, codes(6), SEED)[0]
        chunk.nar_cond_end = 5
        with Recorder() as rec:
            engine = ynar.CachedNAR(model, chunk)
            final = engine.solve(3)
        record_case(tensors, "cond_end_5", rec, final, chunk.noise)
        cases.append({"name": "cond_end_5", "ar_tokens": [chunk.ar_tokens], "steps": 3,
                      "nar_cond_end": 5, "visible": [engine.visible_length],
                      "timestep_shift": float(cfg.timestep_shift)})

        # The released default shift is 1 (the identity); a shifted model exercises the formula.
        cfg.timestep_shift = 3.0
        cases.append(run_synthesis(model, tensors, "shift_3", p, codes(6), steps=4))
        cfg.timestep_shift = 1.0

        # The cached formulation (AR prefill once, NAR attends the cached visible keys) against
        # upstream's joint forward (`nar_velocity`: one sequence, both MoT paths, hybrid mask).
        joint = []
        for cond_end in (0, 5):
            chunk = ynar.song_chunks(p, codes(6), SEED)[0]
            chunk.nar_cond_end = cond_end
            engine = ynar.CachedNAR(model, chunk)
            ar = len(chunk.ar_tokens)
            frames = len(chunk.noise)
            tokens = chunk.ar_tokens + [LATENT_START] + [LATENT_PAD] * frames + [LATENT_END]
            s = len(tokens)
            ar_mask = torch.zeros(1, s, dtype=torch.bool)
            ar_mask[0, :ar] = True
            content = torch.zeros(1, s, dtype=torch.bool)
            content[0, ar + 1:ar + 1 + frames] = True
            state = chunk.noise * 0.5
            for raw in (0.0, 2.5):
                cached = engine.velocity(state, raw)
                full = model.nar_velocity(torch.tensor([tokens]), ar_mask, ~ar_mask, content,
                                          state, raw, nar_cond_end=cond_end)
                name = f"joint/cond{cond_end}/raw{raw}"
                tensors[f"{name}/state"] = state.clone().contiguous()
                tensors[f"{name}/cached"] = cached.float().contiguous()
                tensors[f"{name}/joint"] = full.float().contiguous()
                joint.append({"name": name, "ar_tokens": chunk.ar_tokens, "nar_cond_end": cond_end,
                              "raw_t": raw,
                              "cached_vs_joint_max_abs": float((cached - full).abs().max())})
    meta = {"generator": "scripts/reference/yue2/nar_fixtures.py synthetic",
            "reference": af.env_record(), "seed": SEED,
            "model": {"hidden_size": cfg.hidden_size, "layers": cfg.num_hidden_layers,
                      "max_latent_frames": cfg.max_latent_frames},
            "cases": cases, "joint": joint}
    save_file(tensors, str(FIXTURES / "nar_synthetic.safetensors"))
    print(f"wrote {FIXTURES / 'nar_synthetic.safetensors'} "
          f"({(FIXTURES / 'nar_synthetic.safetensors').stat().st_size / 1024:.0f} KiB)")
    af.write(FIXTURES / "nar_synthetic.json", meta)


def reference_dir() -> Path:
    return Path(os.environ.get("YUE2_NAR_REFERENCE_DIR",
                               Path.home() / ".cache/sceneworks-yue2-fixtures/nar"))


def real(args) -> None:
    snap = af.snapshot_dir()
    ar_real = json.loads((FIXTURES / "ar_real_weights.json").read_text(encoding="utf-8"))
    # The exact semantic prefix of a real request (supplied full score, built by the upstream
    # tokenizer and `token_prefixes`, recorded by ar_fixtures.py real).
    prefix = ar_real["modes"]["supplied_full"]["semantic_prefix"]
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True,
                                            dtype=torch.float32).eval()
    load_seconds = time.perf_counter() - start
    tensors: dict[str, torch.Tensor] = {}
    cases = []
    timings = {}
    # A reduced context makes a short song span several original chunks: 48 frames per chunk,
    # so 100 frames are chunks of 48, 48 and 4 (boundary-sized tail).
    multi = len(prefix) + 3 + 2 * 48
    with torch.inference_mode():
        for name, n, steps, context in (("multi_chunk_32", args.frames, 32, multi),
                                        ("single_chunk_5", 24, 5, CONTEXT)):
            t0 = time.perf_counter()
            cases.append(run_synthesis(model, tensors, name, prefix, codes(n, "real_codes"),
                                       steps=steps, context=context))
            timings[name] = time.perf_counter() - t0
            print(f"{name}: {timings[name]:.1f} s")
    out_dir = reference_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nar_real_reference.safetensors"
    save_file(tensors, str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Machine-dependent run cost: printed, never committed.
    print(json.dumps({"load_seconds": load_seconds, "case_seconds": timings,
                      "peak_rss_gb": peak / (1 << 30 if sys.platform == "darwin" else 1 << 20)}))
    af.write(FIXTURES / "nar_real_reference.json", {
        "generator": "scripts/reference/yue2/nar_fixtures.py real",
        "reference": af.env_record(),
        "weights": {"repo": af.LM_REPO, "revision": af.LM_REVISION,
                    "dtype": "float32 (BF16 checkpoint upcast)"},
        "reference_file": {"name": path.name, "sha256": sha, "bytes": path.stat().st_size},
        "seed": SEED, "cases": cases})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic").set_defaults(fn=synthetic)
    r = sub.add_parser("real")
    r.add_argument("--frames", type=int, default=100)
    r.set_defaults(fn=real)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
