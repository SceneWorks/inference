#!/usr/bin/env python3
"""Regenerate the YuE2 autoregressive-stage parity fixtures (sc-22991, epic sc-22988).

`crates/audio/candle-audio-yue2` checks its native MoT backbone, sampler and generation loop
against this script's output. Everything is produced by the **pinned upstream** code imported
from the reference environment (``import yue2`` — github.com/multimodal-art-projection/YuE @
92a73cc7652fcc1f937855e4b765e0a0edd7ff2e): ``yue2.modeling_yue2.YuE2ForCausalLM``,
``yue2.sampling.generate_tokens`` / ``distribution`` and ``yue2.protocol.token_prefixes`` /
``negative_prefix``. Nothing here re-implements the reference.

Subcommands
-----------
``synthetic``  (no weights; seconds) writes

* ``tests/fixtures/ar_synthetic.json`` — upstream ``generate_tokens`` on a tiny-width
  ``YuE2ForCausalLM`` (real architecture, full 184 704-id vocabulary) whose weights are an integer
  hash of (tensor name, element index); the Rust lib tests rebuild the identical weights
  (``model::synthetic``) so no checkpoint is committed. Greedy, CFG, ABC, ``cot = off`` legacy and
  injected-draw stochastic decodes, plus a prefix longer than one Rust prefill chunk.
* ``tests/fixtures/sampler.json`` — upstream ``distribution`` / the CFG line on fixed logits
  rows (same hash), in F32 and in the BF16 ``legacy_off`` arithmetic, recorded as SHA-256 digests of
  the exact output bytes plus the surviving ids for diagnosis.

``real``  (YuE2-3B, F32 on the CPU; minutes) writes ``tests/fixtures/ar_real_weights.json``: the
exact prompt/negative token ids of every mode (``cot`` full / melody / off, a supplied full score, a
supplied melody score) built by the upstream tokenizer and protocol, and bounded upstream decodes
from them with per-step logit summaries. Expected peak RSS ~16 GB (F32 weights of both MoT paths
plus the mmapped checkpoint); run it alone, under an RSS guard.

The random draw
---------------
Torch's generator cannot be reproduced natively (and a seed is no cross-platform bit-exact
guarantee, epic E9), so for stochastic decodes ``torch.multinomial`` is replaced — only while
``generate_tokens`` runs — by the inverse-CDF draw the Rust sampler makes: the first id, in
vocabulary order, whose float64 cumulative probability exceeds ``u * total``, with ``u`` popped
from a recorded list of uniforms. Every other line of upstream sampling runs unmodified.

Usage (the shared reference environment, see README.md)::

    ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/ar_fixtures.py synthetic
    HF_HUB_OFFLINE=1 YUE2_HF_HUB=/Volumes/Models/huggingface/hub \\
        ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/ar_fixtures.py real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import yue2
import yue2.sampling as ysampling
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.protocol import (
    ABC_END, ABC_START, CODEC_OFFSET, CODEC_SIZE, EOD, INSTRUCTIONS, MUSIC_END, MUSIC_START,
    VOCAB_SIZE, Sampling, SongRequest, negative_prefix, token_prefixes,
)

UPSTREAM_COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"
LM_REPO, LM_REVISION = "m-a-p/YuE2-3B", "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "crates/audio/candle-audio-yue2/tests/fixtures"
MASK32 = 0xFFFFFFFF


# ── deterministic values (mirrors `model::synthetic` in the Rust crate) ──────────────────────


def fnv1a(name: str) -> int:
    h = 0x811C9DC5
    for b in name.encode():
        h = ((h ^ b) * 0x01000193) & MASK32
    return h


def _hash(name: str, n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.uint64)
    x = (np.uint64(fnv1a(name)) ^ ((i * np.uint64(0x9E3779B1)) & np.uint64(MASK32))) & np.uint64(MASK32)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(MASK32)
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(MASK32)
    x ^= x >> np.uint64(16)
    return x


def values(name: str, n: int, scale: float, offset: float) -> np.ndarray:
    """``offset + ((h % 2001) - 1000) * (scale / 1000)`` in float32 — `synthetic::values`."""
    step = np.float32(scale) / np.float32(1000.0)
    k = (_hash(name, n) % np.uint64(2001)).astype(np.float32)
    return (np.float32(offset) + (k - np.float32(1000.0)) * step).astype(np.float32)


def row_values(name: str, n: int, scale: float) -> np.ndarray:
    """``((h % 2000001) - 1000000) * (scale / 1e6)`` in float32 — `synthetic::row_values`."""
    step = np.float32(scale) / np.float32(1_000_000.0)
    k = (_hash(name, n) % np.uint64(2_000_001)).astype(np.float32)
    return ((k - np.float32(1_000_000.0)) * step).astype(np.float32)


def proj(fan_in: int) -> float:
    return float(np.float32(1.7) / np.sqrt(np.float32(fan_in)))


def synthetic_config() -> YuE2Config:
    return YuE2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, intermediate_size=64,
                      vocab_size=VOCAB_SIZE, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                      max_position_embeddings=24576, max_latent_frames=64)


def synthetic_state_dict(cfg: YuE2Config) -> dict[str, tuple[list[int], float, float]]:
    h, hd, nq, nkv, i, v = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads,
                            cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size)
    out = {
        "model.embed_tokens.weight": ([v, h], 1.0, 0.0),
        "model.norm.weight": ([h], 0.25, 1.0),
        "lm_head.weight": ([v, h], proj(h), 0.0),
    }
    for layer in range(cfg.num_hidden_layers):
        for attn, norm_a, norm_m, mlp in (
            ("self_attn", "input_layernorm", "post_attention_layernorm", "mlp"),
            ("nar_self_attn", "nar_input_layernorm", "nar_pre_mlp_layernorm", "nar_mlp"),
        ):
            p = f"model.layers.{layer}"
            out[f"{p}.{norm_a}.weight"] = ([h], 0.25, 1.0)
            out[f"{p}.{norm_m}.weight"] = ([h], 0.25, 1.0)
            out[f"{p}.{attn}.q_proj.weight"] = ([nq * hd, h], proj(h), 0.0)
            out[f"{p}.{attn}.k_proj.weight"] = ([nkv * hd, h], proj(h), 0.0)
            out[f"{p}.{attn}.v_proj.weight"] = ([nkv * hd, h], proj(h), 0.0)
            out[f"{p}.{attn}.o_proj.weight"] = ([h, nq * hd], proj(nq * hd), 0.0)
            out[f"{p}.{attn}.q_norm.weight"] = ([hd], 0.25, 1.0)
            out[f"{p}.{attn}.k_norm.weight"] = ([hd], 0.25, 1.0)
            out[f"{p}.{mlp}.gate_proj.weight"] = ([i, h], proj(h), 0.0)
            out[f"{p}.{mlp}.up_proj.weight"] = ([i, h], proj(h), 0.0)
            out[f"{p}.{mlp}.down_proj.weight"] = ([h, i], proj(i), 0.0)
    return out


def synthetic_model() -> YuE2ForCausalLM:
    cfg = synthetic_config()
    torch.manual_seed(0)
    model = YuE2ForCausalLM(cfg).eval().float()
    state = model.state_dict()
    for name, (shape, scale, offset) in synthetic_state_dict(cfg).items():
        n = int(np.prod(shape))
        tensor = torch.from_numpy(values(name, n, scale, offset).reshape(shape))
        if tuple(state[name].shape) != tuple(shape):
            raise SystemExit(f"{name}: upstream shape {tuple(state[name].shape)} != {shape}")
        state[name] = tensor
    model.load_state_dict(state)
    return model


# ── the injected draw and the step recorder ─────────────────────────────────────────────────


def splitmix_uniforms(seed: int, n: int) -> list[float]:
    """candle-llm `SplitMix64::next_f32` (24-bit mantissa), as exact float32 values."""
    out, state = [], seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        out.append(float(np.float32((z >> 40) / float(1 << 24))))
    return out


class stable_sort:
    """Force ``Tensor.sort`` stable while upstream sampling runs.

    Upstream's top-p sorts with ``torch.sort(descending=True)`` (``stable=False``), which leaves
    the order of **equal** scores to the backend: CUDA's radix sort keeps ascending ids (the
    release platform), the CPU's introsort does not. The order only decides which of a group of
    equal scores straddling the top-p cut is removed. The native sampler uses ascending ids; the
    fixtures are generated the same way so they compare bit for bit.
    """

    def __enter__(self):
        self.original = torch.Tensor.sort

        def sort(tensor, *args, **kwargs):
            kwargs["stable"] = True
            return self.original(tensor, *args, **kwargs)

        torch.Tensor.sort = sort

    def __exit__(self, *exc):
        torch.Tensor.sort = self.original


class Recorder:
    """Wraps upstream `distribution` (records the logits row each step samples from) and replaces
    `torch.multinomial` with the recorded inverse-CDF draw, only inside `run`."""

    def __init__(self, summarize):
        self.summarize = summarize

    def run(self, fn, uniforms=None):
        rows, tokens, margins = [], [], []
        queue = list(uniforms or [])
        original_distribution = ysampling.distribution
        original_multinomial = torch.multinomial

        def distribution(logits, sampling, history, step, phase, legacy_off=False):
            rows.append(self.summarize(logits.detach().float().reshape(-1), phase))
            return original_distribution(logits, sampling, history, step, phase, legacy_off)

        def multinomial(probabilities, num_samples, generator=None):
            if not queue:
                raise SystemExit("stochastic decode ran out of injected uniforms")
            u = queue.pop(0)
            p = probabilities.detach().double().reshape(-1)
            cdf = p.cumsum(0)
            target = u * float(cdf[-1])
            hit = int(torch.nonzero((cdf > target) & (p > 0))[0])
            lower = float(cdf[hit - 1]) if hit else 0.0
            margins.append(min(target - lower, float(cdf[hit]) - target) / float(cdf[-1]))
            return torch.tensor([[hit]], device=probabilities.device)

        ysampling.distribution = distribution
        torch.multinomial = multinomial
        try:
            with stable_sort():
                history, timing, truncated = fn(lambda phase, token: tokens.append(int(token)))
        finally:
            ysampling.distribution = original_distribution
            torch.multinomial = original_multinomial
        used = len(uniforms or []) - len(queue)
        return {"tokens": [int(t) for t in history], "emitted": tokens,
                "truncated": bool(truncated), "uniforms": list(uniforms or [])[:used],
                "draw_margins": margins, "steps": rows}


def sampling_dict(s: Sampling) -> dict:
    return {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k,
            "repetition_penalty": s.repetition_penalty, "penalty_window": s.penalty_window,
            "min_tokens": s.min_tokens, "max_tokens": s.max_tokens}


def allowed_mask(phase: str) -> torch.Tensor:
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    if phase == "abc":
        mask[:EOD] = True
        mask[ABC_END] = True
    else:
        mask[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = True
        mask[MUSIC_END] = True
    return mask


def summarize_row(row: torch.Tensor, phase: str, top: int, probes: list[int]) -> dict:
    """What a Rust test compares a native logits row against: the top ids/values over the
    phase's allow mask, fixed probe ids, and float64 moments of the whole allowed row."""
    mask = allowed_mask(phase)
    allowed = row[mask].double()
    ids = torch.nonzero(mask).reshape(-1)
    vals, idx = row[mask].topk(top)
    second = float(vals[1]) if top > 1 else float("nan")
    return {
        "top_ids": [int(ids[i]) for i in idx],
        "top_values": [float(v) for v in vals],
        "top_margin": float(vals[0]) - second,
        "probe_values": [float(row[p]) for p in probes],
        "sum": float(allowed.sum()),
        "sumsq": float((allowed * allowed).sum()),
        "logsumexp": float(torch.logsumexp(allowed, 0)),
    }


def probe_ids() -> dict[str, list[int]]:
    """Fixed probe ids per phase: spread over the allow range, the end id included."""
    abc = [0, 13, 220, 1000, 4096, 12345, 30000, 65000, 100000, 151642, ABC_END]
    sem = [CODEC_OFFSET + k for k in (0, 1, 777, 4096, 10000, 16384, 24000, 32767)] + [MUSIC_END]
    return {"abc": abc, "semantic": sem}


def env_record() -> dict:
    ref = Path(os.environ.get("YUE2_REF_DIR", Path.home() / ".cache/sceneworks-yue2-ref"))
    env = json.loads((ref / "ENVIRONMENT.json").read_text()) if (ref / "ENVIRONMENT.json").exists() else {}
    return {"upstream_commit": UPSTREAM_COMMIT, "yue2_package": getattr(yue2, "__version__", None),
            "torch": torch.__version__, "numpy": np.__version__,
            "environment_commit": env.get("yue2_commit"), "python": sys.version.split()[0]}


def digest(values_f32: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values_f32, dtype="<f4").tobytes()).hexdigest()


# ── synthetic ────────────────────────────────────────────────────────────────────────────────


def run_generate(recorder, model, prefix, sampling, phase, *, negative=None, cfg=1.0,
                 legacy_off=False, uniforms=None, seed=831001):
    result = recorder.run(lambda on_token: ysampling.generate_tokens(
        model, prefix, sampling, seed, phase, negative=negative, cfg_scale=cfg,
        legacy_off=legacy_off, on_token=on_token, use_cuda_graph=False), uniforms)
    result.update(prefix=list(prefix), phase=phase, sampling=sampling_dict(sampling),
                  negative=negative, cfg_scale=cfg, legacy_off=legacy_off)
    return result


def synthetic(_args) -> None:
    torch.set_num_threads(1)
    model = synthetic_model()
    probes = probe_ids()
    recorder = Recorder(lambda row, phase: summarize_row(row, phase, 4, probes[phase]))
    base = [EOD, 40, 1234, 99, 5, 6, 7]
    abc = [11, 12, 13, 14]
    plan_tail = [ABC_START, *abc, ABC_END, MUSIC_START]
    sem_prefix = base + plan_tail
    negative = [EOD, 40, 1234] + plan_tail
    off_prefix = base + [ABC_START, ABC_END, MUSIC_START]
    off_negative = [EOD, 40, 1234, MUSIC_START]
    long_prefix = [EOD] + [(k * 7919) % EOD for k in range(1, 600)] + plan_tail
    greedy_sem = Sampling(0.0, 0.95, 100, 1.2, 50, 3, 12)
    greedy_abc = Sampling(0.0, 0.9, 30, 1.005, 100, 4, 12)
    stoch_sem = Sampling(1.0, 0.95, 100, 1.2, 50, 4, 12)
    stoch_abc = Sampling(0.7, 0.9, 30, 1.005, 100, 4, 12)
    uniforms = splitmix_uniforms(20260926, 64)
    cases = {
        "semantic_greedy": dict(prefix=sem_prefix, sampling=greedy_sem, phase="semantic"),
        "semantic_greedy_cfg": dict(prefix=sem_prefix, sampling=greedy_sem, phase="semantic",
                                    negative=negative, cfg=1.5),
        "abc_greedy": dict(prefix=base + [ABC_START], sampling=greedy_abc, phase="abc"),
        "off_greedy_legacy_cfg": dict(prefix=off_prefix, sampling=greedy_sem, phase="semantic",
                                      negative=off_negative, cfg=1.01, legacy_off=True),
        "semantic_stochastic_cfg": dict(prefix=sem_prefix, sampling=stoch_sem, phase="semantic",
                                        negative=negative, cfg=2.0, uniforms=uniforms),
        "off_stochastic_legacy": dict(prefix=off_prefix, sampling=stoch_sem, phase="semantic",
                                      negative=off_negative, cfg=1.01, legacy_off=True,
                                      uniforms=uniforms),
        "abc_stochastic": dict(prefix=base + [ABC_START], sampling=stoch_abc, phase="abc",
                               uniforms=uniforms),
        # Natural stops through the real loop: with every allowed id kept (top_k = vocabulary,
        # top_p = 1), u = 0 lands on the lowest allowed id — MUSIC_END in the semantic phase — and
        # u = 1 - 2^-24 on the highest — ABC_END in the ABC phase — once min_tokens allows it.
        "semantic_natural_end": dict(prefix=sem_prefix, phase="semantic",
                                     sampling=Sampling(1.0, 1.0, VOCAB_SIZE, 1.2, 50, 2, 12),
                                     uniforms=[0.5, 0.5, 0.0]),
        "abc_natural_end": dict(prefix=base + [ABC_START], phase="abc",
                                sampling=Sampling(1.0, 1.0, VOCAB_SIZE, 1.005, 100, 1, 12),
                                uniforms=[0.25, 1.0 - 2.0 ** -24]),
        "long_prefix_greedy": dict(prefix=long_prefix, sampling=replace(greedy_sem, max_tokens=4,
                                                                         min_tokens=0),
                                   phase="semantic"),
    }
    out = {"generator": "scripts/reference/yue2/ar_fixtures.py synthetic",
           "reference": env_record(), "probes": probes, "cases": {}}
    for name, case in cases.items():
        kwargs = {k: case[k] for k in ("negative", "cfg", "legacy_off", "uniforms") if k in case}
        result = run_generate(recorder, model, case["prefix"], case["sampling"], case["phase"], **kwargs)
        out["cases"][name] = result
        print(f"{name}: {len(result['tokens'])} tokens, truncated={result['truncated']}, "
              f"min top margin={min(s['top_margin'] for s in result['steps']):.3g}, "
              f"min draw margin={min(result['draw_margins'], default=float('nan')):.3g}")
    write(FIXTURES / "ar_synthetic.json", out)
    write(FIXTURES / "sampler.json", sampler_cases())


def sampler_cases() -> dict:
    """Upstream `distribution` / CFG on fixed rows, digested exactly."""
    rows = {k: row_values(f"sampler/row{k}", VOCAB_SIZE, 8.0) for k in range(3)}
    history = [CODEC_OFFSET + k for k in (5, 5, 5, 9, 9, 100)] + [17, 17, 3]
    # Make the repeated ids competitive so the penalty moves the head of the distribution.
    for row in rows.values():
        for t in history:
            row[t] = np.float32(7.99)
    specs = [
        ("semantic_default", "semantic", Sampling(), 5, False),
        ("semantic_min_tokens", "semantic", Sampling(), 0, False),
        ("abc_default", "abc", Sampling(.7, .9, 30, 1.005, 100, 32, 4096), 40, False),
        ("semantic_greedy_penalty", "semantic", Sampling(0.0, 1.0, 1, 1.2, 50, 0, 10), 3, False),
        ("semantic_wide_top_p", "semantic", Sampling(1.3, 0.5, 5000, 1.1, 4, 0, 10), 3, False),
        ("off_legacy_default", "semantic", Sampling(), 5, True),
        ("off_legacy_tiny_top_p", "semantic", Sampling(1.0, 0.01, 100, 1.2, 50, 0, 10), 3, True),
        ("off_legacy_wide", "semantic", Sampling(0.9, 0.97, 3000, 1.3, 8, 0, 10), 9, True),
    ]
    out = {"generator": "scripts/reference/yue2/ar_fixtures.py synthetic",
           "reference": env_record(), "history": history, "distribution": [], "cfg": []}
    for k, row in rows.items():
        for name, phase, sampling, step, legacy in specs:
            for dtype in ("f32", "bf16"):
                if legacy is False and dtype == "bf16":
                    # Non-legacy upcasts: a BF16 row is sampled in F32 (covered by the F32 case).
                    continue
                t = torch.from_numpy(row.copy())[None]
                if dtype == "bf16":
                    t = t.to(torch.bfloat16)
                with stable_sort():
                    scores = ysampling.distribution(t, sampling, history, step, phase, legacy)
                probs = scores.softmax(-1)
                s = scores.float().reshape(-1).numpy()
                p = probs.float().reshape(-1).numpy()
                finite = np.nonzero(np.isfinite(s))[0]
                out["distribution"].append({
                    "row": k, "case": name, "phase": phase, "dtype": dtype, "step": step,
                    "legacy_off": legacy, "sampling": sampling_dict(sampling),
                    "scores_sha256": digest(s), "probabilities_sha256": digest(p),
                    "finite_ids": [int(i) for i in finite[:64]], "finite_count": int(len(finite)),
                    "finite_scores": [float(s[i]) for i in finite[:64]],
                    "finite_probabilities": [float(p[i]) for i in finite[:64]],
                    "argmax": int(np.argmax(s)),
                })
    out["ops"] = model_ops()
    for dtype in ("f32", "bf16"):
        for scale in (1.01, 1.5, 3.0, 0.0):
            c = torch.from_numpy(rows[0].copy())
            u = torch.from_numpy(rows[1].copy())
            if dtype == "bf16":
                c, u = c.to(torch.bfloat16), u.to(torch.bfloat16)
            mixed = u + scale * (c - u)
            out["cfg"].append({"dtype": dtype, "scale": scale,
                               "sha256": digest(mixed.float().numpy())})
    return out


def model_ops() -> list[dict]:
    """Upstream `RMSNorm` and `project_qkv`'s RoPE (`RotaryEmbedding` + `_apply_rotary`) on fixed
    `[1, 7, 16, 128]` inputs at cache positions 300..307, in F32 and BF16 — the two leaves whose
    rounding sequence the native port reproduces rather than reuses. (CPU Candle has BF16
    elementwise ops but no BF16 matmul, so these leaves are the BF16 arithmetic testable here.)"""
    from yue2.modeling_yue2 import RMSNorm, RotaryEmbedding, _apply_rotary

    shape, n = (1, 7, 16, 128), 7 * 16 * 128
    x = torch.from_numpy(row_values("ops/x", n, 4.0).reshape(shape))
    w = torch.from_numpy(values("ops/w", 128, 0.25, 1.0))
    positions = torch.arange(300, 307)[None]
    cos, sin = RotaryEmbedding(128, 1_000_000.0)(positions)
    out = []
    for dtype_name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        norm = RMSNorm(128, 1e-6).to(dtype)
        with torch.no_grad():
            norm.weight.copy_(w.to(dtype))
            normed = norm(x.to(dtype))
            rotated = _apply_rotary(x.to(dtype), cos.unsqueeze(2), sin.unsqueeze(2))
        for op, result in (("rms_norm", normed), ("rope", rotated)):
            flat = result.float().reshape(-1)
            out.append({"op": op, "dtype": dtype_name, "start": 300,
                        "sha256": digest(flat.numpy()),
                        "head": [float(v) for v in flat[:512]],
                        "max_abs": float(flat.abs().max())})
    return out


# ── real weights ─────────────────────────────────────────────────────────────────────────────


def snapshot_dir() -> Path:
    hub = os.environ.get("YUE2_HF_HUB")
    if not hub:
        raise SystemExit("set YUE2_HF_HUB to the hub directory holding models--m-a-p--YuE2-3B")
    path = Path(hub) / "models--m-a-p--YuE2-3B" / "snapshots" / LM_REVISION
    manifest = json.loads((path / "weights_manifest.json").read_text())
    weights = path / "model.safetensors"
    want = manifest["files"]["model.safetensors"]
    if weights.stat().st_size != want["bytes"]:
        raise SystemExit("model.safetensors size disagrees with the pinned manifest")
    h = hashlib.sha256()
    with weights.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != want["sha256"]:
        raise SystemExit("model.safetensors hash disagrees with the pinned manifest")
    return path


def real(args) -> None:
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    snap = snapshot_dir()
    upstream = Path(os.environ.get("YUE2_REF_DIR", Path.home() / ".cache/sceneworks-yue2-ref")) / "YuE"
    song = json.loads((upstream / "examples/song.json").read_text())
    score = (upstream / "examples/score.abc").read_text()
    melody = (upstream / "examples/melody.abc").read_text()
    tok = YuE2TextTokenizer(snap / "qwen.tiktoken")
    start = time.perf_counter()
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True,
                                            dtype=torch.float32).eval()
    load_seconds = time.perf_counter() - start
    probes = probe_ids()
    recorder = Recorder(lambda row, phase: summarize_row(row, phase, 8, probes[phase]))
    style, lyrics = song["style"], song["lyrics"]
    requests = {
        "full": SongRequest(style, lyrics, cot="full"),
        "melody": SongRequest(style, lyrics, cot="melody", cfg_scale=1.5),
        "off": SongRequest(style, lyrics, cot="off"),
        "supplied_full": SongRequest(style, lyrics, cot="full", abc=score, cfg_scale=2.0),
        "supplied_melody": SongRequest(style, lyrics, cot="melody", abc=melody),
    }
    abc_steps, sem_steps = args.abc_steps, args.semantic_steps
    greedy_abc = Sampling(0.0, .9, 30, 1.005, 100, min(32, abc_steps // 2), abc_steps)
    greedy_sem = Sampling(0.0, .95, 100, 1.2, 50, sem_steps, sem_steps)
    uniforms = splitmix_uniforms(831001, 4 * max(abc_steps, sem_steps))
    out = {"generator": "scripts/reference/yue2/ar_fixtures.py real",
           "reference": env_record(), "weights": {"repo": LM_REPO, "revision": LM_REVISION,
                                                  "dtype": "float32 (BF16 checkpoint upcast)"},
           "probes": probes, "modes": {}}
    timings = {}
    for mode, request in requests.items():
        rec = {"cot": request.cot, "cfg_scale": request.guidance}
        t0 = time.perf_counter()
        if request.cot != "off" and request.abc is None:
            planner = token_prefixes(request, tok)
            rec["planner_prefix"] = planner
            rec["abc"] = run_generate(recorder, model, planner, greedy_abc, "abc")
            abc_ids = rec["abc"]["tokens"]
        elif request.abc is not None:
            abc_ids = tok.encode(request.abc)
        else:
            abc_ids = []
        rec["abc_ids"] = abc_ids
        prefix = token_prefixes(request, tok, abc_ids)
        negative = negative_prefix(request, tok, abc_ids) if request.guidance != 1 else None
        rec["semantic_prefix"], rec["negative_prefix"] = prefix, negative
        rec["semantic"] = run_generate(recorder, model, prefix, greedy_sem, "semantic",
                                       negative=negative, cfg=request.guidance,
                                       legacy_off=request.cot == "off")
        # Full recompute (no cache) of prefix + generated tokens: the last position's logits must
        # equal the cached decode's next-step logits; recorded for the Rust side to compare.
        seq = prefix + rec["semantic"]["tokens"][:-1]
        with torch.inference_mode():
            full = model(torch.tensor([seq]), use_cache=False, logits_to_keep=1).logits[0, -1]
        rec["semantic"]["full_recompute_last"] = summarize_row(full.float(), "semantic", 8,
                                                               probes["semantic"])
        timings[mode] = time.perf_counter() - t0
        out["modes"][mode] = rec
        print(f"{mode}: prefix {len(prefix)}, abc {len(abc_ids)}, semantic "
              f"{len(rec['semantic']['tokens'])} ({timings[mode]:.1f}s)", flush=True)

    # A planner that reaches ABC_END by itself: the supplied melody score teacher-forced after
    # ABC_START, then greedy with no min_tokens (the model closes the voices and ends the score).
    planner_prefix = token_prefixes(SongRequest(style, lyrics, cot="melody"), tok) + tok.encode(melody)
    out["abc_natural_end"] = {"planner_prefix": planner_prefix, **run_generate(
        recorder, model, planner_prefix, Sampling(0.0, .9, 30, 1.005, 100, 0, 24), "abc")}
    if out["abc_natural_end"]["truncated"]:
        raise SystemExit("abc_natural_end no longer reaches ABC_END within its budget")
    # Stochastic decodes with injected draws: the released ABC controls, and cot = off's
    # released semantic controls (legacy arithmetic, default guidance 1.01).
    full_planner = out["modes"]["full"]["planner_prefix"]
    out["abc_stochastic"] = {"planner_prefix": full_planner, **run_generate(
        recorder, model, full_planner, Sampling(.7, .9, 30, 1.005, 100, abc_steps // 2, abc_steps),
        "abc", uniforms=uniforms)}
    off = out["modes"]["off"]
    out["off_stochastic"] = {"semantic_prefix": off["semantic_prefix"],
                             "negative_prefix": off["negative_prefix"], **run_generate(
        recorder, model, off["semantic_prefix"],
        Sampling(1.0, .95, 100, 1.2, 50, sem_steps, sem_steps), "semantic",
        negative=off["negative_prefix"], cfg=1.01, legacy_off=True, uniforms=uniforms)}
    out["sampling"] = {"abc_greedy": sampling_dict(greedy_abc),
                       "semantic_greedy": sampling_dict(greedy_sem)}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["run"] = {"load_seconds": load_seconds, "mode_seconds": timings,
                  "peak_rss_gb": peak / (1 << 30 if sys.platform == "darwin" else 1 << 20)}
    print(json.dumps(out["run"]))
    write(FIXTURES / "ar_real_weights.json", out)


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic").set_defaults(fn=synthetic)
    r = sub.add_parser("real")
    r.add_argument("--abc-steps", type=int, default=32)
    r.add_argument("--semantic-steps", type=int, default=24)
    r.set_defaults(fn=real)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
