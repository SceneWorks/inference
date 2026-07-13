"""Dump flow-match Euler scheduler parity fixtures from the frozen mflux fork.

Run from the fork:  cd ~/repos/mflux && uv run python tools/dump_scheduler.py

Uses the fork's OWN `_compute_empirical_mu` + `_time_shift_exponential_array` as the oracle,
then reconstructs the sigmas exactly as the scheduler does (linspace(1, 1/n, n) -> time-shift ->
append 0). Also dumps one Euler step example. The Rust port recomputes independently and asserts
allclose.
"""

import mlx.core as mx
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)

from _paths import fixture

OUT = fixture("tests/fixtures/scheduler.safetensors")


def build(num_steps: int, seq_len: int):
    mu = S._compute_empirical_mu(seq_len, num_steps)
    sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps, dtype=mx.float32)
    sigmas = S._time_shift_exponential_array(mu, 1.0, sigmas)
    sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
    return float(mu), sigmas


tensors = {}
meta = {}

# (num_steps, width, height) — covers turbo default (4 @ 1024), tiny, large-seq branch, n=1.
cfgs = [(4, 1024, 1024), (4, 256, 256), (8, 1280, 1280), (1, 512, 512)]
meta["num_cfgs"] = str(len(cfgs))
for i, (ns, w, h) in enumerate(cfgs):
    seq = (h // 16) * (w // 16)
    mu, sig = build(ns, seq)
    tensors[f"sigmas_{i}"] = sig
    meta[f"cfg_{i}"] = f"{ns},{w},{h},{seq},{mu!r}"

# One Euler step at the turbo 1024 schedule, t=1: out = latents + (sigma[t+1]-sigma[t]) * noise
mx.random.seed(0)
latents = mx.random.normal((2, 3, 4)).astype(mx.float32)
noise = mx.random.normal((2, 3, 4)).astype(mx.float32)
mu, sig = build(4, 4096)
t = 1
out = latents + (sig[t + 1] - sig[t]) * noise
tensors["step_latents"] = latents
tensors["step_noise"] = noise
tensors["step_out"] = out
meta["step"] = f"4,1024,1024,{t},{mu!r}"

mx.save_safetensors(OUT, tensors, meta)
print(f"wrote {OUT}: {sorted(tensors)} meta={meta}")
