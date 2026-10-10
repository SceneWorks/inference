"""Dump the Qwen-Image 2.1 flow-match schedule table for the Rust scheduler parity test (sc-24108).

For every upstream preset (and the two tiny e2e geometries) this runs the frozen
`FlowMatchEulerDiscreteScheduler` exactly the way `QwenImage21Pipeline.__call__` does — `sigmas =
linspace(1, 1/N, N)`, `mu = calculate_shift(tokens, base_image_seq_len, max_image_seq_len,
base_shift, max_shift)`, `set_timesteps(N, sigmas=sigmas, mu=mu)` — with the pinned
`scheduler_config.json` values, and records `sigmas` (N + 1, trailing 0), `timesteps` (N) and `mu`.

Run with the pinned venv: `python3.12 tools/dump_qwen21_scheduler.py`
Output: `tests/fixtures/qwen21_scheduler.safetensors`.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

from _qwen21_common import FIXTURE_DIR, PRESETS, SCHEDULER_CONFIG, save_safetensors

VAE_SCALE_FACTOR = 16

CASES = [(f"preset_{name.replace(':', 'x')}", w, h, 40) for name, (w, h) in PRESETS.items()]
CASES += [
    ("preset_1x1_8", 2048, 2048, 8),
    ("tiny_32x32_3", 32, 32, 3),
    ("tiny_64x32_2", 64, 32, 2),
    ("tiny_1024x1024_2", 1024, 1024, 2),
]


def main() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)
    out, meta = {}, {}
    for name, width, height, steps in CASES:
        tokens = (height // VAE_SCALE_FACTOR) * (width // VAE_SCALE_FACTOR)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            tokens,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
        out[f"{name}/sigmas"] = scheduler.sigmas.clone()
        out[f"{name}/timesteps"] = scheduler.timesteps.clone()
        out[f"{name}/mu"] = torch.tensor([mu], dtype=torch.float32)
        meta[f"{name}/width"] = width
        meta[f"{name}/height"] = height
        meta[f"{name}/steps"] = steps
        print(f"{name}: tokens={tokens} mu={mu:.6f} sigmas[:3]={scheduler.sigmas[:3].tolist()} last={scheduler.sigmas[-2:].tolist()}")
    save_safetensors(FIXTURE_DIR / "qwen21_scheduler.safetensors", out, meta)


if __name__ == "__main__":
    main()
