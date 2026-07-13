"""Dump a golden for the SVD EDM scheduler (epic 3040 / sc-3371) from the real diffusers
`EulerDiscreteScheduler` configured exactly like the SVD checkpoint. Validates the Rust
`EdmSchedule::karras` + `scale_model_input` + `v_pred_denoised` + `euler_step` byte-close.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_scheduler_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_scheduler_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler
from safetensors.numpy import save_file

from _paths import fixture

N = 25
sched = EulerDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="v_prediction",
    timestep_type="continuous",
    use_karras_sigmas=True,
    sigma_min=0.002,
    sigma_max=700.0,
    steps_offset=1,
    timestep_spacing="leading",  # the SVD checkpoint's spacing (affects init_noise_sigma)
)
sched.set_timesteps(N)
sigmas = sched.sigmas.cpu().numpy().astype(np.float32)  # len N+1
timesteps = sched.timesteps.cpu().numpy().astype(np.float32)  # len N

# One step at index 5: scale_model_input → (synthetic) v-pred model output → step.
rng = np.random.default_rng(3371)
shape = (2, 4, 8, 8)
x = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
v = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
step_idx = 5
t = sched.timesteps[step_idx]
scaled = sched.scale_model_input(x.clone(), t)  # sets internal step_index
out = sched.step(v, t, x.clone())
prev = out.prev_sample
pred_x0 = out.pred_original_sample

tensors = {
    "sigmas": sigmas,
    "timesteps": timesteps,
    "x": x.numpy().astype(np.float32),
    "v": v.numpy().astype(np.float32),
    "scaled": scaled.numpy().astype(np.float32),
    "pred_x0": pred_x0.numpy().astype(np.float32),
    "prev": prev.numpy().astype(np.float32),
    "init_noise_sigma": np.array([float(sched.init_noise_sigma)], dtype=np.float32),
}
meta = {"n": str(N), "step_idx": str(step_idx), "sigma_at_step": str(float(sched.sigmas[step_idx]))}

out_path = fixture("mlx-gen-svd/tests/fixtures/svd_scheduler_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path, metadata=meta)
print(f"wrote {out_path}")
print("  sigmas[0,24,25]:", sigmas[0], sigmas[24], sigmas[25])
print("  timesteps[0,24]:", timesteps[0], timesteps[24])
print("  init_noise_sigma:", float(sched.init_noise_sigma), " sigma@5:", float(sched.sigmas[step_idx]))
