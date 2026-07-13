#!/usr/bin/env python3
"""Regenerate the INDEPENDENT golden trajectory for the gen-core er_sde / dpmpp_2m_sde solvers.

    Regenerate with:   python3 gen-core/tests/fixtures/gen_sde_solver_golden.py
    (requires:         pip install torch numpy   -- a throwaway env is fine)

WHY THIS IS A FETCH, NOT A VENDOR
---------------------------------
The numeric reference is ComfyUI's OWN `sample_er_sde` / `sample_dpmpp_2m_sde`. ComfyUI is licensed
GPL-3.0-or-later; gen-core is Apache-2.0. To avoid distributing GPL source inside an Apache-2.0
repository, this generator does NOT commit ComfyUI's code. At regeneration time it FETCHES
    comfy/k_diffusion/sampling.py @ commit b7a648ca2011489ba40eaacf01a5d6f4e9fab539
into memory (never into the repo), verifies its SHA-256 against a pinned digest so a silent upstream
change is caught, extracts the same fixed line ranges, and imports the two samplers + their SNR
helpers from that in-memory copy. Only the COMPUTED numeric output (sde_solver_golden.json) is
committed -- that is data, not ComfyUI's copyrighted expression, so it carries no GPL obligation.

Offline / air-gapped: set COMFY_SAMPLING_PATH=/path/to/sampling.py to a manually downloaded copy of
the exact pinned URL below; it is still checksum-verified.

WHAT IT COMPUTES
----------------
Drives both samplers in the unified VE-sigma space (alpha == 1) the Rust port targets: VE is selected
by handing the reference `sigma_to_half_log_snr` / `offset_first_sigma_for_snr` a `model_sampling`
object that is NOT a `comfy.model_sampling.CONST` instance -> they take the `sigma.log().neg()` branch
(half_log_snr = -log sigma), so er_lambda == sigma and the first-sigma offset is a no-op. Deterministic
mode: s_noise = 0 for both (reproducible, no RNG). eta stays 1 for dpmpp_2m_sde so its SDE contraction
/ midpoint terms are still exercised. Runs in torch float64; the Rust port carries latents in f32, so
the Rust test compares within an f32-appropriate tolerance (~1e-5; measured drift ~1.2e-7).
"""
import hashlib
import json
import os
import sys
import types
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "sde_solver_golden.json"

# --- pinned upstream reference (fetched at regen time, NOT vendored) ----------------------------
UPSTREAM_COMMIT = "b7a648ca2011489ba40eaacf01a5d6f4e9fab539"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/"
    f"{UPSTREAM_COMMIT}/comfy/k_diffusion/sampling.py"
)
UPSTREAM_SHA256 = "cc5f944efd85c566484c3999beb74e8c19c894f2a50ca574d090c3a46ac6bd06"
UPSTREAM_LICENSE = "GPL-3.0-or-later (ComfyUI) -- deliberately NOT vendored into this Apache-2.0 repo"
# 1-based inclusive line ranges within the pinned sampling.py; extracted verbatim into memory only.
RANGES = {
    "sigma_to_half_log_snr": (152, 157),
    "offset_first_sigma_for_snr": (168, 177),
    "sample_dpmpp_2m_sde": (822, 875),
    "sample_er_sde": (1525, 1588),
}


def fetch_reference_source() -> str:
    """Return the pinned sampling.py text (fetched or COMFY_SAMPLING_PATH), checksum-verified."""
    local = os.environ.get("COMFY_SAMPLING_PATH")
    if local:
        data = Path(local).read_bytes()
        origin = f"COMFY_SAMPLING_PATH={local}"
    else:
        try:
            with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001 -- surface a clear, actionable message
            sys.exit(
                f"ERROR: could not fetch the ComfyUI reference from\n    {UPSTREAM_URL}\n"
                f"  ({type(e).__name__}: {e})\n"
                "This generator does NOT vendor the GPL-3.0 source. To regenerate offline, download\n"
                f"that exact URL (pinned commit {UPSTREAM_COMMIT}) by hand, then re-run with\n"
                "    COMFY_SAMPLING_PATH=/path/to/sampling.py python3 gen_sde_solver_golden.py"
            )
        origin = UPSTREAM_URL
    got = hashlib.sha256(data).hexdigest()
    if got != UPSTREAM_SHA256:
        sys.exit(
            f"ERROR: checksum mismatch for the ComfyUI reference ({origin}).\n"
            f"  expected sha256 {UPSTREAM_SHA256}\n  got      sha256 {got}\n"
            "The upstream file changed (or the wrong file was supplied). Re-pin only after review."
        )
    return data.decode("utf-8")


# --- minimal stubs the extracted functions close over ------------------------------------------
comfy = types.ModuleType("comfy")
comfy.model_sampling = types.ModuleType("comfy.model_sampling")


class CONST:  # the discrete-flow marker the SNR helpers special-case; we are NOT this
    pass


comfy.model_sampling.CONST = CONST


class _VEModelSampling:  # non-CONST -> VE branch (half_log_snr = -log sigma), no noise_scale attr
    pass


class _Patcher:
    def __init__(self, ms):
        self._ms = ms

    def get_model_object(self, name):
        assert name == "model_sampling", name
        return self._ms


class _Inner:
    def __init__(self, ms):
        self.model_patcher = _Patcher(ms)


# The toy denoiser: a NON-CONSTANT field varying in BOTH x and sigma, so the multistep Stage-2/3
# (er_sde) and midpoint (dpmpp_2m_sde) finite-difference terms genuinely engage. This exact formula
# is mirrored in the Rust test.  denoised = 0.5*x + 0.25*sin(1.3*x) + 0.2*sigma*cos(0.7*x) - 0.1*sigma
def toy_denoised(x, sigma_scalar):
    return (
        0.5 * x
        + 0.25 * torch.sin(1.3 * x)
        + 0.2 * sigma_scalar * torch.cos(0.7 * x)
        - 0.1 * sigma_scalar
    )


class ToyModel:
    def __init__(self, ms, recorder):
        self.inner_model = _Inner(ms)
        self._recorder = recorder

    def __call__(self, x, sigma, **extra_args):
        # sigma arrives as sigmas[i] * s_in (shape [batch]); reduce to the scalar level.
        sig = float(sigma.reshape(-1)[0])
        self._recorder.append(x.detach().clone().reshape(-1).tolist())
        return toy_denoised(x, sig)


def load_reference_samplers():
    """Fetch + checksum the pinned sampling.py, extract the fixed line ranges into memory, exec."""
    src_lines = fetch_reference_source().splitlines()
    blob = "\n\n".join("\n".join(src_lines[a - 1 : b]) for (a, b) in RANGES.values())
    ns = {
        "torch": torch,
        "trange": lambda n, disable=None: range(n),
        "partial": partial,
        "comfy": comfy,
        "BrownianTreeNoiseSampler": object,  # never constructed (we pass our own noise_sampler)
    }
    exec(compile(blob, "<comfyui-sampling-fetched-at-regen>", "exec"), ns)
    return ns["sample_er_sde"], ns["sample_dpmpp_2m_sde"]


def run(sampler, x_init, sigmas, **kwargs):
    x = x_init.clone()
    recorder = []
    ms = _VEModelSampling()
    model = ToyModel(ms, recorder)
    # zero noise_sampler: never called (s_noise=0) but must be non-None so dpmpp_2m_sde does not
    # try to build a BrownianTreeNoiseSampler (which needs torchsde).
    noise_sampler = lambda s, s_next: torch.zeros_like(x)
    with torch.no_grad():
        final = sampler(
            model, x, sigmas, extra_args={}, noise_sampler=noise_sampler, s_noise=0.0, **kwargs
        )
    return recorder + [final.detach().reshape(-1).tolist()]


def main():
    sample_er_sde, sample_dpmpp_2m_sde = load_reference_samplers()

    dtype = torch.float64
    # 6 real steps + trailing 0 -> er_sde reaches Stage-3 (i>=2) on steps 2,3,4.
    sigmas = torch.tensor([12.0, 6.5, 3.4, 1.75, 0.85, 0.3, 0.0], dtype=dtype)
    x_init = torch.tensor([[0.7, -1.2, 1.5, -0.3, 0.9, -0.6]], dtype=dtype)  # [1, 6]

    er = run(sample_er_sde, x_init, sigmas, max_stage=3)
    dp = run(sample_dpmpp_2m_sde, x_init, sigmas, eta=1.0, solver_type="midpoint")

    doc = {
        "meta": {
            "purpose": "Independent golden trajectory for gen-core er_sde / dpmpp_2m_sde solvers.",
            "reference": "ComfyUI comfy/k_diffusion/sampling.py sample_er_sde / sample_dpmpp_2m_sde",
            "reference_url": UPSTREAM_URL,
            "reference_commit": UPSTREAM_COMMIT,
            "reference_sha256": UPSTREAM_SHA256,
            "reference_license": UPSTREAM_LICENSE,
            "reference_handling": "FETCHED at regen time and checksum-verified; NOT vendored into this repo (only this computed JSON is committed).",
            "regen_command": "python3 gen-core/tests/fixtures/gen_sde_solver_golden.py  (offline: COMFY_SAMPLING_PATH=/path/to/sampling.py python3 ...)",
            "generator": "gen-core/tests/fixtures/gen_sde_solver_golden.py",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
            "space": "VE (alpha=1, half_log_snr=-log(sigma), er_lambda=sigma)",
            "s_noise": 0.0,
            "toy_denoiser": "0.5*x + 0.25*sin(1.3*x) + 0.2*sigma*cos(0.7*x) - 0.1*sigma",
            "trajectory_layout": "trajectory[k] = latent state x_k at the start of step k; the final entry is the returned x after the terminal step.",
        },
        "sigmas": sigmas.reshape(-1).tolist(),
        "x_init": x_init.reshape(-1).tolist(),
        "er_sde": {"max_stage": 3, "s_noise": 0.0, "trajectory": er},
        "dpmpp_2m_sde": {"eta": 1.0, "s_noise": 0.0, "solver_type": "midpoint", "trajectory": dp},
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("er_sde   final:", [round(v, 6) for v in er[-1]])
    print("dpmpp    final:", [round(v, 6) for v in dp[-1]])
    print("er_sde   traj[0] == x_init:", np.allclose(er[0], x_init.reshape(-1).tolist()))


if __name__ == "__main__":
    main()
