"""LTX-2.5 latent-upsampler reference goldens — sc-18773.

Dumps reference `upsample_video` I/O for BOTH shipped `LatentUpsampler` checkpoints, taken from the
**upstream** module (`Lightricks/LTX-2` @ `d151147788a9284cca791edc6ce898007e727fe6`, v1.2.0,
`ltx_core/model/upsampler/model.py`) built by upstream's own `LatentUpsamplerConfigurator` off each
checkpoint's `__metadata__["config"]`:

  * `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
    (`mid_channels` 1024, `spatial_upsample`) — `H,W → 2H,2W`, frame count untouched;
  * `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`
    (`mid_channels` 512, `temporal_upsample`) — `F → 2F−1` after the leading-frame drop, `H,W`
    untouched.

Everything runs **float32**: the weights ship bf16 and both the MLX and the candle port upcast
losslessly, so the gate is a correctness check rather than a bf16 rounding check. (The LTX-2.3
`ltx_upsampler_golden` is bf16 on purpose — that one gates the bf16 production path and is
untouched here.)

`latent_mean` / `latent_std` are the REAL conv-VAE `per_channel_statistics.{mean,std}-of-means`, so
the golden exercises the un-normalize → upsample → re-normalize wrapper upstream actually runs
(`upsample_video`), not a synthetic stand-in.

`temporal_frame_counts` records `[frames_in, frames_out]` measured by RUNNING the reference at
`F ∈ {1, 9, 17}` — the `n % 8 == 1` edge sizes — so the frame rule is evidence from upstream rather
than arithmetic asserted against itself.

Run:
    LTX2_SRC=~/src/LTX-2/packages/ltx-core/src \\
    LTX25_UPSAMPLER_DIR=/path/to/Lightricks--LTX-2.5/latent_upscale_models \\
    LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \\
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx25_upsampler_goldens.py
Output (committed):
    mlx-gen-ltx/tests/fixtures/ltx25_spatial_upsampler_golden.safetensors
    mlx-gen-ltx/tests/fixtures/ltx25_temporal_upsampler_golden.safetensors
"""

from __future__ import annotations

from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

from _ltx25_diffvae_ref import (
    CONV_VAE,
    REFERENCE_COMMIT,
    ltx_core_on_path,
    probe,
    report,
    require_finite,
    vae_dir,
)
from _paths import fixture, require_env

SPATIAL = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
TEMPORAL = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"

# Small geometry: enough frames for the Conv3d temporal kernel and the frame-axis pixel shuffle,
# small enough that the committed fixtures stay well under a megabyte.
SPATIAL_F, SPATIAL_H, SPATIAL_W = 3, 6, 6
TEMPORAL_F, TEMPORAL_H, TEMPORAL_W = 9, 6, 6
# The `n % 8 == 1` edge sizes the frame rule must hold at, including the single-frame floor.
FRAME_COUNT_PROBES = (1, 9, 17)


def upsampler_dir() -> Path:
    return Path(
        require_env(
            "LTX25_UPSAMPLER_DIR",
            f"directory holding {SPATIAL} / {TEMPORAL} from the gated Lightricks/LTX-2.5 repo",
        )
    ).expanduser()


def load_reference_upsampler(path: Path):
    """Upstream `LatentUpsampler` with the real weights, configured from the file's own metadata."""
    ltx_core_on_path()

    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.model.upsampler import LatentUpsamplerConfigurator

    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(str(path))
    config = metadata["config"]
    model = LatentUpsamplerConfigurator.from_metadata(metadata)
    state = loader.load(str(path))
    missing, unexpected = model.load_state_dict(state.sd, strict=False)
    assert not missing, f"upsampler weights missing: {missing[:8]}"
    assert not unexpected, f"checkpoint keys with no home: {unexpected[:8]}"
    return model.to(dtype=torch.float32).eval(), config


def latent_statistics() -> tuple[torch.Tensor, torch.Tensor]:
    """The conv VAE's `per_channel_statistics`, the exact buffers `upsample_video` normalizes with."""
    path = vae_dir() / CONV_VAE
    with safetensors.safe_open(str(path), framework="pt") as f:
        mean = f.get_tensor("per_channel_statistics.mean-of-means")
        std = f.get_tensor("per_channel_statistics.std-of-means")
    return mean.to(torch.float32), std.to(torch.float32)


def upsample_latents(
    model, latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """`upsample_video`'s arithmetic with the statistics passed in directly.

    Identical composition to `ltx_core.model.upsampler.upsample_video`; the only change is that the
    two `per_channel_statistics` buffers are supplied rather than reached through a `VideoEncoder`,
    so the dump needs no 1.45 GB encoder instance to state the same thing.
    """
    view = (1, -1, 1, 1, 1)
    with torch.no_grad():
        return (model(latent * std.view(view) + mean.view(view)) - mean.view(view)) / std.view(view)


mean, std = latent_statistics()
print(f"[stats] mean{tuple(mean.shape)} std{tuple(std.shape)}")

root = upsampler_dir()
for name, checkpoint, geometry, out_name in (
    (
        "spatial",
        SPATIAL,
        (SPATIAL_F, SPATIAL_H, SPATIAL_W),
        "ltx25_spatial_upsampler_golden.safetensors",
    ),
    (
        "temporal",
        TEMPORAL,
        (TEMPORAL_F, TEMPORAL_H, TEMPORAL_W),
        "ltx25_temporal_upsampler_golden.safetensors",
    ),
):
    path = root / checkpoint
    print(f"[ref] {path}")
    model, config = load_reference_upsampler(path)
    print(f"[ref] config={config}")

    frames, height, width = geometry
    latent = probe((1, int(config["in_channels"]), frames, height, width), seed=11)
    output = upsample_latents(model, latent, mean, std)
    print(f"[{name}] latent{tuple(latent.shape)} -> {tuple(output.shape)}")

    tensors = {
        "latent": latent.to(torch.float16).contiguous(),
        "latent_mean": mean.contiguous(),
        "latent_std": std.contiguous(),
        "output": output.to(torch.float32).contiguous(),
    }
    meta = {
        "story": "sc-18773",
        "reference": f"Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0)",
        "checkpoint": checkpoint,
        "config": repr(config),
        "dtype": "float32 (weights upcast from bf16)",
        "latent": "x".join(map(str, latent.shape)),
        "output": "x".join(map(str, output.shape)),
        "statistics": f"{CONV_VAE} per_channel_statistics.{{mean,std}}-of-means",
        "storage": "float16: latent; float32: everything else",
    }

    if name == "temporal":
        # Measured, not derived: run the reference at each edge size and record what it returns.
        counts = []
        for f_in in FRAME_COUNT_PROBES:
            probe_latent = probe(
                (1, int(config["in_channels"]), f_in, height, width), seed=12 + f_in
            )
            with torch.no_grad():
                f_out = model(probe_latent).shape[2]
            print(f"[temporal] frames {f_in} -> {f_out}")
            counts.append([f_in, f_out])
        tensors["temporal_frame_counts"] = torch.tensor(counts, dtype=torch.int32).contiguous()
        meta["temporal_frame_counts"] = "rows of [frames_in, frames_out] measured on the reference"

    for key, value in tensors.items():
        if value.is_floating_point():
            require_finite(key, value)
        report(key, value.to(torch.float32))

    out = fixture(f"mlx-gen-ltx/tests/fixtures/{out_name}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out}")
