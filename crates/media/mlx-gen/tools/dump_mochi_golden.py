"""Mochi 1 (`genmo/mochi-1-preview`, Apache-2.0) parity-reference harness — sc-11984 (epic A1).

Emits the deterministic real-weight goldens that gate the native Mochi 1 port's Rust parity
suites (stories A2-A4). This is the **parity oracle**: it runs the licensed reference
implementation (Diffusers `MochiPipeline`) at a fixed seed / prompt / geometry and dumps
per-component + end-to-end `.safetensors` tensors that the Rust engines (`mlx-gen-mochi` /
`candle-gen-mochi`, provisioned by A2-A5) must reproduce.

Python is permitted in **test harnesses only** — never in the product path. This mirrors the
existing diffusers-reference dumps in this directory (see `dump_svd_pipeline_golden.py`,
`dump_wanvace_transformer_golden.py`) and the real-weights golden convention documented in
`tools/golden/README.md`.

## Goldens emitted (into the gitignored `tools/golden/`)

| stage       | file                              | reference component                          | consumed by (future) |
| ----------- | --------------------------------- | -------------------------------------------- | -------------------- |
| `te`        | `mochi_te_golden.safetensors`     | T5-XXL text encoder (`encode_prompt`)        | A2 `te_parity`       |
| `vae`       | `mochi_vae_golden.safetensors`    | `AutoencoderKLMochi` decode                  | A2 `vae_parity`      |
| `dit_block` | `mochi_dit_block_golden.safetensors` | one `MochiTransformerBlock` forward       | A3 `block_parity`    |
| `e2e`       | `mochi_e2e_golden.safetensors`    | full txt2v denoise + decode                  | A4 `e2e_parity`      |

The `dit_block` golden is captured **for free** during the `e2e` denoise: a forward hook on
`transformer.transformer_blocks[0]` records that block's real (post patch-embed / time-embed /
RoPE) inputs and output at the first sampler step. This avoids hand-replicating the transformer's
internal pre-block wiring, so the block fixture stays faithful to the reference forward.

## Determinism

Everything is pinned: `MOCHI_SEED` seeds a `torch.Generator` for the init noise (and the VAE-decode
latent), the prompt/negative/geometry/steps are fixed, and tensors are stored upcast to float32.
The reference precision is **bfloat16** (Mochi's shipped precision — the snapshot even carries a
`.bf16` transformer variant), matching how the large-model goldens (FLUX.2, Z-Image) are blessed.

## Prerequisites & run

Needs `torch` + `diffusers` (with `MochiPipeline`) + `safetensors`, and the pinned Mochi snapshot
(~tens of GB) in the standard HF cache or at `$MOCHI_SNAPSHOT`. Pin/verify the revision with
`scripts/release/ensure_model_snapshot.py --model mochi-1-preview` (see
`release/real-weight-models.toml`). Run from a torch+diffusers venv, e.g.:

    MOCHI_SNAPSHOT=/path/to/mochi-1-preview \
      python tools/dump_mochi_golden.py --stage all

`--stage {te,vae,dit_block,e2e,all}` selects which goldens to write (default `all`); the pipeline
is loaded once regardless. `dit_block` implies `e2e` (it is captured during that run).

Env overrides (small deterministic defaults keep the deep model cheap):
`MOCHI_SEED`, `MOCHI_PROMPT`, `MOCHI_NEGATIVE`, `MOCHI_H`, `MOCHI_W`, `MOCHI_FRAMES`,
`MOCHI_STEPS`, `MOCHI_GUIDANCE`, `MOCHI_MAXSEQ`, `MOCHI_DTYPE` (`bfloat16`|`float16`|`float32`),
`MOCHI_DEVICE` (`cuda`|`mps`|`cpu`; default auto).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import MochiPipeline
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

# ---------------------------------------------------------------------------- config

SEED = int(os.environ.get("MOCHI_SEED", "1984"))
PROMPT = os.environ.get(
    "MOCHI_PROMPT",
    "A calico kitten batting a ball of red yarn across a sunlit wooden floor.",
)
NEGATIVE = os.environ.get("MOCHI_NEGATIVE", "")
HEIGHT = int(os.environ.get("MOCHI_H", "64"))
WIDTH = int(os.environ.get("MOCHI_W", "64"))
# Mochi's VAE has a 6x temporal ratio, so num_frames must be 6k+1 (7 -> 2 latent frames).
FRAMES = int(os.environ.get("MOCHI_FRAMES", "7"))
STEPS = int(os.environ.get("MOCHI_STEPS", "2"))
GUIDANCE = float(os.environ.get("MOCHI_GUIDANCE", "4.5"))
MAXSEQ = int(os.environ.get("MOCHI_MAXSEQ", "256"))

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE = _DTYPES[os.environ.get("MOCHI_DTYPE", "bfloat16")]


def _auto_device() -> str:
    if override := os.environ.get("MOCHI_DEVICE"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _auto_device()


def _snapshot_dir() -> str:
    """Resolve the Mochi snapshot: $MOCHI_SNAPSHOT, else the standard HF cache, else the repo id."""
    if env := os.environ.get("MOCHI_SNAPSHOT"):
        return str(Path(env).expanduser())
    cached = hf_hub_cache() / "models--genmo--mochi-1-preview" / "snapshots"
    if cached.is_dir():
        snaps = sorted(cached.iterdir())
        if snaps:
            return str(snaps[-1])
    return "genmo/mochi-1-preview"  # let diffusers resolve from the hub/cache


def _f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).numpy()


def _meta() -> dict[str, np.ndarray]:
    return {
        "geometry": np.array([HEIGHT, WIDTH, FRAMES, STEPS, MAXSEQ], dtype=np.int32),
        "seed": np.array([SEED], dtype=np.int64),
        "guidance": np.array([GUIDANCE], dtype=np.float32),
    }


def _write(rel_name: str, tensors: dict[str, np.ndarray]) -> None:
    out = fixture(f"tools/golden/{rel_name}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out)
    print(f"wrote {out}")
    for k, v in tensors.items():
        print(f"    {k:28s} {tuple(v.shape)} {v.dtype}")


# ------------------------------------------------------------------------- stage: te


def dump_te(pipe: MochiPipeline) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """T5-XXL text-encoder golden. Returns the embeds for reuse by the e2e denoise."""
    with torch.no_grad():
        prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
            prompt=PROMPT,
            negative_prompt=NEGATIVE or None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=MAXSEQ,
            device=DEVICE,
        )
    tensors = {
        "prompt_embeds": _f32(prompt_embeds),
        "prompt_attention_mask": _f32(prompt_mask),
        "negative_prompt_embeds": _f32(neg_embeds),
        "negative_prompt_attention_mask": _f32(neg_mask),
        **_meta(),
    }
    _write("mochi_te_golden.safetensors", tensors)
    return prompt_embeds, prompt_mask, neg_embeds, neg_mask


# ------------------------------------------------------------------------ stage: vae


def _denormalize_latents(pipe: MochiPipeline, latents: torch.Tensor) -> torch.Tensor:
    """Reproduce diffusers MochiPipeline latent de-normalization (per-channel mean/std, else scale)."""
    cfg = pipe.vae.config
    mean = getattr(cfg, "latents_mean", None)
    std = getattr(cfg, "latents_std", None)
    scaling = getattr(cfg, "scaling_factor", 1.0)
    if mean is not None and std is not None:
        c = latents.shape[1]
        mean_t = torch.tensor(mean, device=latents.device, dtype=latents.dtype).view(1, c, 1, 1, 1)
        std_t = torch.tensor(std, device=latents.device, dtype=latents.dtype).view(1, c, 1, 1, 1)
        return latents * std_t / scaling + mean_t
    return latents / scaling


def dump_vae(pipe: MochiPipeline) -> None:
    """VAE-decode golden on a seeded latent (isolates `AutoencoderKLMochi.decode`)."""
    channels = pipe.vae.config.latent_channels
    lat_t = (FRAMES - 1) // pipe.vae_temporal_scale_factor + 1
    lat_h = HEIGHT // pipe.vae_spatial_scale_factor
    lat_w = WIDTH // pipe.vae_spatial_scale_factor
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
    # Keep the latent (and hence the de-normalization + decode) in f32: the Mochi VAE is f32-only
    # (see `main`), so the golden must be a faithful f32 decode, not a bf16 one.
    latents = torch.randn(
        (1, channels, lat_t, lat_h, lat_w), generator=gen, dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        denorm = _denormalize_latents(pipe, latents)
        video = pipe.vae.decode(denorm, return_dict=False)[0]  # [1, 3, F, H, W], ~[-1, 1]
    tensors = {
        "latents": _f32(latents),
        "denormalized_latents": _f32(denorm),
        "video": _f32(video),
        **_meta(),
    }
    _write("mochi_vae_golden.safetensors", tensors)


# --------------------------------------------------------------- stage: e2e + block


class _BlockCapture:
    """Records the first forward of `transformer_blocks[0]` (inputs + output) via a hook."""

    def __init__(self) -> None:
        self.captured: dict[str, np.ndarray] | None = None
        self.handle = None

    def _flatten(self, prefix: str, value: object, out: dict[str, np.ndarray]) -> None:
        if isinstance(value, torch.Tensor):
            out[prefix] = _f32(value)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self._flatten(f"{prefix}.{i}", item, out)

    def hook(self, _module, args, kwargs, output):  # torch forward hook (with_kwargs=True)
        if self.captured is not None:
            return
        rec: dict[str, np.ndarray] = {}
        for i, item in enumerate(args):
            self._flatten(f"block_in.arg{i}", item, rec)
        for name, item in kwargs.items():
            self._flatten(f"block_in.{name}", item, rec)
        self._flatten("block_out", output, rec)
        self.captured = rec
        if self.handle is not None:
            self.handle.remove()

    def register(self, block: torch.nn.Module) -> None:
        self.handle = block.register_forward_hook(self.hook, with_kwargs=True)


def dump_e2e_and_block(
    pipe: MochiPipeline,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_mask: torch.Tensor,
    want_block: bool,
) -> None:
    """Full txt2v e2e golden (final latent + decoded frame). Captures the DiT-block golden en route."""
    capture = _BlockCapture()
    if want_block:
        capture.register(pipe.transformer.transformer_blocks[0])

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        result = pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_attention_mask=neg_mask,
            height=HEIGHT,
            width=WIDTH,
            num_frames=FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
            output_type="latent",
        )
    final_latents = result.frames  # raw denoised latent when output_type == "latent"
    with torch.no_grad():
        # The DiT denoise runs bf16, so `final_latents` is bf16; upcast for the f32 VAE decode
        # (the Mochi VAE is f32-only — see `main`).
        denorm = _denormalize_latents(pipe, final_latents.float())
        video = pipe.vae.decode(denorm, return_dict=False)[0]

    tensors = {
        "final_latents": _f32(final_latents),
        "video": _f32(video),
        **_meta(),
    }
    _write("mochi_e2e_golden.safetensors", tensors)

    if want_block:
        if capture.captured is None:
            raise RuntimeError("block[0] forward hook never fired — cannot emit the DiT-block golden")
        _write("mochi_dit_block_golden.safetensors", {**capture.captured, **_meta()})


# ------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["te", "vae", "dit_block", "e2e", "all"],
        default="all",
    )
    args = parser.parse_args()
    stages = {"te", "vae", "dit_block", "e2e"} if args.stage == "all" else {args.stage}

    snap = _snapshot_dir()
    print(f"loading MochiPipeline from {snap} (device={DEVICE}, dtype={DTYPE})")
    pipe = MochiPipeline.from_pretrained(snap, torch_dtype=DTYPE, variant="bf16")
    pipe.to(DEVICE)

    # The Mochi AsymmVAE is numerically UNSTABLE in bf16: its decoder applies no output normalization
    # and its intermediate activations reach O(100), outside bf16's precise range. A bf16 decode yields
    # a video far outside [-1, 1] (observed range ~[-8, 6]); the f32 decode is the correct [-1, 1]
    # output. So decode the VAE in f32 regardless of the pipeline dtype — the DiT still runs bf16
    # (its shipped precision). Without this the `vae`/`e2e` golden videos are garbage (sc-11985).
    pipe.vae.to(torch.float32)

    # TE embeds are needed to drive e2e/dit_block; compute them if any of those run.
    embeds = None
    if stages & {"te", "e2e", "dit_block"}:
        embeds = dump_te(pipe) if "te" in stages else _encode_only(pipe)

    if "vae" in stages:
        dump_vae(pipe)

    if stages & {"e2e", "dit_block"}:
        assert embeds is not None
        dump_e2e_and_block(pipe, *embeds, want_block="dit_block" in stages)

    print("done.")
    return 0


def _encode_only(pipe: MochiPipeline):
    with torch.no_grad():
        return pipe.encode_prompt(
            prompt=PROMPT,
            negative_prompt=NEGATIVE or None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=MAXSEQ,
            device=DEVICE,
        )


if __name__ == "__main__":
    raise SystemExit(main())
