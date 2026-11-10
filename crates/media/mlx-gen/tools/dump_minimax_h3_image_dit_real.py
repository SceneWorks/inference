"""MiniMax-H3-**Image** DiT real-weight first-step reference (SW-7090).

Runs the official diffusers ``MiniMaxH3ImageTransformer2DModel`` from the real
``transformer_diffusers/`` of ``MiniMaxAI/MiniMax-H3-Image`` (bf16, the published mixed precision:
``proj_in`` / ``time_embedder`` / ``pdd_proj_out`` stay float32) over one small seeded
``text2image`` packed sequence, and writes the inputs and outputs so the real-weight gates
(``mlx-gen-minimax-h3`` ``h3_image_real::h3_image_dit_first_step_matches_diffusers``, and the Candle
twin when it lands) can assert numeric parity on the published tensors.

The sequence is the modular pipeline's own ``text2image`` layout (``build_image_packed_sequence``
with no references, reproduced below line for line): ``NUM_TEXT`` text rows tagged ``1`` at
``(k, 0, 0)``, then a ``LATENT_HW`` latent (a 512x512 canvas, 256 rows) tagged ``0`` on one rotary
frame at ``t = NUM_TEXT`` with the aspect-normalized ``np.linspace(endpoint=False)`` grid. The
context is seeded noise, not a Qwen3-VL encode: this pins the DiT, and keeps the reference free of
the 66 GB text encoder.

Two forwards, both of the denoise loop's own call (``pdd_step_index=i`` and the step's timestep
``1 - sigma_i`` in float32, every row at that one timestep):

* ``step0`` — the first sampling step (``t = 0``, head 0);
* ``step3`` — step 3 (``t = 0.25``, head 3), so the gate also proves step ``i`` reaches head ``i``.

Both forwards also record the reference's own intermediate stages through forward hooks, so a gate
that misses its bound can say WHERE the error enters (``diag.step{i}.*``):

* ``temb`` — ``time_embedder(time_proj(t))``, ``[1, time_embed_dim]`` float32;
* ``norm_out.shift`` / ``norm_out.scale`` — the halves of ``norm_out.linear(silu(temb))``;
* ``block0.adaln`` — block 0's six AdaLN tables stacked in the reference's chunk order,
  ``[6, MODALITY_NUM, hidden_size]``;
* ``hidden.block{k}`` — the residual stream after block ``k`` for every ``k`` in the
  ``diag_layers`` metadata (first, middle, last), ``[1, seq_len, hidden_size]``;
* ``pre_head`` — ``norm_out``'s output, the input of head ``i``, ``[1, seq_len, hidden_size]``.

Step 3 above runs on the step-0 noise rows, which no sampler shows the model at ``t = 0.25``. The
producer therefore also walks the released sampler from those rows (steps 0, 1, 2 with the
``euler_update_form = "velocity"`` update ``x + (sigma_i - sigma_{i+1}) * v`` in float32) and
records the step-3 input it reaches and the step-3 velocity there (``diag.traj.rows3`` /
``diag.traj.out3``) — an on-trajectory comparison the gates print next to the asserted one.

Hash-pinned: the output carries ``producer_sha256``, the SHA-256 of THIS file's bytes, and the Rust
gate compares it to the committed copy. The output is a few tens of MB and is not committed.

Run (CPU torch; needs ~70 GB of host RAM for the bf16 DiT):

    MINIMAX_H3_IMAGE_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_image_dit_real.py \
        --out <dir>/dit-reference.safetensors

The reference class
-------------------

``MiniMaxH3ImageTransformer2DModel`` landed on diffusers ``main`` at ``REFERENCE_COMMIT`` and is in
no released wheel. When the installed diffusers lacks the module, this producer loads that exact
upstream file — refused unless its SHA-256 is ``REFERENCE_MODULE_SHA256`` — on top of the installed
package (0.38.0 in the reviewed Mage reference environment). One import it names is absent from
0.38.0: ``diffusers.hooks.tensor_parallel.PackedColwiseParallel``, which the module only stores in
its ``_tp_plan`` class attribute (tensor-parallel metadata, never called on a single-device
forward); a placeholder is registered for it and the substitution is recorded in the metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

SEED = 7090
MODULE = "diffusers.models.transformers.transformer_minimax_h3_image"
REFERENCE_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"
REFERENCE_MODULE_SHA256 = "0928960f80447c5e4db43d61a361ccc4c785021207325aabe65416f8f6f94e62"
REFERENCE_MODULE_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/{REFERENCE_COMMIT}/"
    "src/diffusers/models/transformers/transformer_minimax_h3_image.py"
)
# pdd_config.json / MiniMaxH3ImageSetTimestepsStep.sample_sigmas
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
NUM_TEXT = 24
LATENT_HW = (32, 32)
STEPS = (0, 3)
TEXT_TAG, IMAGE_TAG = 1, 0
ROPE_SPATIAL_SCALE = 32


def reference_class():
    """Return ``(MiniMaxH3ImageTransformer2DModel, source)`` — installed, or the pinned upstream file."""
    try:
        module = importlib.import_module(MODULE)
        return module.MiniMaxH3ImageTransformer2DModel, "installed"
    except ModuleNotFoundError:
        pass
    with urllib.request.urlopen(REFERENCE_MODULE_URL, timeout=60) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != REFERENCE_MODULE_SHA256:
        raise SystemExit(f"{REFERENCE_MODULE_URL} has sha256 {digest}, expected {REFERENCE_MODULE_SHA256}")
    path = Path(tempfile.mkdtemp()) / "transformer_minimax_h3_image.py"
    path.write_bytes(body)
    import diffusers.hooks
    import diffusers.models.transformers  # noqa: F401 — the parent package of the relative imports

    substituted = []
    try:
        importlib.import_module("diffusers.hooks.tensor_parallel")
    except ModuleNotFoundError:
        stub = types.ModuleType("diffusers.hooks.tensor_parallel")

        class PackedColwiseParallel:  # noqa: D401 — metadata placeholder, never invoked
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs

        stub.PackedColwiseParallel = PackedColwiseParallel
        sys.modules["diffusers.hooks.tensor_parallel"] = stub
        diffusers.hooks.tensor_parallel = stub
        substituted.append("hooks.tensor_parallel.PackedColwiseParallel")

    spec = importlib.util.spec_from_file_location(MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    source = f"diffusers@{REFERENCE_COMMIT}:{digest}"
    if substituted:
        source += " placeholder:" + ",".join(substituted)
    return module.MiniMaxH3ImageTransformer2DModel, source


def spatial_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def t2i_layout(num_text: int, latent_h: int, latent_w: int, patch: tuple[int, int, int]):
    """``build_image_packed_sequence`` with no reference grids (before_denoise.py)."""
    _, ph, pw = patch
    rows = (latent_h // ph) * (latent_w // pw)
    seq = num_text + rows
    position_ids = torch.zeros(seq, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    sqrt_area = np.sqrt(latent_h * latent_w)
    hg = torch.from_numpy(spatial_grid(latent_h, ph, sqrt_area))
    wg = torch.from_numpy(spatial_grid(latent_w, pw, sqrt_area))
    grid = torch.stack([g.reshape(-1) for g in torch.meshgrid(hg, wg, indexing="ij")], dim=-1)
    position_ids[num_text:, 0] = float(num_text)
    position_ids[num_text:, 1:] = grid
    token_tags = torch.full((seq,), IMAGE_TAG, dtype=torch.long)
    token_tags[:num_text] = TEXT_TAG
    return position_ids.to(torch.float32), token_tags, torch.arange(num_text, seq), torch.arange(num_text)


class StageTaps:
    """Forward hooks recording the reference's own intermediate tensors (see the module docs)."""

    def __init__(self, model, layers: tuple[int, ...]):
        self.layers = layers
        self.armed = False
        self.seen: dict[str, torch.Tensor] = {}
        self.handles = [
            model.time_embedder.register_forward_hook(self._put("temb")),
            model.norm_out.linear.register_forward_hook(self._put("norm_out.linear")),
            model.transformer_blocks[0].adaln_proj.register_forward_hook(self._put("block0.adaln")),
            model.norm_out.register_forward_hook(self._put("pre_head")),
        ]
        self.handles += [
            model.transformer_blocks[k].register_forward_hook(self._put(f"hidden.block{k}")) for k in layers
        ]

    def _put(self, key: str):
        def hook(_module, _inputs, output):
            if self.armed:
                self.seen[key] = output

        return hook

    def take(self, step: int) -> dict[str, np.ndarray]:
        seen, self.seen = self.seen, {}
        f32 = lambda t: t.detach().to(torch.float32).numpy().copy()  # noqa: E731
        shift, scale = seen["norm_out.linear"].chunk(2, dim=-1)
        out = {
            f"diag.step{step}.temb": f32(seen["temb"]),
            f"diag.step{step}.norm_out.shift": f32(shift),
            f"diag.step{step}.norm_out.scale": f32(scale),
            f"diag.step{step}.block0.adaln": f32(torch.stack(seen["block0.adaln"])),
            f"diag.step{step}.pre_head": f32(seen["pre_head"]),
        }
        for k in self.layers:
            out[f"diag.step{step}.hidden.block{k}"] = f32(seen[f"hidden.block{k}"])
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    args = ap.parse_args()

    import diffusers

    Model, module_source = reference_class()
    snapshot = Path(os.environ["MINIMAX_H3_IMAGE_SNAPSHOT"])
    producer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(os.cpu_count() or 1)

    model = Model.from_pretrained(
        snapshot / "transformer_diffusers", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    cfg = model.config
    assert cfg.num_pdd_heads == len(SIGMAS) - 1, cfg.num_pdd_heads
    assert model.proj_in.weight.dtype == torch.float32, "proj_in must keep its published float32"
    assert model.pdd_proj_out[0].weight.dtype == torch.float32, "the heads must keep float32"
    patch = tuple(cfg.patch_size)
    patch_dim = cfg.in_channels * patch[0] * patch[1] * patch[2]

    generator = torch.Generator().manual_seed(SEED)
    rows = (LATENT_HW[0] // patch[1]) * (LATENT_HW[1] // patch[2])
    image_rows = torch.randn(1, rows, patch_dim, generator=generator, dtype=torch.float32)
    context = torch.randn(1, NUM_TEXT, cfg.text_dim, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    position_ids, token_tags, image_indices, text_indices = t2i_layout(NUM_TEXT, *LATENT_HW, patch)
    seq = position_ids.shape[0]
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)

    # The released sampler's grid: sigma recovered from each float32 timestep, terminal 0.
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    diag_layers = (0, cfg.num_layers // 2 - 1, cfg.num_layers - 1)
    taps = StageTaps(model, diag_layers)

    def forward(rows: torch.Tensor, step: int) -> torch.Tensor:
        out = model(
            hidden_states=rows,
            encoder_hidden_states=context,
            timestep=timesteps[step : step + 1],
            timestep_indices=torch.zeros(seq, dtype=torch.long),
            token_tags=token_tags,
            position_ids=position_ids,
            image_indices=image_indices,
            text_indices=text_indices,
            pdd_step_index=step,
            return_dict=False,
        )[0]
        print(
            f"step {step} (t={timesteps[step].item():.6f}, head {step}): {tuple(out.shape)} "
            f"std {out.float().std().item():.6f} L2 {out.float().flatten().norm().item():.6f}"
        )
        return out.to(torch.float32)

    outputs = {}
    diag = {}
    with torch.no_grad():
        for step in STEPS:
            taps.armed = True
            outputs[step] = forward(image_rows, step)
            taps.armed = False
            diag.update(taps.take(step))
        # On-trajectory step 3: the sampler's own x_3, reached from the same noise rows.
        rows = image_rows
        velocity = outputs[0]
        for step in range(1, 4):
            rows = rows + (sigmas[step - 1] - sigmas[step]) * velocity
            velocity = forward(rows, step)
        diag["diag.traj.rows3"] = rows.numpy().copy()
        diag["diag.traj.out3"] = velocity.numpy().copy()

    tensors = {
        "in.rows": image_rows.numpy().copy(),
        "in.context": context.to(torch.float32).numpy().copy(),
    }
    for step, out in outputs.items():
        tensors[f"out.step{step}"] = out.numpy().copy()
    tensors.update(diag)
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.MiniMaxH3ImageTransformer2DModel",
        "reference_version": diffusers.__version__,
        "reference_module": module_source,
        "torch": torch.__version__,
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "num_text": str(NUM_TEXT),
        "latent_hw": f"{LATENT_HW[0]}x{LATENT_HW[1]}",
        "steps": ",".join(str(s) for s in STEPS),
        "timesteps": ",".join(f"{timesteps[s].item():.9g}" for s in STEPS),
        "diag_layers": ",".join(str(k) for k in diag_layers),
        "producer": "crates/media/mlx-gen/tools/dump_minimax_h3_image_dit_real.py",
        "producer_sha256": producer_sha256,
        "story": "SW-7090",
    }
    out_path = Path(args.out).expanduser()
    save_file(tensors, str(out_path), metadata=meta)
    print(f"wrote {out_path} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")
    print(f"producer_sha256 {producer_sha256}")


if __name__ == "__main__":
    main()
