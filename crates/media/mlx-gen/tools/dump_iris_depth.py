"""Dump the Iris-3B monocular-depth fixtures for the native MLX and Candle ports (sc-25682).

Runs the FROZEN upstream depth task (`iris3b/downstream/depth.py`, see `_iris_common.py` and the
extra pins below) on CPU in fp32 — upstream's own CPU path (`torch.autocast` is enabled only on
CUDA) — over a miniature seeded export that keeps every architectural switch of the released
`depth/config.yaml`; only the widths shrink (the generation fixtures' `TINY_DIT`).

The export is written in exactly the `scripts/export_downstream.py` layout and loaded back through
upstream's REAL `DepthPredictor` (`load_export` → strict `load_state_dict`), so the key names
(`pixel.*`, `depth_reducer.*`), the `task` section and the empty-prompt file are upstream's own. Every
golden case is upstream's `DepthPredictor.__call__` end to end: the max-side control, the
round-half-even patch-grid size law, PIL's Lanczos resize, the `[-1, 1]` mapping, the single forward
(RGB ‖ zero channel at t = num_train_timesteps, empty-prompt conditioning), the 1x1 reducer and the
bilinear resize back to the source size. The model's input and raw output are captured by hooks so
the native port is checked stage by stage. `colorize` (near = bright, 2nd–98th percentile, inferno)
is dumped for the presentation adapters.

Writes `mlx-gen-iris/tests/fixtures/`:

* `tiny-depth/{config.yaml, model.safetensors, empty_prompt.safetensors}` — the miniature export.
* `iris_depth_golden.safetensors` — per case `<case>/image` (u8 HWC), `<case>/input` (the model's
  `[1, 3, h, w]` input), `<case>/raw` (`[1, 1, h, w]` model output), `<case>/depth` (`[H, W]` at the
  source size), `<case>/colorize` (u8 HWC); `max_side` per case in the metadata.

Run: `IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_depth.py` in a venv with torch 2.8 CPU,
numpy, pillow, safetensors, omegaconf and matplotlib (see `_iris_common.py`).
"""

from __future__ import annotations

import hashlib
import json
import shutil

from _iris_common import FIXTURE_DIR, IRIS_COMMIT, import_upstream, save_safetensors

ROOT = import_upstream()

# The depth task's own upstream modules, pinned on top of `_iris_common.UPSTREAM_SHA256`.
DEPTH_SHA256 = {
    "src/iris3b/downstream/__init__.py": "94034cdb0da5189b07f01b5f57f2a0cd04cda14cfb7cf9df1bc77e375cd02cec",
    "src/iris3b/downstream/depth.py": "5ca54ceacf0ec5b583d3b25cf6e07967b912bb31179518c70d0e47af281a118f",
    "scripts/export_downstream.py": "be14ab9cfc443b8bd348a6ccde19843ae6806125620d1e2a090ee90dfbcc3afa",
}
for rel, want in DEPTH_SHA256.items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from iris3b.config import FlowConfig, ModelConfig, PixelStageConfig, TextEncoderConfig  # noqa: E402
from iris3b.downstream.depth import DepthPredictor, IrisDepth, colorize  # noqa: E402

torch.manual_seed(0)

EXPORT_DIR = FIXTURE_DIR / "tiny-depth"
TEXT_DIM = 32
TEXT_LEN = 10
HIDDEN_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
REAL_TOKENS = 5  # the release's empty prompt keeps the 5 assistant-suffix tokens

# The generation fixtures' miniature backbone (`dump_iris_golden.TINY_DIT`), patch 4.
TINY_DIT = ModelConfig(
    block="single_stream", dual_depth=2, final_block_text="keep", hidden_size=64, depth=3,
    num_heads=4, num_kv_heads=2, gated_attention=True, sandwich_norm=True, patch_size=4,
    in_channels=3, mlp_ratio=4.0, qkv_bias=False, qk_norm=True, norm_eps=1e-6,
    modulation="shared_bias", timestep_max_period=10.0, adaln_zero_init=True, rope_theta=10000.0,
    rope_scale=16.0, rope_aspect="isotropic", rope_frame_pairs=0, text_rope=True,
    text_rope_theta=10000.0, text_abs_pos_embed=True, text_dim=TEXT_DIM, text_len=TEXT_LEN,
    text_adapter="lap_blocks2", text_lap_num_layers=len(HIDDEN_LAYERS), text_lap_num_heads=4,
    text_lap_mlp_ratio=1.3, repa_layer=0,
    pixel=PixelStageConfig(enabled=True, depth=2, hidden_size=8, attn_hidden_size=32, num_heads=2,
                           mlp_ratio=4.0, modulation="post", abs_pos_embed=True),
)
TINY_TEXT = TextEncoderConfig(
    name="qwen3_vl", pretrained="tiny-snapshot/text_encoder", dim=TEXT_DIM, max_length=TEXT_LEN,
    dtype="bfloat16", attn_implementation="sdpa", hidden_layers=list(HIDDEN_LAYERS), compile=False,
    on_caption_overflow="warn",
)
TINY_FLOW = FlowConfig(shift=4.0)

# (name, width, height, max_side): odd / small / non-square / capped / native / round-half-even.
CASES = [
    ("native_odd", 37, 23, 0),       # 9.25 / 5.75 patches -> 36 x 24, resized both ways
    ("capped", 50, 30, 32),          # scale 0.64 -> 32 x 20
    ("tiny", 3, 5, 0),               # below one patch -> 4 x 4 (upsampled)
    ("exact", 24, 16, 0),            # already on the grid: no resize either way
    ("half_even", 10, 14, 0),        # 2.5 / 3.5 patches -> 8 x 16 (Python round-half-even)
    ("capped_noop", 20, 12, 1024),   # the cap never upscales: no resize
]


def tiny_depth() -> IrisDepth:
    model = IrisDepth(TINY_DIT).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            owner = name.rsplit(".", 1)[0].split(".")[-1]
            if "norm" in owner and name.endswith(".weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith("y_pos_embedding"):
                p.copy_(torch.randn_like(p))
            elif name.startswith("depth_reducer."):
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.copy_(0.15 * torch.randn_like(p))
    return model


def write_export(model: IrisDepth) -> None:
    """`scripts/export_downstream.py depth`: the `pixel.` / `depth_reducer.` state dict (FP32), the
    empty-prompt pair and `config.yaml` = the parent's model/text_encoder/flow + `task`."""
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)
    weights = {k: v for k, v in model.state_dict().items() if k.startswith(("pixel.", "depth_reducer."))}
    save_file({k: v.float().contiguous() for k, v in weights.items()}, str(EXPORT_DIR / "model.safetensors"),
              metadata={"format": "pt"})
    g = torch.Generator().manual_seed(4)
    mask = torch.zeros(1, TEXT_LEN, dtype=torch.bool)
    mask[0, :REAL_TOKENS] = True
    embeddings = torch.randn(1, TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g)
    embeddings = embeddings * mask[:, :, None, None]
    save_file({"embeddings": embeddings.float().contiguous(), "mask": mask.contiguous()},
              str(EXPORT_DIR / "empty_prompt.safetensors"))
    sections = {
        "model": OmegaConf.to_container(OmegaConf.structured(TINY_DIT)),
        "text_encoder": OmegaConf.to_container(OmegaConf.structured(TINY_TEXT)),
        "flow": OmegaConf.to_container(OmegaConf.structured(TINY_FLOW)),
    }
    del sections["text_encoder"]["null_embed_dir"]
    OmegaConf.save(OmegaConf.create({**sections, "task": {"name": "depth"}}), str(EXPORT_DIR / "config.yaml"))


def image(width: int, height: int, seed: int) -> Image.Image:
    """A smooth gradient plus seeded noise, so the Lanczos lobes and the clip8 saturation both bite."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    base = torch.stack([xx * 255.0 / max(width - 1, 1), yy * 255.0 / max(height - 1, 1),
                        (xx + yy) * 127.0 / max(width + height - 2, 1)], dim=-1)
    pixels = (base + 60.0 * torch.randn(height, width, 3, generator=g)).clamp(0, 255).round()
    return Image.fromarray(pixels.to(torch.uint8).numpy(), "RGB")


def main() -> None:
    write_export(tiny_depth())
    predictor = DepthPredictor(str(EXPORT_DIR), device="cpu")  # upstream's real loader
    captured: dict[str, torch.Tensor] = {}
    predictor.model.register_forward_pre_hook(lambda m, args: captured.__setitem__("input", args[0].clone()))
    predictor.model.register_forward_hook(lambda m, args, out: captured.__setitem__("raw", out.clone()))
    tensors, meta = {}, {"upstream_commit": IRIS_COMMIT, "torch": torch.__version__}
    for i, (name, width, height, max_side) in enumerate(CASES):
        img = image(width, height, 10 + i)
        depth = predictor(img, max_side=max_side)
        tensors[f"{name}/image"] = torch.from_numpy(np.array(img))
        tensors[f"{name}/input"] = captured["input"]
        tensors[f"{name}/raw"] = captured["raw"]
        tensors[f"{name}/depth"] = torch.from_numpy(depth)
        tensors[f"{name}/colorize"] = torch.from_numpy(np.array(colorize(depth)))
        meta[f"{name}/max_side"] = max_side
        print(f"{name}: {width}x{height} max_side={max_side} -> model {tuple(captured['input'].shape[-2:])} "
              f"depth {depth.shape} range [{depth.min():.3f}, {depth.max():.3f}]")
    meta["cases"] = json.dumps([c[0] for c in CASES])
    save_safetensors(FIXTURE_DIR / "iris_depth_golden.safetensors", tensors, meta)


if __name__ == "__main__":
    main()
