"""Real-weight Iris-3B depth reference for the native ports' ignored real-weight tests (sc-25682).

Runs the FROZEN upstream `DepthPredictor` (see `_iris_common.py` and the pins in
`dump_iris_depth.py`) on CPU in FP32 — upstream's CPU path — against the pinned `depth/` export, on
a real photograph that ships in this repository
(`crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg`), at upstream's default `max_side` (1024).

Writes two files:

* `IRIS_DEPTH_REAL_GOLDEN` (outside the repository, ~20 MB, never committed): the PIL-decoded source
  pixels, the model input, the raw model output and the source-size depth map — the full-resolution
  reference the MLX real-weight test compares against on the machine that dumped it.
* `mlx-gen-iris/tests/fixtures/iris_depth_real_reference.safetensors` (committed, ~40 KB): the
  depth map average-pooled over 16x16 source blocks, its 2nd/98th percentiles, the sha256 of the
  decoded pixels and upstream's OWN bf16-autocast distance from that FP32 map (the same call under
  `torch.autocast(bfloat16)`, its CUDA release policy, measured here on CPU). It is what lets an off-host lane (the CUDA real-weight job) check the native depth
  of the same photo against upstream without the full golden.

Inputs:
  IRIS_SRC                upstream checkout at the pinned commit
  IRIS_DEPTH_DIR          the `depth/` folder of the `speridlabs/iris-3b` snapshot (pinned revision)
  IRIS_DEPTH_REAL_GOLDEN  output .safetensors path (outside the repository)

Run in the reference venv under an external RSS guard (CPU, ~15 GB): see `dump_iris_depth.py`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

# Importing the fixture dumper verifies the checkout + the depth module pins and puts the upstream
# package on sys.path (its `main()` only runs as a script).
import dump_iris_depth  # noqa: F401
from _iris_common import FIXTURE_DIR, IRIS_WEIGHTS_REVISION, save_safetensors
from _paths import REPO_ROOT, require_env

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from iris3b.downstream.depth import DepthPredictor  # noqa: E402

PHOTO = REPO_ROOT / "_vendor/mage_flow/assets/dog.jpg"
MAX_SIDE = 1024
POOL = 16


def main() -> None:
    depth_dir = require_env("IRIS_DEPTH_DIR", "the depth/ folder of the speridlabs/iris-3b snapshot")
    out = Path(require_env("IRIS_DEPTH_REAL_GOLDEN", "output .safetensors path outside the repo"))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    predictor = DepthPredictor(depth_dir, device="cpu")
    captured: dict[str, torch.Tensor] = {}
    predictor.model.register_forward_pre_hook(lambda m, args: captured.__setitem__("input", args[0].clone()))
    predictor.model.register_forward_hook(lambda m, args, o: captured.__setitem__("raw", o.clone()))
    image = Image.open(PHOTO)
    # what the predictor sees after its own `exif_transpose` (the caller's job natively)
    pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    t0 = time.time()
    depth = predictor(image, max_side=MAX_SIDE)
    print(f"depth: {time.time() - t0:.1f}s model input {tuple(captured['input'].shape)} -> {depth.shape} "
          f"range [{depth.min():.3f}, {depth.max():.3f}]", flush=True)
    meta = {
        "photo": str(PHOTO.relative_to(REPO_ROOT.parents[2])),
        "max_side": MAX_SIDE,
        "weights_revision": IRIS_WEIGHTS_REVISION,
        "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "pool": POOL,
    }
    save_safetensors(out, {
        "image": torch.from_numpy(pixels),
        "input": captured["input"],
        "raw": captured["raw"],
        "depth": torch.from_numpy(depth),
    }, meta)
    def pool(d: np.ndarray) -> torch.Tensor:
        return F.avg_pool2d(torch.from_numpy(d)[None, None], POOL, ceil_mode=False)[0, 0]

    pooled = pool(depth)
    low, high = np.percentile(depth, [2, 98])
    # Upstream's OWN release-policy distance: the same call under bf16 autocast (its CUDA path, here
    # on CPU) vs the FP32 map — the evidence behind the native bf16 bounds.
    # `__call__` pins autocast to CUDA (`enabled=device.type == "cuda"`), so replay its model call on
    # the captured input under CPU autocast, then its own `.float()` + bilinear resize back.
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        raw_bf16 = predictor.model(captured["input"], predictor.embeddings, predictor.mask).float()
    with torch.inference_mode():
        depth_bf16 = F.interpolate(raw_bf16, size=depth.shape, mode="bilinear", align_corners=False)
    depth_bf16 = depth_bf16[0, 0].numpy()
    diff = np.abs(depth_bf16 - depth)
    pooled_bf16 = pool(depth_bf16)
    pooled_diff = (pooled_bf16 - pooled).abs()
    r = np.corrcoef(pooled_bf16.flatten().numpy(), pooled.flatten().numpy())[0, 1]
    bf16 = {
        "upstream_bf16_max_abs": float(diff.max()),
        "upstream_bf16_mean_abs": float(diff.mean()),
        "upstream_bf16_pooled_max_abs": float(pooled_diff.max()),
        "upstream_bf16_pooled_mean_abs": float(pooled_diff.mean()),
        "upstream_bf16_pooled_pearson": float(r),
    }
    print(f"bf16 autocast: {time.time() - t0:.1f}s {bf16}", flush=True)
    save_safetensors(FIXTURE_DIR / "iris_depth_real_reference.safetensors", {
        "pooled": pooled,
        "percentiles": torch.tensor([low, high], dtype=torch.float32),
    }, {**meta, **bf16, "height": depth.shape[0], "width": depth.shape[1]})


if __name__ == "__main__":
    main()
