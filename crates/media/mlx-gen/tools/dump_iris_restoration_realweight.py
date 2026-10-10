"""Real-weight Iris-3B restoration references for the native restorers' ignored real-weight tests
(sc-25683).

Runs the FROZEN upstream `Restorer` (see `_iris_common.py` + the restoration pins in
`dump_iris_restoration.py`) on CPU against the pinned `upscaler/` export, 4x with colour fix, in
upstream's FP32 CPU path AND under the release's bf16 autocast forced onto the CPU (upstream enables
autocast only on CUDA; the bf16 run measures upstream's OWN bf16-vs-fp32 distance — the bound the
native bf16 paths are held to). Two cases (`IRIS_RESTORE_CASE`):

* `large` — a 512x384 input → 2048x1536 output (6 overlapping 1024-px tiles, the Gaussian fusion on
  real weights). Tens of MB of float stages: written to `IRIS_RESTORE_GOLDEN` OUTSIDE the repo, never
  committed; read by the MLX real-weight test.
* `small` — a 64x48 input → 256x192 output (enlarged to 1365x1024 = 2 tiles, resized back: the
  small-image path on real weights). RGB8 only (~300 KB): committed as
  `mlx-gen-iris/tests/fixtures/iris_restoration_real_small.safetensors`, so the CUDA real-weight job
  (which has no Python oracle) checks parity against it.

Inputs:
  IRIS_SRC              upstream checkout at the pinned commit
  IRIS_UPSCALER_DIR     the `upscaler/` folder of the `speridlabs/iris-3b` snapshot (pinned revision)
  IRIS_RESTORE_SOURCE   any RGB image; its centre 4:3 crop is resized (PIL Lanczos) to the input size
  IRIS_RESTORE_CASE     `large` (default) or `small`
  IRIS_RESTORE_GOLDEN   output .safetensors path outside the repository (`large` only)

Run: `python -I tools/dump_iris_restoration_realweight.py` in the isolated reference venv (CPU,
~20 GB RAM), under an external RSS guard.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _iris_common import FIXTURE_DIR, IRIS_WEIGHTS_REVISION  # noqa: E402
from _paths import require_env  # noqa: E402
from dump_iris_restoration import restoration, save_safetensors  # noqa: E402  (verifies the pins)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from iris3b.downstream.restoration import Restorer  # noqa: E402

SCALE = 4.0
CASES = {"large": (512, 384), "small": (64, 48)}


def source_image(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    cw, ch = (w, w * 3 // 4) if w * 3 // 4 <= h else (h * 4 // 3, h)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize(size, Image.Resampling.LANCZOS)


def run(restorer: Restorer, image: Image.Image, label: str, floats: bool) -> dict:
    captured: dict = {}
    tiled, fix = restoration.tiled, restoration.wavelet_color_fix

    def tiled_spy(*a, **k):
        captured["fused"] = tiled(*a, **k).clone()
        return captured["fused"]

    def fix_spy(*a, **k):
        captured["restored"] = fix(*a, **k).clone()
        return captured["restored"]

    restoration.tiled, restoration.wavelet_color_fix = tiled_spy, fix_spy
    t0 = time.time()
    out = restorer(image, scale=SCALE, color_fix=True)
    print(f"{label}: {image.size} -> {out.size} in {time.time() - t0:.1f}s", flush=True)
    restoration.tiled, restoration.wavelet_color_fix = tiled, fix
    tensors = {f"{label}/output": torch.from_numpy(np.asarray(out).copy())}
    if floats:
        tensors[f"{label}/fused"] = captured["fused"][0]
        tensors[f"{label}/restored"] = captured["restored"][0]
    return tensors


def bf16_tile(restorer: Restorer):
    """The release's autocast policy, forced onto the CPU (upstream enables it only on CUDA)."""

    def restore_tile(x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0],), restorer.time, device=x.device)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = restorer.embeddings.expand(len(x), *restorer.embeddings.shape[1:])
            v = restorer.model(x, t, y, y_mask=restorer.mask.expand(len(x), -1)).x
        return x.float() - restorer.sigma * v.float()

    return restore_tile


def main() -> None:
    export = require_env("IRIS_UPSCALER_DIR", "the upscaler/ folder of the pinned snapshot")
    source = require_env("IRIS_RESTORE_SOURCE", "an RGB source image")
    case = os.environ.get("IRIS_RESTORE_CASE", "large")
    large = case == "large"
    out = (
        Path(require_env("IRIS_RESTORE_GOLDEN", "output .safetensors path outside the repo"))
        if large
        else FIXTURE_DIR / "iris_restoration_real_small.safetensors"
    )
    image = source_image(source, CASES[case])
    restorer = Restorer(export, device="cpu")
    tensors = {"input": torch.from_numpy(np.asarray(image).copy())}
    tensors.update(run(restorer, image, "fp32", floats=large))
    restorer.restore_tile = bf16_tile(restorer)
    tensors.update(run(restorer, image, "bf16", floats=large))
    a = tensors["fp32/output"].int()
    b = tensors["bf16/output"].int()
    diff = (a - b).abs().float()
    print(f"upstream bf16 vs fp32 RGB8: max {diff.max():.0f} mean {diff.mean():.3f}", flush=True)
    save_safetensors(
        out,
        tensors,
        {
            "case": case,
            "scale": SCALE,
            "weights_revision": IRIS_WEIGHTS_REVISION,
            "source": Path(source).name,
        },
    )


if __name__ == "__main__":
    main()
