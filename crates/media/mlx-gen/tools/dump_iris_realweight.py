"""Real-weight Iris-3B reference for the native port's ignored real-weight test (sc-25679).

Runs the FROZEN upstream (see `_iris_common.py`) on CPU against the pinned weights: the release
`Qwen3VLTextEncoder` (bf16, its configured dtype) on one prompt and the empty CFG null, one FP32
backbone + pixel-head forward on fixed inputs, and a short upstream `generate()` at 256x256 from
fixed injected noise. The output is a machine-local golden (tens of MB) written OUTSIDE the repo —
never committed; the native test reads it through `IRIS_REAL_GOLDEN`.

Inputs:
  IRIS_SRC            upstream checkout at the pinned commit
  IRIS_WEIGHTS_DIR    `speridlabs/iris-3b` snapshot root at the pinned revision
  IRIS_TEXT_ENCODER_DIR  `Qwen/Qwen3-VL-4B-Instruct` snapshot at the pinned revision
  IRIS_REAL_GOLDEN    output .safetensors path (outside the repository)

Run: `python -I tools/dump_iris_realweight.py` in the isolated reference venv (CPU, ~25 GB RAM).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from _iris_common import QWEN3_VL_REVISION, IRIS_WEIGHTS_REVISION, import_upstream, save_safetensors
from _paths import require_env

import_upstream()

import torch  # noqa: E402

from iris3b.config import inference_config  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.sampling import generate, load_for_inference  # noqa: E402
from iris3b.text.qwen3_vl import Qwen3VLTextEncoder  # noqa: E402

PROMPT = "a red fox sleeping in fresh snow, golden hour"
SIZE = 256
STEPS = 20
CFG = 3.0
SEED = 1234


def main() -> None:
    weights_dir = Path(require_env("IRIS_WEIGHTS_DIR", "speridlabs/iris-3b snapshot root"))
    te_dir = require_env("IRIS_TEXT_ENCODER_DIR", "Qwen/Qwen3-VL-4B-Instruct snapshot")
    out = Path(require_env("IRIS_REAL_GOLDEN", "output .safetensors path outside the repo"))
    torch.set_num_threads(max(1, torch.get_num_threads()))

    raw, weights = load_for_inference(weights_dir)
    cfg = inference_config(raw)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.eval().to(dtype=torch.float32)

    with tempfile.TemporaryDirectory() as null_dir:
        cfg.text_encoder.pretrained = te_dir
        cfg.text_encoder.null_embed_dir = null_dir
        enc = Qwen3VLTextEncoder(cfg.text_encoder, device="cpu")
        t0 = time.time()
        cond = enc.encode([PROMPT])
        null = enc.null("")
        print(f"encode: {time.time() - t0:.1f}s", flush=True)

        g = torch.Generator().manual_seed(SEED)
        x = torch.randn(1, 3, SIZE, SIZE, generator=g)
        t = torch.tensor([500.0])
        t0 = time.time()
        with torch.no_grad():
            velocity = model(x, t, cond.embeddings.float(), y_mask=cond.mask).x
        print(f"forward: {time.time() - t0:.1f}s", flush=True)

        noise = torch.randn(1, 3, SIZE, SIZE, generator=g)
        t0 = time.time()
        with torch.no_grad():
            image = generate(
                model, enc, [PROMPT], height=SIZE, width=SIZE, steps=STEPS, order=2, cfg_scale=CFG,
                cfg_interval=(0.0, 1.0), shift=cfg.flow.shift, negative_prompt="", device="cpu",
                noise=noise,
            )
        print(f"generate: {time.time() - t0:.1f}s", flush=True)

    save_safetensors(
        out,
        {
            "cond/embeddings": cond.embeddings[0].float(),
            "cond/mask": cond.mask[0].to(torch.int32),
            "null/embeddings": null.embeddings[0].float(),
            "null/mask": null.mask[0].to(torch.int32),
            "forward/x": x,
            "forward/t": t,
            "forward/velocity": velocity,
            "generate/noise": noise,
            "generate/image": image,
        },
        {
            "prompt": PROMPT,
            "size": SIZE,
            "steps": STEPS,
            "cfg_scale": CFG,
            "weights_revision": IRIS_WEIGHTS_REVISION,
            "qwen3_vl_revision": QWEN3_VL_REVISION,
        },
    )


if __name__ == "__main__":
    main()
