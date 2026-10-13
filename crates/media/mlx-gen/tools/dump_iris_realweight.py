"""Real-weight Iris-3B reference for the native port's ignored real-weight test (sc-25679).

Runs the FROZEN upstream (see `_iris_common.py`) on CPU against the pinned weights: the release
`Qwen3VLTextEncoder` (bf16, its configured dtype) on one prompt and the empty CFG null, one FP32
backbone + pixel-head forward on fixed inputs, and a short upstream `generate()` at 256x256 from
fixed injected noise. It writes two files:

* the full golden (tens of MB, every padded conditioning row in f32) to `IRIS_REAL_GOLDEN`, OUTSIDE
  the repo — never committed; the MLX twin's ignored test reads it there;
* the compact committed fixture `mlx-gen-iris/tests/fixtures/iris_real_golden_256.safetensors`
  (~4.4 MB) that the Candle real-weight tests read on any runner, CUDA included. It drops nothing
  the tests compare: the conditioning keeps only the real (mask = 1) rows — the pad rows are
  asserted all-zero first — stored bf16, which is lossless because the release tower computes in
  bf16 (asserted by round-trip); masks, inputs and outputs are stored as in the full golden.

Inputs:
  IRIS_SRC            upstream checkout at the pinned commit
  IRIS_WEIGHTS_DIR    `speridlabs/iris-3b` snapshot root at the pinned revision
  IRIS_TEXT_ENCODER_DIR  `Qwen/Qwen3-VL-4B-Instruct` snapshot at the pinned revision
  IRIS_REAL_GOLDEN    output .safetensors path (outside the repository)

Run: `python -I tools/dump_iris_realweight.py` in the isolated reference venv (CPU, ~25 GB RAM,
~2 h: the 20-step CFG generate is 40 FP32 3B forwards on CPU).

`python -I tools/dump_iris_realweight.py --compact-from <full golden>` re-derives only the committed
fixture from an existing full golden (no upstream, no weights), through the same `save_compact`.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from _iris_common import (
    FIXTURE_DIR,
    IRIS_WEIGHTS_REVISION,
    QWEN3_VL_REVISION,
    import_upstream,
    save_safetensors,
)
from _paths import require_env

import torch  # noqa: E402

PROMPT = "a red fox sleeping in fresh snow, golden hour"
SIZE = 256
STEPS = 20
CFG = 3.0
SEED = 1234


def main() -> None:
    import_upstream()
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from iris3b.sampling import generate, load_for_inference
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

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

    tensors = {
        "cond/embeddings": cond.embeddings[0].float(),
        "cond/mask": cond.mask[0].to(torch.int32),
        "null/embeddings": null.embeddings[0].float(),
        "null/mask": null.mask[0].to(torch.int32),
        "forward/x": x,
        "forward/t": t,
        "forward/velocity": velocity,
        "generate/noise": noise,
        "generate/image": image,
    }
    metadata = {
        "prompt": PROMPT,
        "size": SIZE,
        "steps": STEPS,
        "cfg_scale": CFG,
        "weights_revision": IRIS_WEIGHTS_REVISION,
        "qwen3_vl_revision": QWEN3_VL_REVISION,
    }
    save_safetensors(out, tensors, metadata)
    save_compact(FIXTURE_DIR / COMPACT_NAME, tensors, metadata)


COMPACT_NAME = "iris_real_golden_256.safetensors"


def save_compact(path: Path, tensors: dict, metadata: dict) -> None:
    """The committed form of the full golden: conditioning reduced to its real rows, in bf16.

    Both reductions are asserted lossless, so the Candle test that zero-pads the rows back and
    widens to f32 compares against exactly the values the full golden holds.
    """
    from safetensors.torch import save_file

    compact = {}
    for key, value in tensors.items():
        value = value.detach().contiguous()
        if key.endswith("/embeddings"):
            mask = tensors[key.replace("/embeddings", "/mask")]
            real = int(mask.sum())
            if not bool(mask[:real].all()):
                raise SystemExit(f"{key}: mask is not a contiguous prefix of real rows")
            if not bool((value[real:] == 0).all()):
                raise SystemExit(f"{key}: pad rows are not all zero")
            rows = value[:real].float()
            packed = rows.to(torch.bfloat16)
            if not bool((packed.float() == rows).all()):
                raise SystemExit(f"{key}: values are not bf16-exact")
            compact[key] = packed.contiguous()
        elif value.is_floating_point():
            compact[key] = value.to(torch.float32)
        else:
            compact[key] = value
    meta = {k: str(v) for k, v in metadata.items()}
    meta["embeddings_layout"] = "real rows only (mask = 1 prefix), bf16; pad rows are zero"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(compact, str(path), metadata=meta)
    print(f"wrote {path} ({len(compact)} tensors, {path.stat().st_size} bytes)")


def compact_from(full: Path) -> None:
    from safetensors import safe_open

    with safe_open(str(full), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        metadata = dict(f.metadata())
    save_compact(FIXTURE_DIR / COMPACT_NAME, tensors, metadata)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--compact-from":
        compact_from(Path(sys.argv[2]))
    else:
        main()
