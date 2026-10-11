"""Dump the Iris-3B **depth-training** golden for the native depth trainers (sc-25687).

The frozen release has no downstream training loop, so this oracle is the paper's recipe (arXiv
2610.09450v1 §5.2) written in torch over the FROZEN upstream model: `iris3b/downstream/depth.py`
`IrisDepth` (widened input projections, RGB ‖ zero channel at t = num_train_timesteps, empty-prompt
conditioning, 1x1 reducer) loaded from the miniature export of `dump_iris_depth.py`
(`tests/fixtures/tiny-depth`) with upstream's strict `load_export`, on CPU in fp32:

* target: per-image relative log depth — `log(d)` with the 2nd / 98th percentiles (numpy `linear`)
  of the **valid** ground truth mapped to -1 / +1 (paper "normalized per image by the 2nd and 98th
  percentiles of the valid ground truth"; `depth.py` docstring "mapped to -1 (near) and +1 (far)");
* loss: Eq. 5, `1/|M| Σ_{p∈M} |d̂ - d| + 5 ||∇d̂ - ∇d||₁` with forward differences in x and y, a
  difference pair counted when both pixels are valid, `M` = every valid pixel of the batch;
* one optimizer step: `clip_grad_norm_(1.0)` then `torch.optim.AdamW(betas=(0.9, 0.999),
  weight_decay=0.01)` (the paper's optimizer; the oracle uses lr 1e-3 so the step is visible).

Writes `mlx-gen-iris/tests/fixtures/iris_depth_train_golden.safetensors`: the batch (`rgb`, raw
`depth`, `mask`, normalized `target`, `valid`), `pred0` / `loss0`, the gradient and post-step value
of selected tensors (`grad/<key>`, `step1/<key>`, native key names = export keys minus `pixel.`), the
per-tensor gradient L2 norms of every tensor (`grad_norms`, key order in the metadata), the global
pre-clip norm and `loss1` after the step.

Run: `IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_depth_train.py` (torch 2.8 CPU, numpy,
safetensors, omegaconf — see `_iris_common.py`).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _iris_common import FIXTURE_DIR, IRIS_COMMIT, import_upstream, save_safetensors  # noqa: E402

ROOT = import_upstream()

DEPTH_SHA256 = {
    "src/iris3b/downstream/__init__.py": "94034cdb0da5189b07f01b5f57f2a0cd04cda14cfb7cf9df1bc77e375cd02cec",
    "src/iris3b/downstream/depth.py": "5ca54ceacf0ec5b583d3b25cf6e07967b912bb31179518c70d0e47af281a118f",
}
for rel, want in DEPTH_SHA256.items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from iris3b.downstream import load_export  # noqa: E402
from iris3b.downstream.depth import IrisDepth  # noqa: E402

EXPORT_DIR = FIXTURE_DIR / "tiny-depth"
OUT = FIXTURE_DIR / "iris_depth_train_golden.safetensors"
B, H, W = 2, 16, 16
GRADIENT_WEIGHT = 5.0
LR, BETAS, WEIGHT_DECAY, CLIP = 1e-3, (0.9, 0.999), 0.01, 1.0
SELECTED = [
    "depth_reducer.weight",
    "depth_reducer.bias",
    "s_embedder.proj.weight",
    "pixel_embedder.proj.weight",
    "final_layer.linear.weight",
    "blocks.2.mlp.w1.weight",
    "y_embedder.refiner.proj.weight",
    "t_embedder.mlp.0.weight",
]


def native(key: str) -> str:
    return key.removeprefix("pixel.")


def normalize(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """2nd / 98th percentiles of the valid log depth -> -1 / +1; invalid pixels 0."""
    logs = torch.log(torch.from_numpy(depth[valid].astype(np.float32))).numpy()
    lo, hi = np.percentile(logs.astype(np.float64), [2.0, 98.0])
    out = np.zeros_like(depth, dtype=np.float32)
    full = torch.log(torch.from_numpy(np.where(valid, depth, 1.0).astype(np.float32))).numpy()
    out[valid] = (2.0 * (full[valid].astype(np.float64) - lo) / (hi - lo) - 1.0).astype(np.float32)
    return out


def eq5(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    value = ((pred - target).abs() * valid).sum()
    dxp, dxt = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    dyp, dyt = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    grad = ((dxp - dxt).abs() * vx).sum() + ((dyp - dyt).abs() * vy).sum()
    return (value + GRADIENT_WEIGHT * grad) / valid.sum()


def main() -> None:
    cfg, _, weights, prompt = load_export(str(EXPORT_DIR), "depth")
    model = IrisDepth(cfg.model)
    model.num_train_timesteps = cfg.flow.num_train_timesteps
    model.load_state_dict(weights, strict=True)
    model = model.float().train()
    embeddings = prompt["embeddings"].float()
    mask_tokens = prompt["mask"].bool()

    g = torch.Generator().manual_seed(25687)
    rgb = torch.rand(B, 3, H, W, generator=g) * 2 - 1
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    depth = torch.stack([
        (1.0 + 0.3 * xx + 0.1 * yy).float() * torch.exp(0.2 * torch.randn(H, W, generator=g)),
        (5.0 + 2.0 * torch.sin(xx / 3.0) + 0.5 * yy).float() * torch.exp(0.1 * torch.randn(H, W, generator=g)),
    ]).numpy().astype(np.float32)
    mask = (torch.rand(B, H, W, generator=g) > 0.2).numpy()
    target = np.stack([normalize(depth[i], mask[i]) for i in range(B)])[:, None]
    valid = mask.astype(np.float32)[:, None]
    target_t, valid_t = torch.from_numpy(target), torch.from_numpy(valid)

    pred0 = model(rgb, embeddings, mask_tokens)
    loss0 = eq5(pred0, target_t, valid_t)
    model.zero_grad()
    loss0.backward()
    named = dict(model.named_parameters())
    grads = {native(k): p.grad.detach().clone() for k, p in named.items()}
    keys = sorted(grads)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY, eps=1e-8)
    opt.step()
    with torch.no_grad():
        loss1 = eq5(model(rgb, embeddings, mask_tokens), target_t, valid_t)

    tensors = {
        "rgb": rgb,
        "depth": torch.from_numpy(depth),
        "mask": torch.from_numpy(valid[:, 0].copy()),
        "target": target_t,
        "valid": valid_t,
        "pred0": pred0.detach(),
        "loss0": loss0.detach().reshape(1),
        "loss1": loss1.reshape(1),
        "total_norm": total.reshape(1),
        "grad_norms": torch.stack([grads[k].norm() for k in keys]),
    }
    for k in SELECTED:
        tensors[f"grad/{k}"] = grads[k]
        tensors[f"step1/{k}"] = dict((native(n), p) for n, p in named.items())[k].detach()
    meta = {
        "upstream_commit": IRIS_COMMIT,
        "torch": torch.__version__,
        "grad_norm_keys": json.dumps(keys),
        "selected": json.dumps(SELECTED),
        "lr": LR,
        "betas": json.dumps(BETAS),
        "weight_decay": WEIGHT_DECAY,
        "clip": CLIP,
        "gradient_weight": GRADIENT_WEIGHT,
    }
    print(f"loss0 {loss0.item():.6f} -> loss1 {loss1.item():.6f}, pre-clip norm {total.item():.4f}")
    save_safetensors(OUT, tensors, meta)


if __name__ == "__main__":
    main()
