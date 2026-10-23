#!/usr/bin/env python3
"""Regenerate tiny sc-26683 fixtures; test oracle only, never an inference dependency.

Executes the pinned Kitchen Hadamard helpers on CPU Torch and explicitly models
its CUDA rowwise quantizer's source-dtype rounding points. No checkpoint payload
is fetched. The committed tensors are consumed by both native backend tests.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

KITCHEN = "f8b7e97a1418d2e529bd5405fbf86312be198ae6"
PRODUCER = "d6797787e6bdb1a1fb0094d588a26f8e71a1c757"
LOADER = "ad1d4e072bd2ed5eec1d30d07ec390f327a3af52"
HF = "a56850c870825d6a8ae501a595c4dbbbc77bbc7e"


def source(repo, revision, path):
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(1024 * 1024)
    return data, {"url": url, "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper, helper_provenance = source("Comfy-Org/comfy-kitchen", KITCHEN, "comfy_kitchen/tensor/int8_utils.py")
    _, cuda_provenance = source("Comfy-Org/comfy-kitchen", KITCHEN, "comfy_kitchen/backends/cuda/ops/int8_linear.cu")
    _, producer_provenance = source("Comfy-Org/comfy-model-tools", PRODUCER, "quant_int8_convrot.py")
    namespace = {"__name__": "frozen_kitchen_int8_utils"}
    exec(compile(helper, helper_provenance["url"], "exec"), namespace)
    cases = []
    for group, k in [(64, 192), (256, 768)]:
        h = namespace["_build_hadamard"](group, "cpu", torch.float32)
        n, m = 16, 5
        j = torch.arange(k)
        rows = torch.stack([
            torch.zeros(k),
            (j % 17 - 8).float() / 4,
            (j % 7 - 3).float() / 32,
            ((j * 13) % 23 - 11).float() / 64,
            torch.zeros(k),
        ])
        rows[2, 0] = 64.0
        ties = torch.zeros(1, k)
        ties[0, :7] = torch.tensor([127.0, .5, 1.5, 2.5, -.5, -1.5, -2.5])
        rows[4] = namespace["_rotate_activation"](ties, h, group)[0]
        codes = (((torch.arange(n)[:, None] * 19 + j[None, :] * 7) % 255) - 127).to(torch.int8)
        weight_scales = (torch.arange(n).float() + 1).reshape(n, 1) / 128
        bias = (torch.arange(n).float() - 8) / 32
        for dtype_name, dtype in [("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)]:
            x = rows.to(dtype)
            rotated = namespace["_rotate_activation"](x, h, group)
            # int8_linear.cu quantize_int8_rowwise_kernel: scales stored as raw F32;
            # only the divisor and quotient are rounded through InputType.
            peak = rotated.float().abs().amax(dim=1, keepdim=True)
            scales = (peak * torch.tensor(1.0 / 127.0, dtype=torch.float32)).clamp_min(1e-30)
            quotient = (rotated.float() / scales.to(dtype).float()).to(dtype).float()
            rounded = torch.round(quotient)
            # CUDA fmaxf(-128, NaN) returns -128 (zero f16 rows underflow the divisor).
            rounded = torch.nan_to_num(rounded, nan=-128.0).clamp(-128, 127)
            q = rounded.to(torch.int8)
            acc = q.to(torch.int32) @ codes.to(torch.int32).T
            y = ((acc.float() * scales) * weight_scales.T + bias.to(dtype).float()).to(dtype)
            cases.append({"group": group, "k": k, "n": n, "m": m, "dtype": dtype_name,
                          "input": x.float().flatten().tolist(), "activation_codes": q.flatten().tolist(),
                          "activation_scales": scales.flatten().tolist(), "expected": y.float().flatten().tolist()})
    result = {"story": "sc-26683", "kitchen_revision": KITCHEN, "producer_revision": PRODUCER,
              "w4a8_loader_revision_not_int8_authority": LOADER, "hf_revision": HF,
              "reference_helper": helper_provenance, "reference_cuda": cuda_provenance,
              "reference_producer": producer_provenance, "torch": torch.__version__,
              "device": "cpu", "cases": cases}
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} tensor-valued CPU oracle cases to {args.output}")


if __name__ == "__main__":
    main()
