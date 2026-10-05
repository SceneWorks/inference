#!/usr/bin/env python3
"""Producer of `latent_perceptual_parity.json` (epic 2123, sc-24833).

Runs the UPSTREAM PyTorch implementations of the two latent-space perceptual losses on
deterministic "formula" weights and inputs, and records their outputs, so the MLX and Candle ports
(`mlx_gen::train::{latent_lpips, vae_anchor}`, `candle_gen::train::{latent_lpips, vae_anchor}`) can
assert parity against upstream without downloading any checkpoint and without committing tens of MB
of random weights: every weight and input is a closed-form function of its key and flat index
that the Rust tests regenerate bit-for-bit (see `formula` below and the ports' `formula_weights`).

Upstream sources (pinned; the script refuses any other checkout):
  * E-LatentLPIPS: github.com/mingukkang/elatentlpips @ f64cd65 — `ELatentLPIPS.forward`,
    `LatentVGG16BN` (VGG16-BN with the latent-channel first conv and the first three max-pools
    replaced by identity), `CalibratedLatentVGG16BN`'s slicing, `normalize_tensor`.
    Called exactly as ai-toolkit-perceptual's SDTrainer does: normalize=False, add_l1_loss=True,
    ensembling=False (augment=None).
  * VAE anchor: github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual @ 6e01a6e —
    `toolkit/vae_anchor.py` (`VAEAnchorEncoder` feature hooks + `compute_loss`) over the FLUX.2
    `extensions_built_in/diffusion_models/flux2/src/autoencoder.py` `Encoder` (width ch=32 here to
    keep the test light; the production width 128 differs only in channel counts).

Usage:
  python3 produce_latent_perceptual_parity.py --elatentlpips <checkout> --ai-toolkit <checkout>

Formula (all in float64, then cast to float32):
  phase(key) = (fnv1a32(utf8(key)) mod 10007) / 1000
  wave(key)[i] = sin(0.7 * i + phase(key)),  i = flat (row-major) index
  conv / linear weight with fan_in f:  sqrt(2 / f) * wave
  1-D "bias" / BN bias:                0.1 * wave
  BN weight:                            1 + 0.2 * wave
  BN running_mean:                      0.1 * wave
  BN running_var:                       1 + 0.5 * wave^2
  E-LatentLPIPS lin head weight:        0.05 * |wave| + 0.01
  GroupNorm weight / bias (VAE):        1 + 0.2 * wave / 0.1 * wave
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import torch

ELATENTLPIPS_REV = "f64cd65"
AI_TOOLKIT_REV = "6e01a6e"


def fnv1a32(key: str) -> int:
    h = 2166136261
    for b in key.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def wave(key: str, n: int) -> np.ndarray:
    phase = (fnv1a32(key) % 10007) / 1000.0
    return np.sin(0.7 * np.arange(n, dtype=np.float64) + phase)


def formula(key: str, shape, role: str) -> torch.Tensor:
    n = int(np.prod(shape)) if len(shape) else 1
    w = wave(key, n)
    if role == "conv":
        fan_in = int(np.prod(shape[1:]))
        v = math.sqrt(2.0 / fan_in) * w
    elif role == "bias":
        v = 0.1 * w
    elif role == "bn_weight":
        v = 1.0 + 0.2 * w
    elif role == "bn_mean":
        v = 0.1 * w
    elif role == "bn_var":
        v = 1.0 + 0.5 * w * w
    elif role == "lin":
        v = 0.05 * np.abs(w) + 0.01
    else:
        raise ValueError(role)
    return torch.from_numpy(v.astype(np.float32)).reshape(shape)


def input_wave(n: int, a: float, f: float, p: float) -> np.ndarray:
    return a * np.sin(f * np.arange(n, dtype=np.float64) + p)


def check_rev(path: str, rev: str):
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    if not head.startswith(rev):
        sys.exit(f"{path} is at {head}, expected {rev}")


def lpips_role(key: str, t: torch.Tensor) -> str:
    if key.startswith("lin"):
        return "lin"
    if key.endswith("running_mean"):
        return "bn_mean"
    if key.endswith("running_var"):
        return "bn_var"
    if t.dim() == 4:
        return "conv"
    if key.endswith("weight"):
        return "bn_weight"
    return "bias"


def lpips_cases(el_path: str):
    sys.path.insert(0, el_path)
    # ai-toolkit-perceptual constructs ELatentLPIPS with augment=None, so the ADA augmentation pipe
    # (which drags in scipy + custom CUDA ops) is never used; stub its module so the import resolves.
    stub = types.ModuleType("elatentlpips.ada_aug")

    class AdaAugment:  # noqa: D401 — never instantiated with augment=None
        pass

    stub.AdaAugment = AdaAugment
    sys.modules["elatentlpips.ada_aug"] = stub
    from elatentlpips.elatentlpips import ELatentLPIPS
    from elatentlpips.vgg16 import LatentVGG16BN

    out = []
    for family, ch in [("sdxl", 4), ("sd3", 16)]:
        model = ELatentLPIPS(pretrained=False, pnet_rand=True, encoder=family, augment=None,
                             verbose=False)
        # CalibratedLatentVGG16BN's pretrained branch slices the LATENT trunk 0-7/7-14/14-24/24-34/
        # 34-44 (its pretrained=False branch would use the 3-channel torchvision VGG instead), so
        # rebuild the slices from LatentVGG16BN exactly as that branch does.
        feats = LatentVGG16BN(ch).model.features
        for s, (lo, hi) in enumerate([(0, 7), (7, 14), (14, 24), (24, 34), (34, 44)], start=1):
            seq = torch.nn.Sequential()
            for x in range(lo, hi):
                seq.add_module(str(x), feats[x])
            setattr(model.net, f"slice{s}", seq)
        sd = model.state_dict()
        new = {}
        for k, t in sd.items():
            if k.endswith("num_batches_tracked") or k.startswith("lins."):
                continue  # `lins.k` aliases `lin{k}`; filled through it.
            new[k] = formula(k, tuple(t.shape), lpips_role(k, t))
        missing = model.load_state_dict(new, strict=False)
        assert all(m.startswith("lins.") or m.endswith("num_batches_tracked")
                   for m in missing.missing_keys), missing
        model.eval()
        h = w = 16
        n = ch * h * w
        x0 = input_wave(n, 0.8, 0.37, 0.5)
        x1 = x0 + input_wave(n, 0.25, 1.3, 2.0)
        in0 = torch.from_numpy(x0.astype(np.float32)).reshape(1, ch, h, w)
        in1 = torch.from_numpy(x1.astype(np.float32)).reshape(1, ch, h, w)
        with torch.no_grad():
            val, per = model(in0, in1, retPerLayer=True, normalize=False, ensembling=False,
                             add_l1_loss=True)
            same = model(in0, in0, normalize=False, ensembling=False, add_l1_loss=True)
            l1 = (in0 - in1).abs().mean().item()
        out.append({
            "family": family,
            "latent_channels": ch,
            "h": h,
            "w": w,
            "inputs": {"in0": [0.8, 0.37, 0.5], "in1_delta": [0.25, 1.3, 2.0]},
            "value": val.item(),
            "per_layer": [p.item() for p in per],
            "l1": l1,
            "self_value": same.item(),
        })
    return out


def bfl_to_diffusers(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "down":
        i = parts[1]
        if parts[2] == "block":
            leaf = "conv_shortcut" if parts[4] == "nin_shortcut" else parts[4]
            return f"encoder.down_blocks.{i}.resnets.{parts[3]}.{leaf}.{parts[5]}"
        if parts[2] == "downsample":
            return f"encoder.down_blocks.{i}.downsamplers.0.conv.{parts[4]}"
    if parts[0] == "mid":
        if parts[1] in ("block_1", "block_2"):
            j = 0 if parts[1] == "block_1" else 1
            leaf = "conv_shortcut" if parts[2] == "nin_shortcut" else parts[2]
            return f"encoder.mid_block.resnets.{j}.{leaf}.{parts[3]}"
        if parts[1] == "attn_1":
            leaf = {"norm": "group_norm", "q": "to_q", "k": "to_k", "v": "to_v",
                    "proj_out": "to_out.0"}[parts[2]]
            return f"encoder.mid_block.attentions.0.{leaf}.{parts[3]}"
    return "encoder." + name


def vae_anchor_case(at_path: str):
    sys.path.insert(0, at_path)
    spec = importlib.util.spec_from_file_location(
        "flux2_autoencoder",
        os.path.join(at_path, "extensions_built_in/diffusion_models/flux2/src/autoencoder.py"))
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)
    from toolkit.vae_anchor import VAEAnchorEncoder

    ch = 32
    enc = ae.Encoder(resolution=32, in_channels=3, ch=ch, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
                     z_channels=32)
    sd = enc.state_dict()
    new = {}
    for k, t in sd.items():
        dk = bfl_to_diffusers(k)
        shape = tuple(t.shape)
        if t.dim() == 4 and shape[2:] == (1, 1) and ".attentions." in dk:
            # diffusers stores the mid-attention projections as Linear [C, C]; same flat order.
            v = formula(dk, (shape[0], shape[1]), "conv").reshape(shape)
        elif t.dim() == 4:
            v = formula(dk, shape, "conv")
        elif dk.endswith("weight") and ("norm" in dk):
            v = formula(dk, shape, "bn_weight")
        else:
            v = formula(dk, shape, "bias")
        new[k] = v
    enc.load_state_dict(new, strict=True)
    enc.eval()
    va = VAEAnchorEncoder()
    va._encoder = enc
    va._register_hooks()
    va._loaded = True
    h = w = 32
    n = 3 * h * w
    p = input_wave(n, 0.9, 0.11, 0.3)
    r = np.clip(p + input_wave(n, 0.3, 0.7, 1.1), -1.0, 1.0)
    pred = torch.from_numpy(p.astype(np.float32)).reshape(1, 3, h, w)
    ref = torch.from_numpy(r.astype(np.float32)).reshape(1, 3, h, w)
    with torch.no_grad():
        _, pf = va.encode_with_features(pred)
        _, rf = va.encode_with_features(ref)
        loss, per_level = VAEAnchorEncoder.compute_loss(pf, rf)
        self_loss, _ = VAEAnchorEncoder.compute_loss(rf, rf)
    stats = {k: {"shape": list(v.shape), "mean": v.mean().item(), "abs_mean": v.abs().mean().item()}
             for k, v in pf.items()}
    return {
        "ch": ch,
        "h": h,
        "w": w,
        "inputs": {"pred": [0.9, 0.11, 0.3], "ref_delta": [0.3, 0.7, 1.1], "ref_clip": [-1.0, 1.0]},
        "loss": loss.item(),
        "per_level": per_level,
        "self_loss": self_loss.item(),
        "pred_feature_stats": stats,
    }


def write_tiny_state_dict(path: str):
    """A `torch.save`d module state dict with a BatchNorm — the format the published E-LatentLPIPS
    checkpoints use (float tensors + int64 `num_batches_tracked` + the `_metadata` attribute) — so
    the Rust `.pth` readers are exercised on the real serialization without the 59 MB file."""
    m = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 3), torch.nn.BatchNorm2d(2))
    sd = m.state_dict()
    for k, t in sd.items():
        if k.endswith("num_batches_tracked"):
            t.fill_(7)
            continue
        role = {"0.weight": "conv", "0.bias": "bias", "1.weight": "bn_weight", "1.bias": "bias",
                "1.running_mean": "bn_mean", "1.running_var": "bn_var"}[k]
        t.copy_(formula(k, tuple(t.shape), role))
    torch.save(sd, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elatentlpips", required=True)
    ap.add_argument("--ai-toolkit", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "latent_perceptual_parity.json"))
    a = ap.parse_args()
    check_rev(a.elatentlpips, ELATENTLPIPS_REV)
    check_rev(a.ai_toolkit, AI_TOOLKIT_REV)
    torch.manual_seed(0)
    with open(__file__, "rb") as f:
        producer_sha = hashlib.sha256(f.read()).hexdigest()
    doc = {
        "producer": os.path.basename(__file__),
        "producer_sha256": producer_sha,
        "torch": torch.__version__,
        "upstream": {"elatentlpips": ELATENTLPIPS_REV, "ai_toolkit_perceptual": AI_TOOLKIT_REV},
        "latent_lpips": lpips_cases(a.elatentlpips),
        "vae_anchor": vae_anchor_case(a.ai_toolkit),
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    write_tiny_state_dict(os.path.join(os.path.dirname(a.out), "tiny_bn_state_dict.pth"))


if __name__ == "__main__":
    main()
