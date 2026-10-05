#!/usr/bin/env python3
"""Produce / verify the body-loss reference fixture (epic 2123, sc-24832).

The three body losses (ViTPose+ proportion, HybrIK shape, Sapiens normals) are ports of
ai-toolkit-perceptual @ 6e01a6e (`toolkit/body_id.py`, `toolkit/body_shape.py`,
`toolkit/normal_id.py`). This script runs the *reference implementations* on tiny, seeded
random-init models and a fixed input, and writes everything the Rust ports need to reproduce the
numbers on both backends — the weights (in each upstream checkpoint's own key layout), the inputs,
and every intermediate the losses are built from — to
`docs/migration/body-losses-reference/body_losses_tiny.safetensors`, with its SHA-256 and this
script's SHA-256 recorded in `manifest.json` next to it (`--verify` re-checks both).

Reference implementations:
- ViTPose: HF transformers `VitPoseForPoseEstimation` (the class upstream loads), driven by
  upstream's `DifferentiableBodyProportionEncoder.forward` / `_compute_ratios` verbatim. The only
  substitution is `dsntnn.dsnt`, a 3-line expectation over `normalized_linspace`, reproduced here
  (`dsnt`) because dsntnn is not installed.
- HybrIK: upstream `DifferentiableBodyShapeEncoder` (`_backbone` / `_predict_betas` / `forward`)
  verbatim over torchvision `BasicBlock`s, with the stage widths/blocks shrunk.
- Sapiens: upstream `SapiensNormal` / `_NormalDecoder` / `DifferentiableNormalEncoder.forward`
  verbatim, with the hard-coded sizes (64x48 position grid, 768 decoder width, 512x384 letterbox,
  256 normal size) lifted into constructor arguments, and the LayerNorm eps set to Sapiens' own
  1e-6 (upstream's port uses torch's 1e-5 default).

Deliberate deviations of the port, reproduced here so the fixture pins the port's behaviour:
- the fixture's losses are the per-sample terms before upstream's `t_ratio` scaling, which the port
  applies on the shared path (`PerceptualLoss::timestep_weight`, tested in the ports);
- the normal loss's subject mask is letterboxed with the normals (upstream resizes the raw mask
  straight to the letterboxed map, misaligning it with the padding).

Run with any Python that has torch, torchvision, transformers (>= 4.47) and safetensors.

Other modes (no fixture write):
- `--convert PTH OUT_DIR` re-containers the HybrIK / Sapiens `.pth` checkpoints as
  `model.safetensors` with keys unchanged — the rehost the catalog pins (neither upstream ships
  safetensors);
- `--real IMAGE OUT_DIR [--vitpose DIR] [--hybrik PTH --bbox X1 Y1 X2 Y2] [--sapiens PTH]` writes
  the real-scale reference outputs of the given models that the ports' ignored real-weight parity
  tests read (`SCENEWORKS_BODY_LOSS_REAL`; each model's re-hosted snapshot sits in `OUT_DIR/<model>/`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs/migration/body-losses-reference"
FIXTURE = OUT_DIR / "body_losses_tiny.safetensors"
MANIFEST = OUT_DIR / "manifest.json"
UPSTREAM = "BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual@6e01a6e"

IMG_H, IMG_W = 40, 30
PERSON_BBOX = [5.0, 8.0, 22.0, 36.0]
VIS_THRESHOLD = 0.2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------
# ViTPose (upstream body_id.py)
# ------------------------------------------------------------------------------------------------


def dsnt(hm: torch.Tensor) -> torch.Tensor:
    """dsntnn.dsnt: expected (x, y) under each heatmap, in [-1, 1] over pixel centres."""
    h, w = hm.shape[-2:]
    xs = (torch.arange(w, dtype=hm.dtype) * 2 + 1) / w - 1
    ys = (torch.arange(h, dtype=hm.dtype) * 2 + 1) / h - 1
    x = (hm.sum(-2) * xs).sum(-1)
    y = (hm.sum(-1) * ys).sum(-1)
    return torch.stack([x, y], dim=-1)


def heatmaps_to_coords(heatmaps):
    hm = heatmaps.clamp(min=0)
    hm = hm / hm.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return dsnt(hm)


def compute_ratios(keypoints, visibilities, ref_ratios=None, include_head=False):
    kp = keypoints
    vis = visibilities
    threshold = VIS_THRESHOLD

    def dist(i, j):
        return (kp[:, i] - kp[:, j]).pow(2).sum(-1).clamp(min=1e-6).sqrt()

    def min_vis(*indices):
        return torch.stack([vis[:, i] for i in indices], dim=-1).min(dim=-1).values

    upper_arm = (dist(5, 7) + dist(6, 8)) / 2
    forearm = (dist(7, 9) + dist(8, 10)) / 2
    thigh = (dist(11, 13) + dist(12, 14)) / 2
    shin = (dist(13, 15) + dist(14, 16)) / 2
    shoulder_mid = (kp[:, 5] + kp[:, 6]) / 2
    hip_mid = (kp[:, 11] + kp[:, 12]) / 2
    torso = (shoulder_mid - hip_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
    shoulder_w = dist(5, 6)
    hip_w = dist(11, 12)
    height = torso + thigh + shin
    height = height.clamp(min=1e-4)
    ratio_list = [
        upper_arm / height,
        forearm / height,
        thigh / height,
        shin / height,
        torso / height,
        shoulder_w / hip_w.clamp(min=1e-4),
        upper_arm / forearm.clamp(min=1e-4),
        thigh / shin.clamp(min=1e-4),
    ]
    vis_list = [
        min_vis(5, 6, 7, 8),
        min_vis(7, 8, 9, 10),
        min_vis(11, 12, 13, 14),
        min_vis(13, 14, 15, 16),
        min_vis(5, 6, 11, 12),
        min_vis(5, 6, 11, 12),
        min_vis(5, 6, 7, 8, 9, 10),
        min_vis(11, 12, 13, 14, 15, 16),
    ]
    if include_head:
        head_height = (kp[:, 0] - shoulder_mid).pow(2).sum(-1).clamp(min=1e-6).sqrt()
        ratio_list.append(head_height / height)
        vis_list.append(min_vis(0, 5, 6))
        head_width = dist(3, 4)
        ratio_list.append(head_width / shoulder_w.clamp(min=1e-4))
        vis_list.append(min_vis(3, 4, 5, 6))
    ratios = torch.stack(ratio_list, dim=-1)
    ratio_vis = torch.stack(vis_list, dim=-1)
    if ref_ratios is not None:
        low_conf = ratio_vis < threshold
        ratios = torch.where(low_conf, ref_ratios.detach(), ratios)
        ratio_vis = torch.where(low_conf, torch.zeros_like(ratio_vis), ratio_vis)
    return ratios, ratio_vis


def vitpose_forward(model, pixels, input_size, ref_ratios=None, include_head=False):
    """Upstream DifferentiableBodyProportionEncoder.forward (one sample)."""
    from transformers.models.vitpose.image_processing_vitpose import (
        box_to_center_and_scale,
        get_warp_matrix,
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    _, c, s_h, s_w = pixels.shape
    out_w, out_h = input_size[1], input_size[0]
    center, scale = box_to_center_and_scale(
        [0, 0, s_w, s_h], out_w, out_h, normalize_factor=200.0, padding_factor=1.25
    )
    warp_mat = get_warp_matrix(
        0, center * 2.0, np.array([out_w - 1, out_h - 1], dtype=np.float32), scale * 200.0
    )
    m = np.vstack([warp_mat, [0, 0, 1]])
    m_inv = np.linalg.inv(m)
    s_in = np.array([[2.0 / (s_w - 1), 0, -1], [0, 2.0 / (s_h - 1), -1], [0, 0, 1]])
    s_out_inv = np.array(
        [[(out_w - 1) / 2.0, 0, (out_w - 1) / 2.0], [0, (out_h - 1) / 2.0, (out_h - 1) / 2.0], [0, 0, 1]]
    )
    theta = torch.from_numpy((s_in @ m_inv @ s_out_inv)[:2, :]).float().unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, out_h, out_w), align_corners=True)
    sample = F.grid_sample(pixels, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    sample = (sample - mean) / std
    heatmaps = model(sample, dataset_index=torch.tensor([0])).heatmaps.float()
    coords = heatmaps_to_coords(heatmaps)
    confidence = heatmaps.flatten(2).max(dim=2).values.detach()
    ratios, ratio_vis = compute_ratios(coords, confidence, ref_ratios, include_head)
    return heatmaps, coords, confidence, ratios, ratio_vis


def proportion_loss(ref_ratios, ref_vis, gen_ratios, gen_vis):
    """Upstream SDTrainer body-proportion term for one valid sample (before the t_ratio scale)."""
    combined_vis = torch.min(ref_vis, gen_vis)
    weighted_diff = (gen_ratios - ref_ratios).abs() * combined_vis
    loss = weighted_diff.sum(dim=-1) / combined_vis.sum(dim=-1).clamp(min=1e-6)
    missing = ((ref_vis >= 0.5) & (gen_vis < VIS_THRESHOLD)).float().sum(dim=-1)
    ref_high = (ref_vis >= 0.5).float().sum(dim=-1).clamp(min=1.0)
    return loss + missing / ref_high


def build_vitpose():
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation

    backbone = VitPoseBackboneConfig(
        image_size=[64, 48],
        patch_size=[8, 8],
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=2,
        num_experts=2,
        part_features=4,
        layer_norm_eps=1e-6,
        out_features=["stage2"],
        out_indices=[2],
    )
    model = VitPoseForPoseEstimation(VitPoseConfig(backbone_config=backbone, use_simple_decoder=False, num_labels=17))
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 1.0 / math.sqrt(max(p[0].numel(), 1))))
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(torch.randn_like(b) * 0.1)
            elif name.endswith("running_var"):
                b.copy_(torch.rand_like(b) + 0.5)
        # Spread the keypoint peaks across the visibility threshold so the fixture exercises both
        # the trusted and the low-confidence (reference-substituted) ratio paths.
        model.head.conv.bias.copy_(torch.linspace(0.1, 1.5, 17)[torch.randperm(17)])
    return model.eval()


# ------------------------------------------------------------------------------------------------
# HybrIK (upstream body_shape.py)
# ------------------------------------------------------------------------------------------------


class TinyHybrik(nn.Module):
    """Upstream DifferentiableBodyShapeEncoder with a shrunk torchvision BasicBlock ResNet."""

    def __init__(self, blocks=(1, 1, 1, 1), widths=(4, 4, 8, 8), fc_hidden=8, input_size=32):
        super().__init__()
        from torchvision.models.resnet import BasicBlock, conv1x1

        self.INPUT_SIZE = input_size
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inplanes = widths[0]
        layers = []
        for i, (n, w) in enumerate(zip(blocks, widths)):
            stride = 1 if i == 0 else 2
            down = None
            if stride != 1 or inplanes != w:
                down = nn.Sequential(conv1x1(inplanes, w, stride), nn.BatchNorm2d(w))
            mods = [BasicBlock(inplanes, w, stride, down)]
            inplanes = w
            mods += [BasicBlock(inplanes, w) for _ in range(1, n)]
            layers.append(nn.Sequential(*mods))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(widths[3], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.decshape = nn.Linear(fc_hidden, 10)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.register_buffer("init_shape", torch.zeros(1, 10))
        self.register_buffer("img_mean", torch.tensor([0.406, 0.457, 0.480]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1))

    def _backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def _predict_betas(self, features):
        x = self.drop1(self.fc1(features))
        x = self.drop2(self.fc2(x))
        return self.decshape(x) + self.init_shape

    def forward(self, pixels, person_bboxes=None):
        pixels = pixels.float()
        crops = []
        for i in range(pixels.shape[0]):
            bbox = person_bboxes[i]
            ph, pw = pixels.shape[2], pixels.shape[3]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            cx_bbox = (x1 + x2) / 2
            cy_bbox = (y1 + y2) / 2
            size = max(bw, bh) * 1.25
            half = size / 2
            cx1 = max(0, int(round(float(cx_bbox - half))))
            cy1 = max(0, int(round(float(cy_bbox - half))))
            cx2 = min(pw, int(round(float(cx_bbox + half))))
            cy2 = min(ph, int(round(float(cy_bbox + half))))
            if cx2 > cx1 and cy2 > cy1:
                crop = pixels[i : i + 1, :, cy1:cy2, cx1:cx2]
            else:
                crop = pixels[i : i + 1]
            crop = F.interpolate(
                crop, size=(self.INPUT_SIZE, self.INPUT_SIZE), mode="bilinear", align_corners=False
            )
            crops.append(crop)
        pixels = torch.cat(crops, dim=0)
        pixels = (pixels - self.img_mean) / self.img_std
        return self._predict_betas(self._backbone(pixels))


def build_hybrik():
    model = TinyHybrik()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn_like(p) * math.sqrt(2.0 / p[0].numel()))
            elif name.startswith(("bn", "layer")) and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(torch.randn_like(b) * 0.1)
            elif name.endswith("running_var"):
                b.copy_(torch.rand_like(b) + 0.5)
        for lin in (model.fc1, model.fc2, model.decshape):
            lin.bias.mul_(0.05)
        model.init_shape.copy_(torch.randn(1, 10) * 0.05)
    return model.eval()


def hybrik_checkpoint_keys(model):
    """The HybrIK checkpoint layout: backbone under `preact.`, head at the root, `init_shape` 1-D."""
    out = {}
    for k, v in model.state_dict().items():
        if k in ("img_mean", "img_std") or k.endswith("num_batches_tracked"):
            continue
        if k == "init_shape":
            out["init_shape"] = v.reshape(10)
        elif k.startswith(("fc1.", "fc2.", "decshape.")):
            out[k] = v
        else:
            out["preact." + k] = v
    return out


# ------------------------------------------------------------------------------------------------
# Sapiens (upstream normal_id.py, sizes parameterised)
# ------------------------------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.attn = _Attention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        self.ffn = _FFN(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _NormalDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        deconv_layers = []
        conv_layers = []
        for i in range(3):
            in_c = in_channels if i == 0 else mid_channels
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(in_c, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(mid_channels, affine=False),
                    nn.SiLU(inplace=True),
                ]
            )
            conv_layers.extend(
                [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                    nn.InstanceNorm2d(mid_channels, affine=False),
                    nn.SiLU(inplace=True),
                ]
            )
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_seg = nn.Conv2d(mid_channels, 3, kernel_size=1)

    def forward(self, x):
        for i in range(3):
            x = self.deconv_layers[i * 3 : (i + 1) * 3](x)
            x = self.conv_layers[i * 3 : (i + 1) * 3](x)
        return self.conv_seg(x)


class SapiensNormal(nn.Module):
    def __init__(self, embed_dim=16, num_layers=2, num_heads=2, ffn_dim=32, patch=8, pos=(8, 6), mid=8, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_h, self.pos_w = pos
        self.patch_embed = nn.Module()
        self.patch_embed.projection = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch, padding=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos[0] * pos[1], embed_dim))
        self.layers = nn.ModuleList([_TransformerBlock(embed_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.ln1 = nn.LayerNorm(embed_dim, eps=eps)
        self.decode_head = _NormalDecoder(embed_dim, mid)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
        x = self.patch_embed.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if x.shape[1] != self.pos_embed.shape[1]:
            pe = self.pos_embed.reshape(1, self.pos_h, self.pos_w, self.embed_dim).permute(0, 3, 1, 2)
            pe = F.interpolate(pe.float(), size=(H, W), mode="bilinear", align_corners=False)
            pe = pe.to(x.dtype).permute(0, 2, 3, 1).reshape(1, H * W, self.embed_dim)
            x = x + pe
        else:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.ln1(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.decode_head(x)


def letterbox_tensor(pixels, target_h, target_w):
    B, C, H, W = pixels.shape
    scale = min(target_w / W, target_h / H)
    new_w, new_h = int(W * scale), int(H * scale)
    resized = F.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    pad_r = target_w - new_w - pad_x
    pad_b = target_h - new_h - pad_y
    return F.pad(resized, (pad_x, pad_r, pad_y, pad_b), value=0.0)


def sapiens_forward(model, pixels, train_size=(32, 24), normal_size=16):
    """Upstream DifferentiableNormalEncoder.forward."""
    B, C, H, W = pixels.shape
    target_h, target_w = train_size if H >= W else (train_size[1], train_size[0])
    pixels = letterbox_tensor(pixels, target_h, target_w)
    raw = model(pixels)
    raw = F.interpolate(raw.float(), size=(normal_size, normal_size), mode="bilinear", align_corners=False)
    return raw / (raw.norm(dim=1, keepdim=True) + 1e-5)


def sapiens_mask(mask, train_size=(32, 24), normal_size=16):
    """The port's mask path: the same letterbox + resize as the normals (see module docstring)."""
    B, C, H, W = mask.shape
    target_h, target_w = train_size if H >= W else (train_size[1], train_size[0])
    m = letterbox_tensor(mask, target_h, target_w)
    return F.interpolate(m, size=(normal_size, normal_size), mode="bilinear", align_corners=False)[:, 0]


def normal_loss(ref, gen, mask=None):
    """Upstream SDTrainer normal term for one valid sample (before the t_ratio scale)."""
    cos_per_pixel = (ref * gen).sum(dim=1)
    l1_per_pixel = (ref - gen).abs().mean(dim=1)
    if mask is not None:
        s = mask.sum(dim=(1, 2)).clamp(min=1.0)
        cos_mean = (cos_per_pixel * mask).sum(dim=(1, 2)) / s
        l1_mean = (l1_per_pixel * mask).sum(dim=(1, 2)) / s
    else:
        cos_mean = cos_per_pixel.mean(dim=(1, 2))
        l1_mean = l1_per_pixel.mean(dim=(1, 2))
    return (1.0 - cos_mean) + l1_mean


def build_sapiens():
    model = SapiensNormal()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 1.0 / math.sqrt(max(p[0].numel(), 1))))
    return model.eval()


def sapiens_checkpoint_keys(model):
    """The Sapiens checkpoint layout (`backbone.*`, `decode_head.*`, mmcv FFN naming)."""
    out = {}
    for k, v in model.state_dict().items():
        if k in ("img_mean", "img_std"):
            continue
        if k.startswith("decode_head."):
            out[k] = v
            continue
        k = k.replace("ffn.fc1.", "ffn.layers.0.0.").replace("ffn.fc2.", "ffn.layers.1.")
        out["backbone." + k] = v
    return out


# ------------------------------------------------------------------------------------------------


def to_f16_and_back(model):
    """Round every parameter/buffer to f16 (the fixture's storage precision) so the stored weights
    reproduce the stored outputs exactly. The normalization constants are not stored, so they stay
    exact."""
    with torch.no_grad():
        buffers = [
            b
            for n, b in model.named_buffers()
            if b.is_floating_point() and n not in ("img_mean", "img_std")
        ]
        for p in list(model.parameters()) + buffers:
            p.copy_(p.half().float())


def produce() -> dict[str, torch.Tensor]:
    torch.manual_seed(2123_24832)
    vit, hyb, sap = build_vitpose(), build_hybrik(), build_sapiens()
    for m in (vit, hyb, sap):
        to_f16_and_back(m)
    img_a = torch.rand(1, 3, IMG_H, IMG_W)
    img_b = torch.rand(1, 3, IMG_H, IMG_W) ** 2
    mask = torch.zeros(1, 1, IMG_H, IMG_W)
    mask[:, :, 6:34, 4:24] = 1.0

    t: dict[str, torch.Tensor] = {"input.a": img_a, "input.b": img_b, "input.mask": mask}
    t["input.person_bbox"] = torch.tensor(PERSON_BBOX)
    with torch.no_grad():
        hm_a, co_a, cf_a, r_a, v_a = vitpose_forward(vit, img_a, (64, 48), include_head=True)
        hm_b, co_b, cf_b, r_b, v_b = vitpose_forward(vit, img_b, (64, 48), ref_ratios=r_a, include_head=True)
        t.update(
            {
                "vitpose.out.heatmaps_a": hm_a,
                "vitpose.out.coords_a": co_a,
                "vitpose.out.confidence_a": cf_a,
                "vitpose.out.ratios_a": r_a,
                "vitpose.out.ratio_vis_a": v_a,
                "vitpose.out.ratios_b": r_b,
                "vitpose.out.ratio_vis_b": v_b,
                "vitpose.out.loss": proportion_loss(r_a, v_a, r_b, v_b),
            }
        )
        beta_a = hyb(img_a, [PERSON_BBOX])
        beta_b = hyb(img_b, [PERSON_BBOX])
        t.update(
            {
                "hybrik.out.betas_a": beta_a,
                "hybrik.out.betas_b": beta_b,
                "hybrik.out.cos": F.cosine_similarity(beta_b, beta_a, dim=-1),
                "hybrik.out.l1": (beta_b - beta_a).abs().mean(dim=-1),
            }
        )
        n_a = sapiens_forward(sap, img_a)
        n_b = sapiens_forward(sap, img_b)
        m = sapiens_mask(mask)
        t.update(
            {
                "sapiens.out.normals_a": n_a,
                "sapiens.out.normals_b": n_b,
                "sapiens.out.mask": m,
                "sapiens.out.loss": normal_loss(n_a, n_b),
                "sapiens.out.loss_masked": normal_loss(n_a, n_b, m),
            }
        )
    for k, v in vit.state_dict().items():
        if not k.endswith("num_batches_tracked"):
            t["vitpose.w." + k] = v
    for k, v in hybrik_checkpoint_keys(hyb).items():
        t["hybrik.w." + k] = v
    for k, v in sapiens_checkpoint_keys(sap).items():
        t["sapiens.w." + k] = v
    print(
        "vitpose vis_a", [round(x, 3) for x in v_a[0].tolist()],
        "\nhybrik cos", float(t["hybrik.out.cos"]),
        "\nproportion loss", float(t["vitpose.out.loss"]),
        "normal loss", float(t["sapiens.out.loss"]), float(t["sapiens.out.loss_masked"]),
    )
    out = {}
    for k, v in t.items():
        v = v.detach().contiguous().float()
        out[k] = v.half() if ".w." in k else v
    return out


def torch_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """The tensors of a torch `.pth` checkpoint (`state_dict` when wrapped), keys unchanged."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def convert(src: Path, dst_dir: Path) -> None:
    """Re-container a HybrIK / Sapiens `.pth` as `model.safetensors` with the keys UNCHANGED — the
    layout the ports load (`preact.*` / `backbone.*` + `decode_head.*`). This is the whole rehost
    transformation; no tensor is renamed, cast or dropped (only non-tensor entries)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous() for k, v in torch_checkpoint_state(src).items()}
    save_file(sd, str(dst_dir / "model.safetensors"))
    print("wrote", dst_dir / "model.safetensors", len(sd), "tensors")


def real_reference(image: Path, out: Path, vitpose_dir: Path | None, hybrik_pth: Path | None,
                   sapiens_pth: Path | None, bbox: list[float] | None) -> None:
    """The reference implementations at real scale on `image` — what the ports' ignored
    `real_*_matches_the_reference_implementation` tests compare against. Each model is optional;
    only the given ones write their outputs (never run in ordinary CI)."""
    from PIL import Image

    img = torch.from_numpy(np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0)
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
    t = {"input.a": img}
    with torch.no_grad():
        if vitpose_dir is not None:
            from transformers import VitPoseForPoseEstimation

            vit = VitPoseForPoseEstimation.from_pretrained(str(vitpose_dir)).float().eval()
            hm, co, cf, r, v = vitpose_forward(vit, img, (256, 192), include_head=True)
            t.update({"vitpose.out.heatmaps_a": hm, "vitpose.out.ratios_a": r, "vitpose.out.ratio_vis_a": v})
        if hybrik_pth is not None:
            if bbox is None:
                raise SystemExit("--hybrik needs --bbox X1 Y1 X2 Y2")
            hyb = TinyHybrik(blocks=(3, 4, 6, 3), widths=(64, 128, 256, 512), fc_hidden=1024, input_size=256)
            sd = {}
            for k, v in torch_checkpoint_state(hybrik_pth).items():
                if k == "init_shape":
                    sd[k] = v.reshape(1, 10)
                elif k.startswith("preact."):
                    sd[k[len("preact."):]] = v
                elif k.startswith(("fc1.", "fc2.", "decshape.")):
                    sd[k] = v
            res = hyb.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if k not in ("img_mean", "img_std")]
            assert not missing, missing
            hyb.eval()
            t["input.person_bbox"] = torch.tensor(bbox)
            t["hybrik.out.betas_a"] = hyb(img, [bbox])
        if sapiens_pth is not None:
            sap = SapiensNormal(embed_dim=1024, num_layers=24, num_heads=16, ffn_dim=4096, patch=16,
                                pos=(64, 48), mid=768)
            sd = {}
            for k, v in torch_checkpoint_state(sapiens_pth).items():
                if k.startswith("backbone."):
                    k = k[len("backbone."):].replace("ffn.layers.0.0.", "ffn.fc1.").replace("ffn.layers.1.", "ffn.fc2.")
                elif not k.startswith("decode_head."):
                    continue
                sd[k] = v
            res = sap.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if k not in ("img_mean", "img_std")]
            assert not missing, missing
            sap.eval()
            t["sapiens.out.normals_a"] = sapiens_forward(sap, img, train_size=(512, 384), normal_size=256)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: x.contiguous().float() for k, x in t.items()}, str(out / "reference.safetensors"))
    print("wrote", out / "reference.safetensors", sorted(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check the manifest hashes only")
    ap.add_argument("--convert", nargs=2, metavar=("PTH", "OUT_DIR"),
                    help="re-container a HybrIK/Sapiens .pth as OUT_DIR/model.safetensors (keys unchanged)")
    ap.add_argument("--real", nargs=2, metavar=("IMAGE", "OUT_DIR"),
                    help="real-scale reference outputs for the ports' real-weight parity tests")
    ap.add_argument("--vitpose", metavar="DIR", help="--real: the vitpose-plus-base snapshot")
    ap.add_argument("--hybrik", metavar="PTH", help="--real: upstream hybrik_resnet34.pth")
    ap.add_argument("--sapiens", metavar="PTH", help="--real: upstream sapiens normal 0.3b .pth")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                    help="the person box HybrIK crops to in --real mode")
    args = ap.parse_args()
    script = Path(__file__).resolve()
    if args.convert:
        convert(Path(args.convert[0]), Path(args.convert[1]))
        return
    if args.real:
        opt = lambda x: Path(x) if x else None  # noqa: E731
        real_reference(Path(args.real[0]), Path(args.real[1]), opt(args.vitpose), opt(args.hybrik),
                       opt(args.sapiens), list(args.bbox) if args.bbox else None)
        return
    if args.verify:
        manifest = json.loads(MANIFEST.read_text())
        assert manifest["producer_sha256"] == sha256(script), "producer drifted: regenerate"
        assert manifest["fixture_sha256"] == sha256(FIXTURE), "fixture drifted: regenerate"
        print("ok")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(produce(), str(FIXTURE))
    import torchvision
    import transformers

    MANIFEST.write_text(
        json.dumps(
            {
                "story": "sc-24832",
                "upstream": UPSTREAM,
                "producer": str(script.relative_to(ROOT)),
                "producer_sha256": sha256(script),
                "fixture_sha256": sha256(FIXTURE),
                "environment": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "transformers": transformers.__version__,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", FIXTURE, FIXTURE.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
