"""Producer of the committed face-loss parity fixture (epic 2123, sc-24831, AC3).

    python3 crates/media/face_loss_fixtures/produce_face_loss_fixtures.py

writes `face_loss_fixtures.json` next to this script. The fixture is the **torch reference** output of
the two decoded-x0 face losses — upstream ai-toolkit-perceptual's `DifferentiableFaceEncoder.forward`
(bbox crop + 15 % pad, zero pad to square, `F.interpolate(112, bilinear, align_corners=False)`,
`(px·255 − 127.5)/127.5`, ArcFace, L2 normalize) and `DifferentiableLandmarkEncoder.forward` (bbox crop
+ 15 % pad, `F.interpolate(256)`, FaceMesh, `out[0].reshape(B,478,3)[..., :2]`, nose/inter-eye
normalization, region-weighted landmark distance) — on **synthetic** weights and images that both
native backends regenerate bit-exactly from the same counter-based generator (`synth` below; Rust
twins in `mlx-gen-face` / `candle-gen-face` `train::synth`). Both ports must reproduce it, so the
fixture pins MLX ≈ torch ≈ Candle without any downloaded checkpoint.

Two deliberate differences from upstream, both shared by reference and live paths so the loss is
unchanged in kind: the ArcFace input stays **RGB** (insightface's canonical `swapRB=True`
preprocessing, which the existing native glintr100 port and its onnx goldens use — upstream flips to
BGR on both sides), and the FaceMesh program is a **synthetic** network with the real checkpoint's op
set (conv / depthwise conv / PReLU / max-pool / pad incl. channel pad / add / mul / sigmoid /
reshape), lowered through `mlx-gen/tools/fx_program.py` exactly as the real one is — so the fixture
also pins the converter.

The identity numbers are upstream's bias-centred cosines (`normalize(e - noise_mean)`, the mean
unit embedding of 200 counter-based noise images), including the dataset-average mode's clean-cos
normalizer and noise probes that must land below the 0.2 gate.

The committed JSON records this script's sha256; the Rust tests refuse a fixture whose producer bytes
changed without regenerating it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mlx-gen", "tools"))
import fx_program  # noqa: E402

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


# ---------------------------------------------------------------------------------------------
# Counter-based synthetic generator (mirrors gen_core::train::splitmix64 + `train::synth`).
# ---------------------------------------------------------------------------------------------
def splitmix64_np(z: np.ndarray) -> np.ndarray:
    z = (z + np.uint64(GOLDEN)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return z ^ (z >> np.uint64(31))


def splitmix64(z: int) -> int:
    return int(splitmix64_np(np.array([z], dtype=np.uint64))[0])


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & M64
    return h


def uniform(seed: int, key: str, n: int) -> np.ndarray:
    """`n` values in [0, 1) (f64): u_i = (splitmix64(k + i) >> 40) / 2^24, k = splitmix64(seed ^ fnv(key))."""
    k = splitmix64((seed ^ fnv1a64(key)) & M64)
    with np.errstate(over="ignore"):
        idx = (np.uint64(k) + np.arange(n, dtype=np.uint64)).astype(np.uint64)
        z = splitmix64_np(idx)
    return (z >> np.uint64(40)).astype(np.float64) / 16777216.0


def gaussian(seed: int, key: str, n: int) -> np.ndarray:
    """`n` standard normals: Box-Muller `sqrt(-2 ln(u0 + 2^-25)) * cos(2 pi u1)` over `uniform` pairs
    (twin of gen_core `face_loss::synth::gaussian`)."""
    u = uniform(seed, key, 2 * n)
    r = np.sqrt(-2.0 * np.log(u[0::2] + 2.0 ** -25))
    return r * np.cos(2.0 * np.pi * u[1::2])


def identity_noise_image(index: int, edge: int) -> torch.Tensor:
    """Upstream's bias-direction noise `clamp(randn*0.3+0.5, 0, 1)` as NCHW [1,3,edge,edge]
    (generated NHWC; twin of `synth::identity_noise_image`)."""
    z = gaussian(IDENTITY_NOISE_SEED, f"identity-noise-{index}", edge * edge * 3)
    v = np.clip(z * 0.3 + 0.5, 0.0, 1.0).astype(np.float32).reshape(1, edge, edge, 3)
    return torch.from_numpy(v).permute(0, 3, 1, 2).contiguous()


def role(key: str, shape: tuple[int, ...]) -> tuple[float, float]:
    """(offset, half-range) of a synthetic tensor, by key/shape — identical in the Rust twins."""
    if "prelu" in key or key.endswith(".slope"):
        return 0.25, 0.05
    if key.endswith(".scale"):
        return 1.0, 0.1
    if key.endswith(".shift") or key.endswith(".bias"):
        return 0.0, 0.05
    if len(shape) >= 2:
        fan_in = int(np.prod(shape)) // shape[0]
        return 0.0, math.sqrt(3.0 / fan_in)
    return 0.0, 0.1


def synth(seed: int, key: str, shape: tuple[int, ...]) -> torch.Tensor:
    off, half = role(key, shape)
    n = int(np.prod(shape))
    v = off + (2.0 * uniform(seed, key, n) - 1.0) * half
    return torch.from_numpy(v.astype(np.float32).reshape(shape))


def image(seed: int, key: str, h: int, w: int) -> torch.Tensor:
    """A synthetic NCHW `[1,3,h,w]` image in [0, 1] (generated in NHWC order)."""
    v = uniform(seed, key, h * w * 3).astype(np.float32).reshape(1, h, w, 3)
    return torch.from_numpy(v).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------------------------
# Tiny IResNet (the native ArcFace key schema; conv kernels stored OHWI like the converted files).
# ---------------------------------------------------------------------------------------------
ARC_SEED = 0x24831A
ARC_LAYERS = [1, 2, 1, 1]
ARC_WIDTHS = [8, 16, 32, 64]
ARC_STEM = 8
ARC_EMB = 32


def arcface_keys() -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {}

    def conv(p, cin, cout, k):
        keys[f"{p}.weight"] = (cout, k, k, cin)
        keys[f"{p}.bias"] = (cout,)

    def aff(p, c):
        keys[f"{p}.scale"] = (c,)
        keys[f"{p}.shift"] = (c,)

    conv("stem.conv", 3, ARC_STEM, 3)
    keys["stem.prelu.weight"] = (ARC_STEM,)
    cin = ARC_STEM
    for li, (nb, cout) in enumerate(zip(ARC_LAYERS, ARC_WIDTHS)):
        for b in range(nb):
            p = f"layer{li + 1}.{b}"
            bin_ = cin if b == 0 else cout
            aff(f"{p}.bn1", bin_)
            conv(f"{p}.conv1", bin_, cout, 3)
            keys[f"{p}.prelu.weight"] = (cout,)
            conv(f"{p}.conv2", cout, cout, 3)
            if b == 0:
                conv(f"{p}.downsample", bin_, cout, 1)
        cin = cout
    aff("bn2", cin)
    keys["fc.weight"] = (ARC_EMB, cin * 7 * 7)
    keys["fc.bias"] = (ARC_EMB,)
    aff("features", ARC_EMB)
    return keys


def arcface_weights():
    return {k: synth(ARC_SEED, k, s) for k, s in arcface_keys().items()}


def arcface_forward(w, x):
    """`x` NCHW [N,3,112,112] normalized → raw [N, emb] (the native iresnet.rs graph)."""

    def conv(t, p, stride, pad):
        k = w[f"{p}.weight"].permute(0, 3, 1, 2)
        return F.conv2d(t, k, w[f"{p}.bias"], stride, pad)

    def aff(t, p):
        s, b = w[f"{p}.scale"], w[f"{p}.shift"]
        if t.dim() == 4:
            return t * s.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
        return t * s + b

    def prelu(t, key):
        return F.prelu(t, w[key])

    h = prelu(conv(x, "stem.conv", 1, 1), "stem.prelu.weight")
    for li, nb in enumerate(ARC_LAYERS):
        for b in range(nb):
            p = f"layer{li + 1}.{b}"
            stride = 2 if b == 0 else 1
            t = aff(h, f"{p}.bn1")
            t = conv(t, f"{p}.conv1", 1, 1)
            t = prelu(t, f"{p}.prelu.weight")
            t = conv(t, f"{p}.conv2", stride, 1)
            ident = conv(h, f"{p}.downsample", stride, 0) if b == 0 else h
            h = t + ident
    h = aff(h, "bn2")
    h = h.reshape(h.shape[0], -1)
    h = h @ w["fc.weight"].t() + w["fc.bias"]
    return aff(h, "features")


# ---------------------------------------------------------------------------------------------
# Upstream crop paths (DifferentiableFaceEncoder / DifferentiableLandmarkEncoder .forward).
# ---------------------------------------------------------------------------------------------
def crop_box(bbox, ph, pw):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = bw * 0.15, bh * 0.15
    cx1 = max(0, int(round(float(x1 - pad_w))))
    cy1 = max(0, int(round(float(y1 - pad_h))))
    cx2 = min(pw, int(round(float(x2 + pad_w))))
    cy2 = min(ph, int(round(float(y2 + pad_h))))
    return cx1, cy1, cx2, cy2


def identity_crop(px, box):
    cx1, cy1, cx2, cy2 = box
    crop = px[:, :, cy1:cy2, cx1:cx2]
    _, _, ch, cw = crop.shape
    if cw != ch:
        diff = abs(cw - ch)
        if cw > ch:
            top = diff // 2
            crop = F.pad(crop, (0, 0, top, diff - top), mode="constant", value=0)
        else:
            left = diff // 2
            crop = F.pad(crop, (left, diff - left, 0, 0), mode="constant", value=0)
    return F.interpolate(crop, size=(112, 112), mode="bilinear", align_corners=False)


def embed(w, px, box):
    crop = identity_crop(px, box)
    emb = arcface_forward(w, (crop * 255.0 - 127.5) / 127.5)
    return F.normalize(emb, p=2, dim=-1)


IDENTITY_NOISE_SEED = 0x5EED_24831
IDENTITY_NOISE_SAMPLES = 200
NOISE_PROBES = 4


def noise_mean(w):
    """Upstream's `_identity_mean_embed`: the mean unit ArcFace embedding of 200 noise images at 112^2
    (the full noise image, no crop)."""
    embs = []
    for i in range(IDENTITY_NOISE_SAMPLES):
        x = identity_noise_image(i, 112)
        embs.append(F.normalize(arcface_forward(w, (x * 255.0 - 127.5) / 127.5), p=2, dim=-1))
    return torch.cat(embs).mean(dim=0)


def noise_image(h: int, w: int, key: str) -> torch.Tensor:
    """`clamp(randn*0.3+0.5, 0, 1)` at h x w as NCHW (twin of `synth::noise_image`)."""
    z = gaussian(IDENTITY_NOISE_SEED, key, h * w * 3)
    v = np.clip(z * 0.3 + 0.5, 0.0, 1.0).astype(np.float32).reshape(1, h, w, 3)
    return torch.from_numpy(v).permute(0, 3, 1, 2).contiguous()


def center(e, mean):
    return F.normalize(e - mean, p=2, dim=-1)


# ---------------------------------------------------------------------------------------------
# Synthetic FaceMesh-shaped network (the real checkpoint's op set, tiny widths).
# ---------------------------------------------------------------------------------------------
MESH_SEED = 0x24831B


class TinyMesh(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=0)
        self.p1 = nn.PReLU(8)
        self.dw = nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8)
        self.pw = nn.Conv2d(8, 8, 1)
        self.p2 = nn.PReLU(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(8, 12, 3, stride=2, padding=1)
        self.p3 = nn.PReLU(12)
        self.c4 = nn.Conv2d(12, 12, 3, stride=2, padding=1)
        self.gate = nn.Conv2d(12, 12, 1)
        self.sig = nn.Sigmoid()
        self.c5 = nn.Conv2d(12, 12, 3, stride=2, padding=1, bias=False)
        self.p5 = nn.PReLU(12)
        self.head = nn.Conv2d(12, 1434, 8)
        self.flag = nn.Conv2d(12, 1, 8)

    def forward(self, x):
        h = self.p1(self.c1(F.pad(x, (0, 1, 0, 1))))             # 256 → 128 (tflite SAME pad)
        h = self.p2(self.pw(self.dw(h)) + h)                      # depthwise residual
        a = self.c3(h)                                            # 128 → 64
        b = F.pad(self.pool(h), (0, 0, 0, 0, 0, 4))               # max-pool + channel pad 8 → 12
        h = self.p3(a + b)
        h = self.c4(h)                                            # 64 → 32
        h = h * self.sig(self.gate(h))                            # sigmoid gating
        h = self.p5(self.c5(h))                                   # 32 → 16
        h = F.max_pool2d(h, 2, 2)                                 # 16 → 8
        lm = torch.reshape(self.head(h), (-1, 1, 1, 1434))
        flag = torch.sigmoid(torch.reshape(self.flag(h), (-1, 1, 1, 1)))
        return lm, flag


def mesh_model():
    m = TinyMesh()
    with torch.no_grad():
        for name, p in m.named_parameters():
            key = name if not name.endswith(".weight") or p.dim() > 1 else name[: -len(".weight")] + ".slope"
            p.copy_(synth(MESH_SEED, key, tuple(p.shape)))
    return m.eval()


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 370]
MIDFACE = LEFT_EYE + RIGHT_EYE + NOSE


def normalize_landmarks(lm):
    nose = lm[..., 1:2, :]
    inter = (lm[..., 133, :] - lm[..., 362, :]).norm(dim=-1, keepdim=True).unsqueeze(-2).clamp(min=0.01)
    return (lm - nose) / inter


def landmarks(model, px, box):
    cx1, cy1, cx2, cy2 = box
    crop = F.interpolate(px[:, :, cy1:cy2, cx1:cx2], size=(256, 256), mode="bilinear", align_corners=False)
    out = model(crop)[0]
    return normalize_landmarks(out.reshape(crop.shape[0], 478, 3)[..., :2])


def landmark_loss(gen, ref):
    eps = 1e-6

    def region(idx):
        return (gen[idx] - ref[idx]).pow(2).sum(-1).clamp(min=eps).sqrt().mean()

    return (region(FACE_OVAL) * 3 + region(LIPS) * 2 + region(MIDFACE) * 1) / 6.0


# ---------------------------------------------------------------------------------------------
IMG_SEED = 0x24831C
IMG_H, IMG_W = 72, 88
BBOX = [20.3, 10.7, 61.2, 58.9]
MIN_COS_OPEN = -1.0


def fl(t):
    return [float(v) for v in t.reshape(-1).tolist()]


def main():
    torch.set_grad_enabled(False)
    live = image(IMG_SEED, "live", IMG_H, IMG_W)
    ref = image(IMG_SEED, "reference", IMG_H, IMG_W)
    box = crop_box(BBOX, IMG_H, IMG_W)

    aw = arcface_weights()
    e_live = embed(aw, live, box)
    e_ref = embed(aw, ref, box)
    mean = noise_mean(aw)
    # Upstream's bias-centred cosine (SDTrainer ~2950-2961).
    cos = float((center(e_live, mean) * center(e_ref, mean)).sum())
    # Noise crops against the synthetic reference: 112^2 noise drawn like the bias set (full-frame
    # crop), scored bias-centred, all land below upstream's 0.2 gate.
    probe_box = (0, 0, 112, 112)
    noise_probe_cos = []
    for i in range(NOISE_PROBES):
        e_probe = embed(aw, noise_image(112, 112, f"noise-probe-{i}"), probe_box)
        noise_probe_cos.append(float((center(e_probe, mean) * center(e_ref, mean)).sum()))
    assert max(noise_probe_cos) < 0.2, noise_probe_cos
    # Dataset-average mode (upstream identity_loss_use_average): references {reference, reference2}.
    ref2 = image(IMG_SEED, "reference2", IMG_H, IMG_W)
    e_ref2 = embed(aw, ref2, box)
    avg = F.normalize((e_ref + e_ref2) / 2.0, p=2, dim=-1)
    clean = max(float((center(e_ref, mean) * center(avg, mean)).sum()), 0.1)
    avg_cos = float((center(e_live, mean) * center(avg, mean)).sum())
    avg_loss = max(0.0, 1.0 - avg_cos / clean)

    mesh = mesh_model()
    program, params = fx_program.lower(mesh)
    for k, v in params.items():
        assert torch.equal(v, synth(MESH_SEED, k, tuple(v.shape))), k
    l_live = landmarks(mesh, live, box)[0]
    l_ref = landmarks(mesh, ref, box)[0]
    lm_loss = float(landmark_loss(l_live, l_ref))

    with open(__file__, "rb") as f:
        producer_sha = hashlib.sha256(f.read()).hexdigest()
    out = {
        "producer": "crates/media/face_loss_fixtures/produce_face_loss_fixtures.py",
        "producer_sha256": producer_sha,
        "torch": torch.__version__,
        "image": {"seed": IMG_SEED, "h": IMG_H, "w": IMG_W, "live_key": "live", "reference_key": "reference"},
        "bbox": BBOX,
        "crop_box": list(box),
        "arcface": {
            "seed": ARC_SEED,
            "layers": ARC_LAYERS,
            "keys": {k: list(s) for k, s in arcface_keys().items()},
            "embedding": fl(e_live),
            "reference_embedding": fl(e_ref),
            "noise_seed": IDENTITY_NOISE_SEED,
            "noise_samples": IDENTITY_NOISE_SAMPLES,
            "noise_mean": fl(mean),
            "cos": cos,
            "identity_loss": 1.0 - cos,
            "noise_probe_keys": [f"noise-probe-{i}" for i in range(NOISE_PROBES)],
            "noise_probe_cos": noise_probe_cos,
            "average": {
                "reference2_key": "reference2",
                "clean_cos": clean,
                "cos": avg_cos,
                "identity_loss": avg_loss,
            },
        },
        "facemesh": {
            "seed": MESH_SEED,
            "program": program,
            "param_shapes": {k: list(v.shape) for k, v in params.items()},
            "landmarks": fl(l_live),
            "reference_landmarks": fl(l_ref),
            "landmark_loss": lm_loss,
        },
    }
    path = os.path.join(HERE, "face_loss_fixtures.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {path}: cos={cos:.6f} identity_loss={1 - cos:.6f} landmark_loss={lm_loss:.6f} "
          f"noise_probe_max={max(noise_probe_cos):.4f} avg_loss={avg_loss:.6f} clean={clean:.4f}")


if __name__ == "__main__":
    main()
