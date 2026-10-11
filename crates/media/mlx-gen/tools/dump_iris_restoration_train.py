"""Dump the Iris-3B restoration-TRAINING goldens (sc-25688) for the native MLX / Candle trainers.

The Iris release has no restoration training loop; the paper (arXiv 2610.09450v1 §5.3) follows
HYPIR. This oracle runs HYPIR's OWN modules (`XPixelGroup/HYPIR` @ HYPIR_COMMIT, sha256-pinned
below) and the reference packages they import (`lpips==0.1.4`, `vision-aided-loss==0.1.0`,
`open_clip_torch==2.31.0`, `timm==1.0.15`, `torch==2.6.0` — HYPIR's `requirements.txt`) on CPU in
fp32. Nothing at runtime depends on Python.

Writes `mlx-gen-iris/tests/fixtures/iris_restoration_train_golden.safetensors`:

* `kernels/<i>` — `bivariate_Gaussian` / `bivariate_generalized_Gaussian` / `bivariate_plateau` (iso
  and aniso) and `circular_lowpass_kernel` on explicit parameters, zero-padded to 21 and cast to
  f32 exactly as `realesrgan.py` does; `kernels/usm` — `USMSharp().kernel`. Parameters in the
  `kernels` metadata (JSON).
* `chain/<case>/...` — `RealESRGANBatchTransform.__call__` (the second-order pipeline, USM
  sharpening included) run on one image with every random choice SCRIPTED (patched `random`,
  `np.random.uniform`, `torch.rand`, `Tensor.uniform_`) and every per-pixel draw RECORDED
  (`torch.randn` → `normals`, `torch.poisson` → `poisson`). Every pipeline op is recorded with its
  input and output (`op<k>/{in,out}`, kinds in the `chains` metadata), plus `hq`, `gt`, `lq`.
* `lpips/...` — `lpips.LPIPS(net="vgg", version="0.1")` with every weight replaced by the shared
  deterministic formula (`mlx_gen::train::formula`, torchvision key names): inputs, per-sample
  distances, and `d mean(LPIPS) / d in0`.
* `critic/...` — HYPIR's `ImageConvNextDiscriminator` forward / loss path (its `forward`, the
  `MultiLevelDConv` head with spectral norm + BlurPool, `multilevel_loss(alpha=0.8)`) on a
  miniature timm `ConvNeXt` trunk wrapped exactly as `backbone._visual_forward` expects: trunk +
  head state (`critic/w/*`), the trunk features, the generator term and its input gradient (eval
  mode, no power iteration), the critic step (train mode: two power iterations, `u`/`v` after
  each, `loss_D`, every head gradient), and one `clip_grad_norm_(1.0)` + `torch.optim.AdamW` step of
  the head.
* `gloss/...` — the generator loss of `BaseTrainer.optimize_generator` (`F.mse_loss` + 5·LPIPS
  + 0.5·D(x, for_G)) on the formula LPIPS and the miniature critic: each term, the total, and the
  input gradient of each term and of the total.

Run: `HYPIR_SRC=/path/to/HYPIR python -I tools/dump_iris_restoration_train.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import fixture, require_env  # noqa: E402

HYPIR_COMMIT = "b61d107c6cef38f01a93c7833558869731cfa8c1"
HYPIR_SHA256 = {
    "HYPIR/utils/degradation.py": "fbae8e401c4727029880c060785114e7d23e221154f7ceb8ea0c76c943418a9f",
    "HYPIR/dataset/batch_transform.py": "d5a1cc0e50b8f480b56ccb7c92fdd1283c116ccf20ab1806ea254749a1c2b58f",
    "HYPIR/dataset/utils.py": "7ab172117e5ffa7483c87aa679b1a5cc0d06c046eeed9011b58e212d7746670d",
    "HYPIR/dataset/diffjpeg.py": "e69c8313c8f097d4bc47c22fe907c9fd749ecbc165306bc15b82d1cacdaf4861",
    "HYPIR/dataset/realesrgan.py": "aa01d75baaa1ba2137da0db165e01bc8220c538cb4922a3c10c18b52a4b38af4",
    "HYPIR/model/D.py": "5b8d3fb3cd537d3e24cf7d67db5424fda5dd5d095420c3303110359d14926434",
    "HYPIR/model/backbone.py": "b7413e9b616e626dcf624f498488f3d264852b74fd6b88fb2ccfc15f5a9156de",
    "HYPIR/trainer/base.py": "abad97b7b8f52c973af7c02c43ca94769b1113e5fbf531ed2c527266f48a6a1a",
    "configs/sd2_train.yaml": "203175fdf1d340051a2ecfd4a2e25bdf7703d2eaea0d6f7a9fa5250944ae7823",
}
PACKAGES = {
    "torch": "2.6.0",
    "lpips": "0.1.4",
    "open_clip": "2.31.0",
    "timm": "1.0.15",
}

ROOT = Path(require_env("HYPIR_SRC", "a checkout of XPixelGroup/HYPIR at " + HYPIR_COMMIT))
head = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip()
if head != HYPIR_COMMIT:
    raise SystemExit(f"HYPIR_SRC HEAD is {head}, expected {HYPIR_COMMIT}")
for rel, want in HYPIR_SHA256.items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lpips  # noqa: E402
import open_clip  # noqa: E402
import timm  # noqa: E402

from importlib.metadata import version as _dist_version  # noqa: E402

PACKAGES["vision-aided-loss"] = "0.1.0"
for mod, want in PACKAGES.items():
    dist = {"open_clip": "open_clip_torch"}.get(mod, mod)
    have = _dist_version(dist)
    if not str(have).startswith(want):
        raise SystemExit(f"{mod} {have} != pinned {want}")

from HYPIR.dataset import batch_transform as bt  # noqa: E402
from HYPIR.dataset.utils import USMSharp  # noqa: E402
from HYPIR.model import backbone  # noqa: E402
from HYPIR.model.D import ImageConvNextDiscriminator, MultiLevelDConv  # noqa: E402
from HYPIR.utils import degradation as deg  # noqa: E402
from vision_aided_loss.cv_losses import multilevel_loss  # noqa: E402

torch.manual_seed(0)
OUT: dict[str, torch.Tensor] = {}
META: dict[str, str] = {"hypir": HYPIR_COMMIT}


def put(key: str, t) -> None:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    OUT[key] = t.detach().to(torch.float32).contiguous().clone()


def wave(shape, a, f, p, lo=None, hi=None) -> torch.Tensor:
    n = int(np.prod(shape))
    v = a * np.sin(f * np.arange(n, dtype=np.float64) + p)
    if lo is not None:
        v = lo + (hi - lo) * (v - v.min()) / (v.max() - v.min())
    return torch.from_numpy(v.reshape(shape).astype(np.float32))


# =============================================================================================
# Kernels
# =============================================================================================
kernel_cases = [
    ("iso", 7, 1.3, 1.3, 0.0, 1.0),
    ("aniso", 11, 2.1, 0.6, 0.7, 1.0),
    ("generalized_iso", 13, 1.7, 1.7, 0.0, 0.62),
    ("generalized_aniso", 9, 0.9, 2.4, -1.9, 3.1),
    ("plateau_iso", 15, 2.2, 2.2, 0.0, 1.4),
    ("plateau_aniso", 21, 2.8, 1.1, 2.6, 1.8),
    ("sinc", 13, 1.234, 0.0, 0.0, 0.0),
    ("sinc", 21, 0.71, 0.0, 0.0, 0.0),
]
for i, (kind, size, sx, sy, th, beta) in enumerate(kernel_cases):
    iso = kind.endswith("iso") and not kind.endswith("aniso")
    if kind == "sinc":
        k = deg.circular_lowpass_kernel(sx, size, pad_to=False)
    elif kind in ("iso", "aniso"):
        k = deg.bivariate_Gaussian(size, sx, sy, th, isotropic=iso)
    elif kind.startswith("generalized"):
        k = deg.bivariate_generalized_Gaussian(size, sx, sy, th, beta, isotropic=iso)
    else:
        k = deg.bivariate_plateau(size, sx, sy, th, beta, isotropic=iso)
    pad = (21 - size) // 2
    k = np.pad(k, ((pad, pad), (pad, pad)))
    put(f"kernels/{i}", torch.FloatTensor(k))
META["kernels"] = json.dumps(
    [dict(kind=k, size=s, sx=a, sy=b, theta=t, beta=be) for k, s, a, b, t, be in kernel_cases]
)
put("kernels/usm", USMSharp().kernel[0])


# =============================================================================================
# The second-order chain, with scripted choices and recorded draws
# =============================================================================================
class Script:
    def __init__(self, values):
        self.q = {k: list(v) for k, v in values.items()}

    def take(self, name):
        q = self.q[name]
        if not q:
            raise SystemExit(f"script for {name} exhausted")
        return q.pop(0)

    def drained(self):
        left = {k: v for k, v in self.q.items() if v}
        if left:
            raise SystemExit(f"unused scripted values: {left}")


def run_chain(name, hq, k1, k2, sinc, script_values):
    script = Script(script_values)
    ops = []
    normals, poissons = [], []
    real_randn, real_poisson, real_rand = torch.randn, torch.poisson, torch.rand
    real_uniform_ = torch.Tensor.uniform_

    def rec(kind, fn):
        def wrapped(x, *a, **kw):
            # snapshot tensor kwargs first: DiffJPEG rewrites `quality` into its factor in place
            snap = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
            out = fn(x, *a, **kw)
            ops.append((kind, x.detach().clone(), out.detach().clone(), a, snap))
            return out
        return wrapped

    tr = bt.RealESRGANBatchTransform(
        hq_key="hq",
        extra_keys=[],
        use_sharpener=True,
        queue_size=0,
        resize_prob=[0.2, 0.7, 0.1],
        resize_range=[0.15, 1.5],
        gaussian_noise_prob=0.5,
        noise_range=[1, 30],
        poisson_scale_range=[0.05, 3],
        gray_noise_prob=0.4,
        jpeg_range=[30, 95],
        second_blur_prob=0.8,
        stage2_scale=4,
        resize_prob2=[0.3, 0.4, 0.3],
        resize_range2=[0.3, 1.2],
        gaussian_noise_prob2=0.5,
        noise_range2=[1, 25],
        poisson_scale_range2=[0.05, 2.5],
        gray_noise_prob2=0.4,
        jpeg_range2=[30, 95],
        resize_back=True,
    )
    patches = []

    def patch(obj, attr, val):
        patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    patch(bt.random, "choices", lambda pop, w: [script.take("choices")])
    patch(bt.random, "choice", lambda pop: script.take("choice"))
    patch(bt.np.random, "uniform", lambda *a, **k: script.take("uniform"))
    patch(torch, "rand", lambda *a, **k: torch.tensor([script.take("rand")], dtype=torch.float32))
    patch(torch.Tensor, "uniform_", lambda self, *a, **k: self.fill_(script.take("tensor_uniform")))

    def randn(*a, **k):
        t = real_randn(*a, **k)
        normals.append(t.flatten().clone())
        return t

    def poisson(x, *a, **k):
        t = real_poisson(x, *a, **k)
        poissons.append(t.flatten().clone())
        return t

    patch(torch, "randn", randn)
    patch(torch, "poisson", poisson)
    patch(bt, "filter2D", rec("filter2d", bt.filter2D))
    patch(bt, "random_add_gaussian_noise_pt", rec("gaussian", bt.random_add_gaussian_noise_pt))
    patch(bt, "random_add_poisson_noise_pt", rec("poisson", bt.random_add_poisson_noise_pt))
    patch(bt, "F", SimpleNamespace(interpolate=rec("interpolate", F.interpolate)))
    try:
        tr.jpeger.forward = rec("jpeg", tr.jpeger.forward)
        sharp = tr.usm_sharpener.forward
        tr.usm_sharpener.forward = rec("usm", sharp)
        batch = {"hq": hq[None], "kernel1": k1[None], "kernel2": k2[None], "sinc_kernel": sinc[None]}
        res = tr(batch)
    finally:
        for obj, attr, val in reversed(patches):
            setattr(obj, attr, val)
    script.drained()
    pre = f"chain/{name}"
    put(f"{pre}/hq", hq)
    put(f"{pre}/k1", k1)
    put(f"{pre}/k2", k2)
    put(f"{pre}/sinc", sinc)
    put(f"{pre}/gt", res["GT"][0])
    put(f"{pre}/lq", res["LQ"][0])
    put(f"{pre}/normals", torch.cat(normals) if normals else torch.zeros(0))
    put(f"{pre}/poisson", torch.cat(poissons) if poissons else torch.zeros(0))
    kinds = []
    for i, (kind, x, y, a, kw) in enumerate(ops):
        put(f"{pre}/op{i}/in", x[0])
        put(f"{pre}/op{i}/out", y[0])
        info = {"kind": kind}
        if kind == "interpolate":
            info["mode"] = kw.get("mode")
            if kw.get("scale_factor") is not None:
                info["scale_factor"] = float(kw["scale_factor"])
            if kw.get("size") is not None:
                info["size"] = [int(v) for v in kw["size"]]
        if kind == "jpeg":
            info["quality"] = float(kw["quality"][0])
        if kind == "filter2d":
            k = a[0]
            info["kernel"] = "usm" if k.shape[-1] == 51 else None
            put(f"{pre}/op{i}/kernel", k[0] if k.dim() == 3 else k)
        kinds.append(info)
    return kinds


def stage_values(updown, scale_u, mode, gaussian, rand_level, rand_gray, jpeg_q):
    """The scripted values of one stage, in `__call__`'s call order."""
    v = {"choices": [updown], "choice": [mode], "uniform": [], "rand": [rand_level, rand_gray]}
    if updown != "keep":
        v["uniform"].append(scale_u)
    v["uniform"].append(0.1 if gaussian else 0.9)  # `< gaussian_noise_prob`
    v["tensor_uniform"] = [jpeg_q]
    return v


def stage_draw(updown, scale_u, mode, gaussian, rand_level, rand_gray, jpeg_q, stage):
    """What one scripted stage means — the native `StageDraw` (levels computed as the reference
    computes them: `torch.rand(b) * (hi - lo) + lo` in f32)."""
    noise_range, poisson_range, gray_prob = {
        1: ((1, 30), (0.05, 3), 0.4),
        2: ((1, 25), (0.05, 2.5), 0.4),
    }[stage]
    lo, hi = noise_range if gaussian else poisson_range
    level = (torch.tensor([rand_level], dtype=torch.float32) * (hi - lo) + lo)[0].item()
    return {
        "scale": 1.0 if updown == "keep" else scale_u,
        "mode": mode,
        "noise": "gaussian" if gaussian else "poisson",
        "level": level,
        "gray": bool(rand_gray < gray_prob),
        "jpeg": jpeg_q,
    }


def merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


hq = wave((3, 48, 64), 1.0, 0.0173, 0.4, 0.05, 0.95)
hq = hq + 0.15 * wave((3, 48, 64), 1.0, 0.61, 1.3)
hq = hq.clamp(0, 1)
K = {i: OUT[f"kernels/{i}"] for i in range(8)}
pulse = torch.zeros(21, 21)
pulse[10, 10] = 1
chains = {}
# A: up/bilinear + gray Gaussian; second blur; down/area + colour Poisson; [resize+sinc] then JPEG
DRAWS = {
    "a": {"s1": stage_draw("up", 1.31, "bilinear", True, 0.4, 0.2, 45.5, 1), "second_blur": True,
          "s2": stage_draw("down", 0.7, "area", False, 0.6, 0.9, 71.25, 2), "sinc_first": True,
          "back": "bicubic"},
    "b": {"s1": stage_draw("down", 0.61, "area", False, 0.3, 0.1, 88.0, 1), "second_blur": False,
          "s2": stage_draw("up", 1.15, "bicubic", True, 0.5, 0.9, 33.0, 2), "sinc_first": False,
          "back": "bilinear"},
    "c": {"s1": stage_draw("keep", None, "bicubic", True, 0.7, 0.8, 60.0, 1), "second_blur": True,
          "s2": stage_draw("keep", None, "bilinear", False, 0.2, 0.05, 52.0, 2), "sinc_first": True,
          "back": "area"},
}
META["draws"] = json.dumps(DRAWS)
chains["a"] = run_chain(
    "a", hq, K[1], K[3], K[7],
    merge(
        stage_values("up", 1.31, "bilinear", True, 0.4, 0.2, 45.5),
        {"uniform": [0.3]},  # second blur (< 0.8)
        stage_values("down", 0.7, "area", False, 0.6, 0.9, 71.25),
        {"uniform": [0.2], "choice": ["bicubic"]},  # order (< 0.5): resize back + sinc first
    ),
)
# B: down/area + gray Poisson; no second blur; up/bicubic + colour Gaussian; JPEG then resize+sinc
chains["b"] = run_chain(
    "b", hq, K[4], K[6], pulse,
    merge(
        stage_values("down", 0.61, "area", False, 0.3, 0.1, 88.0),
        {"uniform": [0.95]},  # no second blur
        stage_values("up", 1.15, "bicubic", True, 0.5, 0.9, 33.0),
        {"uniform": [0.8], "choice": ["bilinear"]},
    ),
)
# C: keep/bicubic + colour Gaussian; second blur; keep/bilinear + gray Poisson; resize+sinc first
chains["c"] = run_chain(
    "c", hq, K[0], K[2], K[6],
    merge(
        stage_values("keep", None, "bicubic", True, 0.7, 0.8, 60.0),
        {"uniform": [0.1]},
        stage_values("keep", None, "bilinear", False, 0.2, 0.05, 52.0),
        {"uniform": [0.4], "choice": ["area"]},
    ),
)
META["chains"] = json.dumps(chains)


# =============================================================================================
# LPIPS on formula weights
# =============================================================================================
def fnv1a32(key: str) -> int:
    h = 2166136261
    for b in key.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def formula(key: str, shape, role: str) -> torch.Tensor:
    n = int(np.prod(shape))
    phase = (fnv1a32(key) % 10007) / 1000.0
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    w = np.sin(0.7 * np.arange(n, dtype=np.float64) + phase)
    if role == "conv":
        v = math.sqrt(2.0 / max(fan_in, 1)) * w
    elif role == "bias":
        v = 0.1 * w
    elif role == "lin":
        v = 0.05 * np.abs(w) + 0.01
    else:
        raise ValueError(role)
    return torch.from_numpy(v.astype(np.float32).reshape(shape))


net = lpips.LPIPS(net="vgg", version="0.1", pretrained=False, pnet_rand=True, verbose=False).eval()
sd = net.state_dict()
for k in list(sd):
    if k.startswith("net.slice"):
        idx = k.split(".")[2]
        leaf = k.split(".")[3]
        tv = f"features.{idx}.{leaf}"
        sd[k] = formula(tv, tuple(sd[k].shape), "conv" if leaf == "weight" else "bias")
    elif k.startswith("lin"):
        # `lin{k}` and `lins.{k}` are the same modules registered twice; both get lin{k}'s values.
        idx = k[len("lins."):].split(".")[0] if k.startswith("lins.") else k[3]
        sd[k] = formula(f"lin{idx}.model.1.weight", tuple(sd[k].shape), "lin")
net.load_state_dict(sd)
in0 = wave((2, 3, 40, 40), 0.8, 0.05, 0.3).requires_grad_(True)
in1 = wave((2, 3, 40, 40), 0.7, 0.031, 1.1)
d = net(in0, in1)
d.mean().backward()
put("lpips/in0", in0)
put("lpips/in1", in1)
put("lpips/dist", d.flatten())
put("lpips/grad_in0", in0.grad)


# =============================================================================================
# The miniature critic
# =============================================================================================
torch.manual_seed(7)
DIMS, DEPTHS, EMBED, WIDTH = (8, 16, 24, 32), (1, 1, 2, 1), 16, 12
trunk = timm.models.convnext.ConvNeXt(
    depths=DEPTHS, dims=DIMS, norm_eps=1e-5, num_classes=0, drop_path_rate=0.0
).eval()
head = nn.Sequential(OrderedDict(drop=nn.Dropout(0.0), proj=nn.Linear(DIMS[-1], EMBED, bias=False)))
with torch.no_grad():
    for name, p in list(trunk.named_parameters()) + [("proj", head.proj.weight)]:
        if name.endswith("gamma"):
            p.copy_(0.5 + 0.4 * torch.rand_like(p))
        elif p.dim() == 1 and ("norm" in name or "stem.1" in name or "downsample.0" in name) and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.dim() == 1:
            p.copy_(0.1 * torch.randn_like(p))
        else:
            fan = p[0].numel()
            p.copy_(torch.randn_like(p) / math.sqrt(fan))
visual = nn.Module()
visual.trunk = trunk
visual.head = head
clip_like = SimpleNamespace(visual=visual)


class MiniTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = visual

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return backbone._visual_forward(clip_like, image, return_feats, return_pooled_feats)


class MiniD(ImageConvNextDiscriminator):
    """HYPIR's discriminator with the miniature trunk in place of `ImageOpenCLIPConvNext`; every
    method (forward / train / eval / requires_grad_) is HYPIR's."""

    def __init__(self):
        nn.Module.__init__(self)
        self.model = MiniTrunk()
        self.model.eval().requires_grad_(False)
        self.decoder = MultiLevelDConv(level=4, in_ch1=list(DIMS[:3]), in_ch2=EMBED, out_ch=WIDTH, down=2)
        self.loss_fn = multilevel_loss(alpha=0.8)
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32))


D = MiniD()
for k, v in trunk.state_dict().items():
    put(f"critic/w/visual.trunk.{k}", v)
put("critic/w/visual.head.proj.weight", head.proj.weight)
SN = [f"decoder.{i}.1" for i in range(3)] + [f"decoder.{i}.4" for i in range(3)] + ["decoder.3.0", "out"]


def sn_state(prefix):
    mod = D.decoder
    if prefix == "out":
        m = mod.out
    else:
        _, i, j = prefix.split(".")
        m = mod.decoder[int(i)][int(j)]
    return m


for p in SN:
    m = sn_state(p)
    put(f"critic/head0/{p}.weight_orig", m.weight_orig)
    put(f"critic/head0/{p}.bias", m.bias)
    put(f"critic/head0/{p}.weight_u", m.weight_u)
    put(f"critic/head0/{p}.weight_v", m.weight_v)

x = wave((2, 3, 64, 64), 0.9, 0.013, 0.2).requires_grad_(True)
gt = wave((2, 3, 64, 64), 0.85, 0.0071, 2.0)
put("critic/x", x)
put("critic/gt", gt)
with torch.no_grad():
    xn = (x * 0.5 + 0.5 - D.image_mean[:, None, None]) / D.image_std[:, None, None]
    feats = D.model.encode_image(xn, return_pooled_feats=True)
for i, f in enumerate(feats):
    put(f"critic/feat{i}", f)

# generator term (D.eval(), no power iteration) and its input gradient
D.eval().requires_grad_(False)
g_loss, g_logits = D(x, for_G=True, return_logits=True)
put("critic/g_loss", g_loss.flatten())
for i, l in enumerate(g_logits):
    put(f"critic/g_logits{i}", l)
g_loss.mean().backward()
put("critic/g_grad_x", x.grad)
x.grad = None

# critic step (D.train(): each forward runs one power iteration)
D.train().requires_grad_(True)
loss_real, real_logits = D(gt, for_real=True, return_logits=True)
for p in SN:
    put(f"critic/after_real/{p}.weight_u", sn_state(p).weight_u)
    put(f"critic/after_real/{p}.weight_v", sn_state(p).weight_v)
loss_fake, fake_logits = D(x.detach(), for_real=False, return_logits=True)
for p in SN:
    put(f"critic/after_fake/{p}.weight_u", sn_state(p).weight_u)
    put(f"critic/after_fake/{p}.weight_v", sn_state(p).weight_v)
loss_d = loss_real.mean() + loss_fake.mean()
put("critic/loss_real", loss_real.flatten())
put("critic/loss_fake", loss_fake.flatten())
put("critic/loss_d", loss_d.reshape(1))
loss_d.backward()
params = []
for p in SN:
    m = sn_state(p)
    put(f"critic/grad/{p}.weight_orig", m.weight_orig.grad)
    put(f"critic/grad/{p}.bias", m.bias.grad)
    params += [m.weight_orig, m.bias]
norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
put("critic/grad_norm", norm.reshape(1))
opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
opt.step()
for p in SN:
    m = sn_state(p)
    put(f"critic/stepped/{p}.weight_orig", m.weight_orig)
    put(f"critic/stepped/{p}.bias", m.bias)

# =============================================================================================
# The generator loss (`BaseTrainer.optimize_generator`, lambda_l2 = 1, lambda_lpips = 5,
# lambda_gan = 0.5) — D back in eval mode with its post-step state.
# =============================================================================================
D.eval().requires_grad_(False)
xg = wave((2, 3, 64, 64), 0.75, 0.021, 0.9).requires_grad_(True)
put("gloss/x", xg)
terms = {
    "l2": lambda v: F.mse_loss(v, gt, reduction="mean") * 1.0,
    "lpips": lambda v: net(v, gt).mean() * 5.0,
    "gan": lambda v: D(v, for_G=True).mean() * 0.5,
}
total = 0
for name, fn in terms.items():
    xi = xg.detach().clone().requires_grad_(True)
    val = fn(xi)
    val.backward()
    put(f"gloss/{name}", val.reshape(1))
    put(f"gloss/grad_{name}", xi.grad)
loss_g = sum(fn(xg) for fn in terms.values())
loss_g.backward()
put("gloss/total", loss_g.reshape(1))
put("gloss/grad_total", xg.grad)
for p in SN:
    m = sn_state(p)
    put(f"gloss/head/{p}.weight_orig", m.weight_orig)
    put(f"gloss/head/{p}.bias", m.bias)
    put(f"gloss/head/{p}.weight_u", m.weight_u)
    put(f"gloss/head/{p}.weight_v", m.weight_v)

from safetensors.torch import save_file  # noqa: E402

path = Path(fixture("mlx-gen-iris/tests/fixtures/iris_restoration_train_golden.safetensors"))
save_file(OUT, str(path), metadata=META)
print(f"wrote {path} ({len(OUT)} tensors, {path.stat().st_size} bytes)")
