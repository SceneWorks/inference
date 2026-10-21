"""Real-weight reference outputs of the restoration recipe's frozen training networks (sc-25688).

Runs the PINNED weights through the reference implementations HYPIR uses (`lpips==0.1.4`
`LPIPS(net="vgg")`, `open_clip_torch==2.31.0` `convnext_xxlarge` read out by HYPIR's own
`backbone._visual_forward`) on CPU in fp32, and writes the small committed reference
`mlx-gen-iris/tests/fixtures/iris_restoration_train_real.safetensors`:

* `lpips/{in0,in1,dist,tap<k>}` — two 48×48 pairs: the distances and the five per-tap terms;
* `clip/{x,feat0,feat1,feat2,feat3}` — one 32×32 image: the stage-0/1/2 maps and the pooled
  embedding HYPIR's critic reads.

The native real-weight tests (`restoration_train_real_weights.rs`, MLX and Candle, `#[ignore]`) load
the same pinned files and compare. Inputs (sha256-checked against
`gen_core::iris::restoration_train`'s resource identities):
  IRIS_LPIPS_VGG16   timm/vgg16.tv_in1k @ b8d8aa2 model.safetensors (= torchvision vgg16-397923af)
  IRIS_LPIPS_LIN     vivym/lpips @ 270571f vgg_lpips_linear.safetensors (= lpips v0.1 vgg.pth)
  IRIS_CRITIC_TRUNK  open_clip_pytorch_model.bin @ laion/CLIP-convnext_xxlarge-…-soup 9f3e8ee
  HYPIR_SRC          XPixelGroup/HYPIR @ b61d107 (its `backbone.py`)

Run: `python -I tools/dump_iris_restoration_train_realweight.py` (isolated venv, HYPIR's pins).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import fixture, require_env  # noqa: E402

PINS = {
    "IRIS_LPIPS_VGG16": "57b026918159a6bf9faf8405c3a551903768e7138989d9c6224a14227203fad8",
    "IRIS_LPIPS_LIN": "0c6387bf2e51e434dedbc0ca4c5894f605c415b8d128bfe720e13456eb4333ac",
    "IRIS_CRITIC_TRUNK": "d571d992c3fc1d9d6f771ad4eb0ae4cb63745d3bd0cf3363ecde5ff249e2f155",
}
HYPIR_COMMIT = "b61d107c6cef38f01a93c7833558869731cfa8c1"
BACKBONE_SHA256 = "b7413e9b616e626dcf624f498488f3d264852b74fd6b88fb2ccfc15f5a9156de"

paths = {}
for env, want in PINS.items():
    p = Path(require_env(env, "the pinned file"))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"{env}: sha256 {h.hexdigest()} != pinned {want}")
    paths[env] = p
root = Path(require_env("HYPIR_SRC", "XPixelGroup/HYPIR @ " + HYPIR_COMMIT))
head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
if head != HYPIR_COMMIT or hashlib.sha256((root / "HYPIR/model/backbone.py").read_bytes()).hexdigest() != BACKBONE_SHA256:
    raise SystemExit("HYPIR_SRC is not the pinned checkout")
sys.path.insert(0, str(root))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
import lpips  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from HYPIR.model import backbone  # noqa: E402

torch.set_grad_enabled(False)
OUT = {}


def wave(shape, a, f, p):
    n = int(np.prod(shape))
    return torch.from_numpy((a * np.sin(f * np.arange(n, dtype=np.float64) + p)).reshape(shape).astype(np.float32))


# ---- LPIPS-VGG ------------------------------------------------------------------------------
net = lpips.LPIPS(net="vgg", version="0.1", pretrained=False, pnet_rand=True, verbose=False).eval()
from safetensors.torch import load_file as _load_st  # noqa: E402
vgg = _load_st(str(paths["IRIS_LPIPS_VGG16"]))
lin = {f"lin{k[0]}.model.1.weight": v for k, v in _load_st(str(paths["IRIS_LPIPS_LIN"])).items()}
sd = net.state_dict()
for k in list(sd):
    if k.startswith("net.slice"):
        _, _, idx, leaf = k.split(".")
        sd[k] = vgg[f"features.{idx}.{leaf}"]
    elif k.startswith("lins."):
        sd[k] = lin["lin" + k[len("lins."):]]
    elif k.startswith("lin"):
        sd[k] = lin[k]
net.load_state_dict(sd)
in0 = wave((2, 3, 48, 48), 0.8, 0.037, 0.3)
in1 = (in0 + wave((2, 3, 48, 48), 0.25, 0.71, 1.1)).clamp(-1, 1)
OUT["lpips/in0"], OUT["lpips/in1"] = in0, in1
OUT["lpips/dist"] = net(in0, in1).flatten()
o0, o1 = net.net(net.scaling_layer(in0)), net.net(net.scaling_layer(in1))
for k in range(5):
    d = (lpips.normalize_tensor(o0[k]) - lpips.normalize_tensor(o1[k])) ** 2
    OUT[f"lpips/tap{k}"] = lpips.spatial_average(net.lins[k](d), keepdim=True).flatten()

# ---- OpenCLIP ConvNeXt-XXL trunk ----------------------------------------------------------------
model, _, _ = open_clip.create_model_and_transforms(
    "convnext_xxlarge", pretrained=str(paths["IRIS_CRITIC_TRUNK"]), precision="fp32"
)
model.eval()
x = wave((1, 3, 32, 32), 0.9, 0.013, 0.2)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
feats = backbone._visual_forward(model, (x * 0.5 + 0.5 - mean) / std, return_pooled_feats=True)
OUT["clip/x"] = x
for i, f in enumerate(feats):
    OUT[f"clip/feat{i}"] = f.contiguous()

path = Path(fixture("mlx-gen-iris/tests/fixtures/iris_restoration_train_real.safetensors"))
save_file({k: v.contiguous() for k, v in OUT.items()}, str(path))
print(f"wrote {path} ({path.stat().st_size} bytes)")
