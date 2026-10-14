"""Dump the Iris-3B **training** goldens for the native trainers (sc-25685).

The frozen upstream training code runs on CPU in fp32 over the miniature backbone of
`dump_iris_golden.py` (`tests/fixtures/tiny-snapshot/iris`): `RectifiedFlow.training_loss`, the
accelerate accumulation contract (`loss / grad_accum` before backward), `clip_grad_norm_`, the
upstream `EMA` (updated from the pre-step weights, as `Trainer.run` does), `build_muon_param_groups`
routing and Dion's `Muon` (pinned `microsoft/dion@58d38adb`, the commit `pyproject.toml` pins) or
`torch.optim.AdamW`, `LambdaLR` via `train/lr.py`.

Upstream trains no adapters, so the LoRA / LoKr oracle is the standard parameterization applied to
the frozen upstream Linear: `W + (alpha/rank)·B·A` (PEFT LoRA) and `W + (alpha/rank)·kron(w1, w2)`
(LyCORIS LoKr, the repo's `reconstruct_lokr_delta`), trained with the same objective/clip/EMA and
AdamW or Muon (a 2-D factor of a Muon-routed hidden matrix is its own Muon matrix, no row split;
everything else AdamW) — the native rule in `gen_core::iris::train::adapter_param_route`.

Dion's `use_triton=True, use_polar_express=False` selects `newton_schulz_triton`, whose math is
`zeropower_via_newtonschulz5` (same bf16 quintic, same coefficients); triton is CUDA-only, so the
oracle runs the latter, eagerly (`torch._dynamo.config.disable`).

Writes `mlx-gen-iris/tests/fixtures/iris_train_golden.safetensors`.

Run (isolated venv: torch 2.8 CPU, safetensors, omegaconf):
    IRIS_SRC=/path/to/iris-3b DION_SRC=/path/to/dion python -I tools/dump_iris_train.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

# `python -I` drops the script directory from sys.path; the shared helpers live beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _iris_common import FIXTURE_DIR, import_upstream, save_safetensors
from _paths import require_env

import_upstream()

DION_COMMIT = "58d38adb5fcd403a116cb16157a5ca824f8538a5"
UPSTREAM_TRAIN_SHA256 = {
    "src/iris3b/train/optim.py": "ad856403db849e3c6d0cb39f81377c3d38d3a4e18e5340dc956105fbb2225377",
    "src/iris3b/train/ema.py": "0f9b4ea1df8311e1425d64f4acc0fb630da16e6c57d639a407cee905780cca54",
    "src/iris3b/flow/transport.py": "af62589e16956ea33d8affe78849c75f716f632a6029acbab75439d6c4786679",
    "src/iris3b/flow/timesteps.py": "aceeb75c0d7141c2b2ef343225ac0bc373ddb1fc23c384d4af19f7741083caef",
    "src/iris3b/train/lr.py": "e5a315b8fbe97b8f4cd78b9aea29c3e7bf86231cc217e4304a1c1002266d9bd9",
    "src/iris3b/seeding.py": "793dadc6303250ec2271f39e9307fea62ea7c9949c315764d216bbf26329138a",
    "src/iris3b/train/trainer.py": "bd1cbe40adb0cca47a5198b38380c98ad6d7890fcd0fc5cdb9d566bab973f8d8",
}
DION_SHA256 = {
    "dion/muon.py": "ebec9abc47d66dda148ffa94146f9b861dc413a3722866a39c4953716893e220",
    "dion/megabatch_base.py": "2991fa30d7ae80850f08f2a2e1b564d993c461d6f0a3a2c19013aee529afb4f4",
    "dion/scalar_opts.py": "ed1e43b486a293d8f8163fb99d9ba65c79321a294dff28b1aacce4270d21508d",
}


def _verify(root: str, pins: dict[str, str], commit: str | None) -> None:
    if commit is not None:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if head != commit:
            raise SystemExit(f"{root} HEAD is {head}, expected {commit}")
    for rel, want in pins.items():
        got = hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
        if got != want:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")


_verify(require_env("IRIS_SRC", "the iris-3b checkout"), UPSTREAM_TRAIN_SHA256, None)
DION_SRC = require_env("DION_SRC", f"a checkout of microsoft/dion at {DION_COMMIT}")
_verify(DION_SRC, DION_SHA256, DION_COMMIT)

import torch  # noqa: E402

torch._dynamo.config.disable = True  # Dion's @torch.compile kernels run eagerly (same math)
sys.path.insert(0, DION_SRC)

from dion.muon import Muon  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from iris3b.config import FlowConfig, ModelConfig, OptimizerConfig, PixelStageConfig  # noqa: E402
from iris3b.flow.transport import RectifiedFlow  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.train.ema import EMA  # noqa: E402
from iris3b.train.lr import build_lr_scheduler  # noqa: E402
from iris3b.train.optim import build_muon_param_groups  # noqa: E402

torch.manual_seed(1234)
BACKBONE = FIXTURE_DIR / "tiny-snapshot" / "iris"
OUT = FIXTURE_DIR / "iris_train_golden.safetensors"

raw = OmegaConf.to_container(OmegaConf.load(BACKBONE / "config.yaml"))
model_cfg = dict(raw["model"])
model_cfg["pixel"] = PixelStageConfig(**model_cfg["pixel"])
MODEL_CFG = ModelConfig(**model_cfg)
BASE = load_file(str(BACKBONE / "model.safetensors"))

# ---- the step inputs (two micro-batches) ------------------------------------------------------
B, C, S = 2, 3, 16
T, L, DT = MODEL_CFG.text_len, MODEL_CFG.text_lap_num_layers, MODEL_CFG.text_dim
SHIFT = 4.0
LR, WD, CLIP, EMA_DECAY, ACCUM = 1e-3, 0.01, 0.5, 0.9, 2
STEPS = 2


def micro_batch(i: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(100 + i)
    mask = torch.zeros(B, T, dtype=torch.int64)
    mask[0, :7] = 1
    mask[1, :4] = 1
    y = torch.randn(B, T, L, DT, generator=g) * mask[:, :, None, None]
    return {
        "x0": torch.rand(B, C, S, S, generator=g) * 2 - 1,
        "noise": torch.randn(B, C, S, S, generator=g),
        "t_idx": torch.tensor([123 + 400 * i, 871 - 300 * i]),
        "y": y,
        "y_mask": mask,
    }


MICRO = [micro_batch(0), micro_batch(1)]
FLOW = RectifiedFlow(FlowConfig(shift=SHIFT, prediction="v"))
FLOW_X = RectifiedFlow(FlowConfig(shift=SHIFT, prediction="x", x_pred_sigma_min=0.05))


def fresh_model() -> IrisDiT:
    m = IrisDiT(MODEL_CFG)
    m.load_state_dict(BASE, strict=True)
    return m.float()


def flow_loss(model, mb, flow=FLOW):
    return flow.training_loss(
        model, mb["x0"], mb["y"], timestep_idx=mb["t_idx"], noise=mb["noise"],
        model_kwargs={"y_mask": mb["y_mask"]},
    ).loss


def optimizer(kind: str, named: list[tuple[str, torch.nn.Parameter]], core: IrisDiT | None, adapter_groups=None):
    params = [p for _, p in named]
    cfg = OptimizerConfig(name=kind, lr=LR, weight_decay=WD, warmup_steps=0, schedule="constant")
    if kind == "adamw":
        opt = torch.optim.AdamW(params, lr=LR, betas=tuple(cfg.betas), eps=1e-8, weight_decay=WD)
    else:
        groups = adapter_groups if adapter_groups is not None else build_muon_param_groups(
            core, params, [n for n, _ in named]
        )
        opt = Muon(
            groups, distributed_mesh=None, lr=LR, mu=cfg.muon_momentum, betas=tuple(cfg.betas)[:2],
            weight_decay=WD, epsilon=1.0e-8, nesterov=cfg.muon_nesterov, adjust_lr=cfg.muon_adjust_lr,
            use_triton=False, use_polar_express=False,
        )
    sched = build_lr_scheduler(cfg, opt, world_size=1, total_steps=STEPS)
    return opt, sched


def run(loss_of, named, opt, sched, ema_update):
    """`Trainer.run`'s accumulate → clip → EMA → step → scheduler sequence, STEPS times."""
    losses, norms = [], []
    params = [p for _, p in named]
    for _ in range(STEPS):
        for mb in MICRO:
            loss = loss_of(mb)
            (loss / ACCUM).backward()
            losses.append(loss.item())
        norms.append(float(torch.nn.utils.clip_grad_norm_(params, CLIP)))
        ema_update()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    return torch.tensor(losses), torch.tensor(norms)


out: dict[str, torch.Tensor] = {}
for i, mb in enumerate(MICRO):
    for k, v in mb.items():
        out[f"mb{i}.{k}"] = v.float() if k != "t_idx" else v.to(torch.int32)

# ---- losses ------------------------------------------------------------------------------------
with torch.no_grad():
    m = fresh_model()
    out["loss_v"] = flow_loss(m, MICRO[0]).reshape(1)
    out["loss_x"] = flow_loss(m, MICRO[0], FLOW_X).reshape(1)

# ---- full training -----------------------------------------------------------------------------
SUBSET = [
    "blocks.0.attn.q_proj_x.weight",
    "blocks.2.mlp.w1.weight",
    "modulation_cores.adaln_img.weight",
    "y_embedder.layer_blocks.0.attn.qkv.weight",
    "pixel_blocks.0.attn.qkv.weight",
    "s_embedder.proj.weight",
    "blocks.1.adaln_txt.bias",
    "y_pos_embedding",
    "final_layer.linear.weight",
    "blocks.0.norm_x1.weight",
]
for kind in ("adamw", "muon"):
    model = fresh_model()
    named = list(model.named_parameters())
    init = {n: p.detach().clone() for n, p in named}
    ema = EMA(model, EMA_DECAY)
    opt, sched = optimizer(kind, named, model)
    losses, norms = run(lambda mb: flow_loss(model, mb), named, opt, sched, lambda: ema.update(model))
    out[f"full_{kind}.losses"] = losses
    out[f"full_{kind}.grad_norms"] = norms
    ema_sd = ema.state_dict()
    names = [n for n, _ in named]
    out[f"full_{kind}.delta_norms"] = torch.stack(
        [(dict(named)[n].detach() - init[n]).norm() for n in names]
    )
    out[f"full_{kind}.ema_delta_norms"] = torch.stack([(ema_sd[n] - init[n]).norm() for n in names])
    for n in SUBSET:
        out[f"full_{kind}.delta.{n}"] = dict(named)[n].detach() - init[n]
        out[f"full_{kind}.ema.{n}"] = ema_sd[n] - init[n]
FULL_ORDER = [n for n, _ in fresh_model().named_parameters()]


# ---- adapters ----------------------------------------------------------------------------------
def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """LyCORIS `factorization` (the repo's `mlx_gen::train::lora::factorization`)."""
    if factor > 0 and dimension % factor == 0:
        m, n = factor, dimension // factor
        return (n, m) if m > n else (m, n)
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
        length = m + n
    return (n, m) if m > n else (m, n)


RANK, ALPHA = 2, 4.0
TARGETS = sorted(
    n[: -len(".weight")]
    for n, p in fresh_model().named_parameters()
    if n.startswith("blocks.") and n.endswith(".weight") and p.ndim == 2
)


def adapter_factors(kind: str) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7 if kind == "lora" else 8)
    f = {}
    for t in TARGETS:
        out_f, in_f = BASE[f"{t}.weight"].shape
        if kind == "lora":
            f[f"{t}.lora_A.weight"] = torch.randn(RANK, in_f, generator=g) * 0.1
            f[f"{t}.lora_B.weight"] = torch.randn(out_f, RANK, generator=g) * 0.1
        else:
            oa, ob = factorization(out_f)
            ia, ib = factorization(in_f)
            f[f"{t}.lokr_w1"] = torch.randn(oa, ia, generator=g) * 0.1
            if RANK < max(ob, ib) / 2:
                f[f"{t}.lokr_w2_a"] = torch.randn(ob, RANK, generator=g) * 0.1
                f[f"{t}.lokr_w2_b"] = torch.randn(RANK, ib, generator=g) * 0.1
            else:
                f[f"{t}.lokr_w2"] = torch.randn(ob, ib, generator=g) * 0.1
    return f


def delta(kind: str, t: str, p: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = ALPHA / RANK
    if kind == "lora":
        return scale * (p[f"{t}.lora_B.weight"] @ p[f"{t}.lora_A.weight"])
    w2 = p.get(f"{t}.lokr_w2")
    if w2 is None:
        w2 = p[f"{t}.lokr_w2_a"] @ p[f"{t}.lokr_w2_b"]
    out_f, in_f = BASE[f"{t}.weight"].shape
    return scale * torch.kron(p[f"{t}.lokr_w1"], w2).reshape(out_f, in_f)


for kind in ("lora", "lokr"):
    init = adapter_factors(kind)
    for k, v in init.items():
        out[f"{kind}.init.{k}"] = v
    for opt_kind in ("adamw", "muon"):
        model = fresh_model()
        for p in model.parameters():
            p.requires_grad_(False)
        factors = {k: torch.nn.Parameter(v.clone()) for k, v in init.items()}
        linears = dict(model.named_modules())
        for t in TARGETS:
            lin = linears[t]

            def forward(x, lin=lin, t=t):
                return torch.nn.functional.linear(x, lin.weight + delta(kind, t, factors), lin.bias)

            lin.forward = forward
        named = sorted(factors.items())
        ema = {k: v.detach().clone() for k, v in named}

        def ema_update():
            with torch.no_grad():
                for k, p in named:
                    ema[k].mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)

        groups = None
        if opt_kind == "muon":
            mats = [p for _, p in named if p.ndim == 2]
            rest = [p for _, p in named if p.ndim != 2]
            groups = [{"params": mats, "algorithm": "muon"}]
            if rest:
                groups.append({"params": rest, "algorithm": "adamw"})
        opt, sched = optimizer(opt_kind, named, None, groups)
        losses, norms = run(lambda mb: flow_loss(model, mb), named, opt, sched, ema_update)
        out[f"{kind}_{opt_kind}.losses"] = losses
        out[f"{kind}_{opt_kind}.grad_norms"] = norms
        for k, p in named:
            out[f"{kind}_{opt_kind}.final.{k}"] = p.detach().clone()
            out[f"{kind}_{opt_kind}.ema.{k}"] = ema[k]

# ---- Newton–Schulz ------------------------------------------------------------------------------
from dion.muon import zeropower_via_newtonschulz5  # noqa: E402

g = torch.Generator().manual_seed(5)
for name, shape in (("tall", (12, 4)), ("wide", (4, 10)), ("square", (6, 6))):
    m_ = torch.randn(*shape, generator=g)
    out[f"ns.{name}.in"] = m_
    out[f"ns.{name}.out"] = zeropower_via_newtonschulz5(m_.to(torch.bfloat16), epsilon=1e-8).float()

save_safetensors(
    OUT,
    out,
    metadata={
        "full_order": ",".join(FULL_ORDER),
        "targets": ",".join(TARGETS),
        "lr": LR, "wd": WD, "clip": CLIP, "ema_decay": EMA_DECAY, "accum": ACCUM, "steps": STEPS,
        "shift": SHIFT, "rank": RANK, "alpha": ALPHA, "dion": DION_COMMIT,
    },
)


# =================================================================================================
# sc-25689 — curricula, shapes, REPA / iREPA, activation checkpointing, holdout validation
# =================================================================================================
# Writes `mlx-gen-iris/tests/fixtures/iris_curricula_golden.safetensors` from the same frozen
# modules (plus `repa.py`, `models/ac.py`, `data/{shapes,buckets,samplers,datasets}.py`) and the
# `facebookresearch/dinov2` model code @ DINOV2_COMMIT (the source's torch-hub teacher). The REPA
# teacher here is a seeded miniature `DinoVisionTransformer` (the same class `dinov2_vitb14`
# builds) handed to the frozen `REPALoss` through `torch.hub.load`, so the alignment math runs
# upstream's code verbatim. Run the whole script (it regenerates both fixtures):
#     IRIS_SRC=… DION_SRC=… DINOV2_SRC=/path/to/dinov2 python -I tools/dump_iris_train.py
# With DINOV2_HUB_WEIGHTS (dinov2_vitb14_pretrain.pth) and DINOV2_HF_DIR (facebook/dinov2-base @
# DINOV2_HF_REVISION) set it also proves the two release-teacher resources hold the same tensors and
# writes the real-teacher reference `iris_repa_teacher_real.safetensors`.
import json  # noqa: E402
import math  # noqa: E402
from functools import partial  # noqa: E402

S11_UPSTREAM_SHA256 = {
    "src/iris3b/repa.py": "a2895bc4d903db9aaa3f44888d202b2c0a132127d6cc6aabfc951bb2f6262a75",
    "src/iris3b/models/ac.py": "c62f9a67d3849bfea966593051d1f9499c78c967a1f3dea7bd3fe990f38ce01a",
    "src/iris3b/data/shapes.py": "cded8bf52607fe65cc36f36e3f60d03daf193ef8d6cc3d17620ae34eb4d41209",
    "src/iris3b/data/buckets.py": "58fdb2c1d0e840c4ff7ee55f0e53975b209612240243d70bf8fc03da00d367c1",
    "src/iris3b/data/samplers.py": "c8a0ed0a5709d6c9faa248fba3c70b768ae0d7542709e823a844843731a09ad6",
    "src/iris3b/data/datasets.py": "05ab9064d9fbd2b7bdd390957512be6ba65e4d131796911dd325b229a2b48df8",
}
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_HF_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
_verify(require_env("IRIS_SRC", "the iris-3b checkout"), S11_UPSTREAM_SHA256, None)
DINOV2_SRC = require_env("DINOV2_SRC", f"a checkout of facebookresearch/dinov2 at {DINOV2_COMMIT}")
_verify(DINOV2_SRC, {}, DINOV2_COMMIT)
sys.path.insert(0, DINOV2_SRC)

from dinov2.layers import MemEffAttention, NestedTensorBlock  # noqa: E402
from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: E402
from PIL import Image  # noqa: E402

from iris3b.config import RepaConfig  # noqa: E402
from iris3b.data.buckets import TRAIN_BUCKETS  # noqa: E402
from iris3b.data.datasets import _center_crop, _resize_covering, _to_tensor  # noqa: E402
from iris3b.data.samplers import RangedSampler, ShapeBatchSampler  # noqa: E402
from iris3b.data.shapes import AreaPolicy, BucketPolicy  # noqa: E402
from iris3b.models import ac as upstream_ac  # noqa: E402
from iris3b.repa import REPALoss, _spatial_normalize  # noqa: E402
from iris3b.train.trainer import TrainModel  # noqa: E402

cur: dict[str, torch.Tensor] = {}
meta: dict[str, str] = {"dinov2": DINOV2_COMMIT}

# ---- shapes ------------------------------------------------------------------------------------
NATIVE = [(1000, 1000), (300, 1200), (2000, 500), (600, 300), (480, 640), (31, 47), (1024, 256), (700, 701)]
for table in ("shared21-512", "shared21-1024"):
    pol = BucketPolicy(table)
    meta[f"bucket.{table}"] = json.dumps([[s.height, s.width] for s in (pol.target(h, w) for h, w in NATIVE)])
area = AreaPolicy(tokens=(256 // 16) ** 2, patch=16, align=32, max_ratio=4.0)
meta["area.256.16.32.4"] = json.dumps([[s.height, s.width] for s in (area.target(h, w) for h, w in NATIVE)])
small = AreaPolicy(tokens=(32 // 4) ** 2, patch=4, align=8, max_ratio=4.0)
meta["area.32.4.8.4"] = json.dumps([[s.height, s.width] for s in (small.target(h, w) for h, w in NATIVE)])
# PixelDataset's shape-varying transform (covering resize + center crop) of a 31x47 image.
gimg = torch.Generator().manual_seed(11)
rgb = (torch.rand(31, 47, 3, generator=gimg) * 255).to(torch.uint8)
cur["shape.rgb"] = rgb
tgt = small.target(31, 47)
pil = Image.fromarray(rgb.numpy(), "RGB")
cur["shape.chw"] = _to_tensor(_center_crop(_resize_covering(pil, 31, 47, tgt.height, tgt.width), tgt.height, tgt.width))
meta["shape.target"] = json.dumps([tgt.height, tgt.width])

# ---- samplers ----------------------------------------------------------------------------------
s0 = RangedSampler(1300, rank=0, world_size=2)
s1 = RangedSampler(1300, rank=1, world_size=2)
meta["ranged.1300.w2"] = json.dumps([list(map(list, s0.ranges)), list(map(list, s1.ranges))])
state = RangedSampler(640, rank=0, world_size=2).resume_state(3, 4)
meta["ranged.640.w2.state"] = json.dumps(state)
re4 = []
for r in range(4):
    s = RangedSampler(640, rank=r, world_size=4)
    s.load_resume_state(state)
    re4.append(list(map(list, s.ranges)))
meta["ranged.640.w2_to_w4"] = json.dumps(re4)


class _KeyDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def key_of(self, idx):
        return "a" if idx % 3 == 0 else "b"


kd = _KeyDataset(30)
sb = ShapeBatchSampler(RangedSampler(30), kd, 2, drop_last=True)
sb.set_epoch(1)
meta["shape_batches.30.b2"] = json.dumps(sb.plan())
sb.set_epoch(1)
sb.set_start_batches(0, 2)
sstate = sb.resume_state(4, 2, samples_per_rank=8)
meta["shape_batches.state"] = json.dumps(sstate)
sb3 = ShapeBatchSampler(RangedSampler(30), kd, 3, drop_last=True)
sb3.set_epoch(1)
sb3.load_resume_state(sstate)
meta["shape_batches.regroup_b3"] = json.dumps(list(iter(sb3)))

# ---- activation checkpointing: identical gradients ---------------------------------------------
def grads_under(policy: str, every: int = 2) -> dict[str, torch.Tensor]:
    m = fresh_model()
    m.train()
    m.activation_checkpointing = policy
    m.ac_selective_every = every
    flow_loss(m, MICRO[0]).backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


g_none = grads_under("none")
ac_max = {}
for policy, every in (("full", 2), ("selective_op", 2), ("selective_layer", 2), ("selective_layer", 1)):
    upstream_ac.validate_policy(policy, every)
    g = grads_under(policy, every)
    ac_max[f"{policy}:{every}"] = max((g[n] - g_none[n]).abs().max().item() for n in g_none)
    assert set(g) == set(g_none)
meta["ac.max_abs_diff_vs_none"] = json.dumps(ac_max)
for n in SUBSET:
    if n in g_none:
        cur[f"ac.none.grad.{n}"] = g_none[n]

# ---- non-square (bucket) step, both RoPE laws --------------------------------------------------
gns = torch.Generator().manual_seed(21)
B2 = 2
ns = {
    "x0": torch.rand(B2, C, 8, 16, generator=gns) * 2 - 1,
    "noise": torch.randn(B2, C, 8, 16, generator=gns),
    "t_idx": torch.tensor([250, 777]),
    "y": MICRO[0]["y"],
    "y_mask": MICRO[0]["y_mask"],
}
for k, v in ns.items():
    cur[f"ns.{k}"] = v.float() if k != "t_idx" else v.to(torch.int32)
for aspect in ("isotropic", "square"):
    cfg_a = ModelConfig(**{**model_cfg, "rope_aspect": aspect})
    m = IrisDiT(cfg_a)
    m.load_state_dict(BASE, strict=True)
    m = m.float()
    loss = flow_loss(m, ns)
    loss.backward()
    cur[f"ns.{aspect}.loss"] = loss.detach().reshape(1)
    named = dict(m.named_parameters())
    for n in SUBSET:
        cur[f"ns.{aspect}.grad.{n}"] = named[n].grad.detach().clone()

# ---- REPA / iREPA ------------------------------------------------------------------------------
torch.manual_seed(77)


def tiny_teacher(registers: int = 0) -> DinoVisionTransformer:
    t = DinoVisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=2,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=registers,
        interpolate_antialias=registers > 0,
        interpolate_offset=0.0 if registers > 0 else 0.1,
    )
    with torch.no_grad():
        for p in t.parameters():
            p.add_(torch.randn_like(p) * 0.02)  # move off the trunc-normal / ones init
        for blk in t.blocks:
            blk.ls1.gamma.uniform_(0.5, 1.5)
            blk.ls2.gamma.uniform_(0.5, 1.5)
    return t.eval()


TEACHER = tiny_teacher()
TEACHER_REG = tiny_teacher(registers=4)
for name, t in (("teacher", TEACHER), ("teacher_reg", TEACHER_REG)):
    for k, v in t.state_dict().items():
        cur[f"{name}.{k}"] = v
torch.hub.load = lambda *a, **k: TEACHER  # REPALoss's teacher, through upstream's own call

IMAGES = MICRO[0]["x0"]  # [2, 3, 16, 16] clean images in [-1, 1]
CASES = {
    # teacher grid 2x2 < student 4x4: the student meets the teacher on 2x2
    "repa_t2": dict(variant="repa", teacher_image_size=28, teacher_match_student=False),
    # teacher sized to the student grid; iREPA's spatial normalization
    "irepa_match": dict(variant="irepa", teacher_image_size=224, teacher_match_student=True, spatial_norm_gamma=0.7),
    # teacher grid 6x6 > student 4x4: the teacher meets the student on 4x4
    "repa_t6": dict(variant="repa", teacher_image_size=84, teacher_match_student=False),
}
REPA_WEIGHT = 0.5
REPA_LAYER = 2
for case, kw in CASES.items():
    rcfg = RepaConfig(
        weight=REPA_WEIGHT, teacher_dim=32, proj_hidden_dim=24, teacher_patch_size=14, **kw
    )
    torch.manual_seed(1000 + len(case))
    repa = REPALoss(rcfg, student_dim=MODEL_CFG.hidden_size)
    assert repa.available
    cfg_r = ModelConfig(**{**model_cfg, "repa_layer": REPA_LAYER})
    core = IrisDiT(cfg_r)
    core.load_state_dict(BASE, strict=True)
    core = core.float()
    tm = TrainModel(core, repa)
    for k, v in repa.projector.state_dict().items():
        cur[f"repa.{case}.projector.{k}"] = v.detach().clone()
    out_l = FLOW.training_loss(
        tm, IMAGES, MICRO[0]["y"], timestep_idx=MICRO[0]["t_idx"], noise=MICRO[0]["noise"],
        model_kwargs={"clean": IMAGES, "y_mask": MICRO[0]["y_mask"]},
    )
    total = out_l.loss + REPA_WEIGHT * out_l.repa_loss
    total.backward()
    cur[f"repa.{case}.flow"] = out_l.loss.detach().reshape(1)
    cur[f"repa.{case}.repa"] = out_l.repa_loss.detach().reshape(1)
    named = dict(tm.named_parameters())
    for k, p in repa.projector.named_parameters():
        cur[f"repa.{case}.grad.projector.{k}"] = p.grad.detach().clone()
    for n in SUBSET:
        cur[f"repa.{case}.grad.{n}"] = named[f"model.{n}"].grad.detach().clone()
    grid = repa._teacher_grid((4, 4))
    with torch.no_grad():
        tt = repa._teacher_tokens(IMAGES, grid)
        cur[f"repa.{case}.teacher_tokens"] = tt
        if kw["variant"] == "irepa":
            cur[f"repa.{case}.teacher_normalized"] = _spatial_normalize(tt, rcfg.spatial_norm_gamma)
    meta[f"repa.{case}"] = json.dumps({**kw, "grid": list(grid)})
# the register teacher's antialiased position-table resize
with torch.no_grad():
    x = (IMAGES.float() + 1.0) / 2.0
    x = torch.nn.functional.interpolate(x, size=(42, 42), mode="bicubic", align_corners=False)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    cur["teacher_reg.tokens_3x3"] = TEACHER_REG.forward_features((x - mean) / std)["x_norm_patchtokens"]
meta["repa.weight"] = str(REPA_WEIGHT)
meta["repa.layer"] = str(REPA_LAYER)

# ---- frozen-grid holdout validation ------------------------------------------------------------
M64 = (1 << 64) - 1


def mix_seed(parts):
    h = 0
    for p in parts:
        h = (h + p + 0x9E3779B97F4A7C15) & M64
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & M64
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & M64
        h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF


class HostRng:
    """`gen_core::iris::train::HostRng` (SplitMix64 + Box-Muller), for the native noise grid."""

    def __init__(self, seed):
        self.state = seed & M64
        self.spare = None

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def normal(self):
        if self.spare is not None:
            v, self.spare = self.spare, None
            return v
        u1 = 1.0 - self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        self.spare = r * math.sin(th)
        return r * math.cos(th)


VAL_GRID_SEED, VAL_T_STRIDE = 0x5AF1, 619
ks = [0, 1, 2]
gv = torch.Generator().manual_seed(31)
val_x0 = torch.rand(len(ks), C, S, S, generator=gv) * 2 - 1
val_noise = torch.stack([
    torch.tensor((lambda r: [r.normal() for _ in range(C * S * S)])(HostRng(mix_seed([VAL_GRID_SEED, k]))), dtype=torch.float32).reshape(C, S, S)
    for k in ks
])
val_t = torch.tensor([(k * VAL_T_STRIDE) % 1000 for k in ks])
yv = torch.cat([MICRO[0]["y"], MICRO[1]["y"][:1]])
mv = torch.cat([MICRO[0]["y_mask"], MICRO[1]["y_mask"][:1]])
mval = fresh_model().eval()
with torch.no_grad():
    per = FLOW.training_loss(mval, val_x0, yv, timestep_idx=val_t, noise=val_noise, reduction="none",
                             model_kwargs={"y_mask": mv}).loss
cur["val.x0"] = val_x0
cur["val.noise"] = val_noise
cur["val.t_idx"] = val_t.to(torch.int32)
cur["val.y"] = yv
cur["val.y_mask"] = mv
cur["val.per_sample"] = per

save_safetensors(FIXTURE_DIR / "iris_curricula_golden.safetensors", cur, metadata=meta)

# ---- the release teacher (real weights, optional) ----------------------------------------------
HUB_WEIGHTS = os.environ.get("DINOV2_HUB_WEIGHTS")
HF_DIR = os.environ.get("DINOV2_HF_DIR")
if HUB_WEIGHTS and HF_DIR:
    from dinov2.hub.backbones import dinov2_vitb14

    hub_sd = torch.load(HUB_WEIGHTS, map_location="cpu")
    hf = load_file(os.path.join(HF_DIR, "model.safetensors"))
    vitb = dinov2_vitb14(pretrained=False)
    vitb.load_state_dict(hub_sd, strict=True)
    vitb.eval()

    def hf_name(k):  # hub → HF Dinov2Model (gen_core::iris::train::repa::hf_sources)
        if k.startswith("blocks."):
            i, leaf = k[len("blocks."):].split(".", 1)
            base = f"encoder.layer.{i}"
            return {
                "attn.proj.weight": [f"{base}.attention.output.dense.weight"],
                "attn.proj.bias": [f"{base}.attention.output.dense.bias"],
                "ls1.gamma": [f"{base}.layer_scale1.lambda1"],
                "ls2.gamma": [f"{base}.layer_scale2.lambda1"],
                "attn.qkv.weight": [f"{base}.attention.attention.{p}.weight" for p in ("query", "key", "value")],
                "attn.qkv.bias": [f"{base}.attention.attention.{p}.bias" for p in ("query", "key", "value")],
            }.get(leaf, [f"{base}.{leaf}"])
        return [{
            "cls_token": "embeddings.cls_token",
            "pos_embed": "embeddings.position_embeddings",
            "mask_token": "embeddings.mask_token",
            "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
            "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
            "norm.weight": "layernorm.weight",
            "norm.bias": "layernorm.bias",
        }[k]]

    worst = 0.0
    for k, v in hub_sd.items():
        parts = [hf[n] for n in hf_name(k)]
        w = torch.cat(parts, 0) if len(parts) > 1 else parts[0].reshape(v.shape)
        worst = max(worst, (w.float() - v.float()).abs().max().item())
    assert worst == 0.0, f"the HF dinov2-base mirror differs from the hub checkpoint by {worst}"
    gr = torch.Generator().manual_seed(41)
    img = torch.rand(1, 3, 64, 64, generator=gr) * 2 - 1
    real = {"image": img}
    torch.hub.load = lambda *a, **k: vitb
    for grid_name, rkw in (("g16", dict(teacher_image_size=224)), ("g4_match", dict(teacher_match_student=True))):
        rcfg = RepaConfig(weight=0.5, **rkw)
        rl = REPALoss(rcfg, student_dim=MODEL_CFG.hidden_size)
        grid = rl._teacher_grid((4, 4))
        with torch.no_grad():
            tok = rl._teacher_tokens(img, grid)
        real[f"{grid_name}.tokens"] = tok
    save_safetensors(
        FIXTURE_DIR / "iris_repa_teacher_real.safetensors",
        real,
        metadata={"dinov2": DINOV2_COMMIT, "hf_revision": DINOV2_HF_REVISION, "hub_vs_hf_max_abs": str(worst)},
    )
