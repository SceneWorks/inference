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
