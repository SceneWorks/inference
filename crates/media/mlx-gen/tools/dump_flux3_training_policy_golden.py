"""Independent CPU source-LR/clip/accumulation oracle; research/test only.

-I, --source exact diffusers7ecb optimization.py, --out an owned JSON fixture.
Only audited scheduler function definitions are extracted; no package/CLI execution,
model weights, GPU calls, credential access or production training dependency.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
source = args.source.read_bytes()
expected_hash = "e9cd7c846beec6c4e0f55a74c8a07b55c2d192200a0f7ece9022a3c40846930e"
if hashlib.sha256(source).hexdigest() != expected_hash:
    raise SystemExit("optimization.py differs from exact diffusers7ecb1e1b source")
functions = {
    "constant": "get_constant_schedule",
    "constant_with_warmup": "get_constant_schedule_with_warmup",
    "linear": "get_linear_schedule_with_warmup",
    "cosine": "get_cosine_schedule_with_warmup",
    "cosine_with_restarts": "get_cosine_with_hard_restarts_schedule_with_warmup",
    "polynomial": "get_polynomial_decay_schedule_with_warmup",
}
module = ast.parse(source)
definitions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in functions.values()]
if len(definitions) != len(functions):
    raise SystemExit("missing audited source scheduler definition")
namespace = {"Optimizer": torch.optim.Optimizer, "LambdaLR": torch.optim.lr_scheduler.LambdaLR, "math": math}
exec(compile(ast.Module(body=definitions, type_ignores=[]), "frozen-diffusers7ecb-optimization-functions", "exec"), namespace)
torch.set_num_threads(2)
schedules = []
for name, function in functions.items():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.0003)
    kwargs = {} if name == "constant" else {"num_warmup_steps": 2}
    if name not in ["constant", "constant_with_warmup"]:
        kwargs["num_training_steps"] = 6
    if name == "cosine_with_restarts":
        kwargs["num_cycles"] = 2
    if name == "polynomial":
        kwargs["power"] = 2.0
    scheduler = namespace[function](optimizer, **kwargs)
    values = []
    for step in range(8):
        values.append(optimizer.param_groups[0]["lr"] / 0.0003)
        parameter.grad = torch.tensor([0.5])
        optimizer.step()
        scheduler.step()
    schedules.append({"name": name, "warmup": 2, "total_updates": 6, "cycles": 2, "power": 2.0, "base_lr": 0.0003, "multipliers": values})

clips = []
for gradient, maximum in [([3.0, 4.0], 1.0), ([3.0, 4.0], 5.0), ([0.0, 0.0], 0.0)]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor(gradient)
    norm = torch.nn.utils.clip_grad_norm_([parameter], maximum)
    coefficient = torch.clamp(maximum / (norm + 1e-6), max=1.0)
    clips.append({"gradient": gradient, "max_norm": maximum, "norm": norm.item(), "coefficient": coefficient.item(), "clipped": parameter.grad.tolist()})

parameter = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0)
updates = []
# Accelerator.backward's non-DeepSpeed contract divides EACH loss by configured N.
# Epoch/final short-window sync does not replace N by the actual pending count.
for micro, gradient in enumerate([1.0, 3.0, 9.0], 1):
    (parameter.sum() * gradient / 2).backward()
    if micro in [2, 3]:
        updates.append({"micro_step": micro, "accumulated_gradient": parameter.grad.item()})
        optimizer.step()
        updates[-1]["parameter"] = parameter.item()
        optimizer.zero_grad(set_to_none=True)

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

output = {
    "schema": 1,
    "training_reference": "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
    "optimization_sha256": expected_hash,
    "reference_environment": {"python": sys.version, **{name: version(name) for name in ["torch", "diffusers", "transformers", "datasets", "Pillow", "accelerate"]}},
    "accumulation_evidence": "Accelerate1.12.0 authoritative source backward/default-plugin contract audited; NOT a dependency pin or installed Accelerate execution",
    "schedules": schedules, "clips": clips, "three_micro_accumulation_two": updates,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf8")
print("CPU policy oracle: six exact-source LR schedules, Torch clip coefficients, configured-N short accumulation tail")
