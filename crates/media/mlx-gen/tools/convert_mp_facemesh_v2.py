"""Convert the MediaPipe FaceMesh-v2 landmark detector to the native fx-program checkpoint the
epic-2123 face-landmark loss runs (sc-24831).

Source: `py-feat/mp_facemesh_v2` → `face_landmarks_detector_Nx3x256x256_onnx.pth` — the exact
checkpoint upstream ai-toolkit-perceptual's `DifferentiableLandmarkEncoder` loads (an onnx2torch
`GraphModule` of MediaPipe's `face_landmarks_detector`: input `[N,3,256,256]` RGB in `[0,1]`,
output 0 `[N,1,1,1434]` = 478 × (x, y, z) in 256-px crop space).

    pip install torch onnx2torch safetensors
    python3 convert_mp_facemesh_v2.py <path/to/face_landmarks_detector_Nx3x256x256_onnx.pth> \
        <out_dir>

writes `<out_dir>/face_landmarks_detector.safetensors` (see `fx_program.py` for the format) and
prints a self-check: the lowered program is re-executed by a tiny reference interpreter and must
match the torch module's output 0 to 1e-4. Any op the lowering does not know raises naming it.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_program  # noqa: E402

OUT_FILE = "face_landmarks_detector.safetensors"


def run_program(program, params, x):
    """Reference interpreter for the fx-program format (torch, NCHW) — the self-check oracle."""
    env = {program["inputs"][0]: x}

    def val(o):
        if o.startswith("param:"):
            return params[o[6:]]
        if o.startswith("scalar:"):
            return float(o[7:])
        return env[o]

    for n in program["nodes"]:
        ins = [val(i) for i in n["inputs"]]
        op = n["op"]
        if op == "conv2d":
            y = F.conv2d(ins[0], params[n["weight"]], params[n["bias"]] if n["bias"] else None,
                         n["stride"], n["padding"], n["dilation"], n["groups"])
        elif op == "prelu":
            y = F.prelu(ins[0], params[n["slope"]])
        elif op == "relu":
            y = F.relu(ins[0])
        elif op == "sigmoid":
            y = torch.sigmoid(ins[0])
        elif op in ("add", "sub", "mul", "div"):
            a, b = ins
            y = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        elif op == "maxpool2d":
            y = F.max_pool2d(ins[0], n["kernel"], n["stride"], n["padding"])
        elif op == "pad":
            flat = []
            for before, after in reversed(n["pads"]):
                flat += [before, after]
            y = F.pad(ins[0], flat, value=n["value"])
        elif op == "reshape":
            y = ins[0].reshape(n["shape"])
        elif op == "concat":
            y = torch.cat(ins, dim=n["axis"])
        elif op == "getitem":
            y = ins[0][n["index"]]
        else:
            raise ValueError(op)
        env[n["out"]] = y
    return [val(o) for o in program["outputs"]]


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    model = torch.load(src, map_location="cpu", weights_only=False)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, OUT_FILE)
    program, params = fx_program.lower(model)
    fx_program.save(model, path)
    x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = model(x)
        want = want[0] if isinstance(want, (tuple, list)) else want
        got = run_program(program, params, x)[0]
    err = (got - want).abs().max().item()
    ops = sorted({n["op"] for n in program["nodes"]})
    print(json.dumps({"out": path, "nodes": len(program["nodes"]), "ops": ops,
                      "params": len(params), "output0_shape": list(want.shape),
                      "self_check_max_abs": err}))
    if err > 1e-4:
        sys.exit(f"self-check failed: max abs {err}")


if __name__ == "__main__":
    main()
