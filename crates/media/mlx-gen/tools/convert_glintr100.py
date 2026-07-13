#!/usr/bin/env python3
"""Convert antelopev2 `glintr100.onnx` (ArcFace iresnet100) -> safetensors for the
native MLX port (sc-3081), and dump golden inputs + embeddings for the parity test.

The onnx export (opset 11) is the standard insightface iresnet100 with BatchNorm
*after* each conv ALREADY FOLDED into the conv (every Conv carries a bias). Only the
"pre-activation" BNs stay explicit: each block's `bn1`, the final `bn2`, and the
`features` BatchNorm1d. We fold those to per-channel affine (scale/shift) here so the
Rust forward is pure conv+prelu+affine+add+linear (no runtime BatchNorm op).

Graph (verified): stem `Conv(3->64,3x3,s1)[+bias] -> PRelu`; then layers [3,13,30,3]
of IBasicBlock `BN(bn1) -> Conv(conv1)[+b] -> PRelu -> Conv(conv2,stride)[+b]
-> [Conv(downsample) on block 0] -> Add`; head `BN(bn2) -> Flatten(NCHW) -> Gemm(fc)
-> BN(features)` -> 512-d.

Conv weights are onnx OIHW -> transposed to MLX OHWI [out,kH,kW,in]. fp32 throughout
(the reference runs fp32; cosine parity is trivial at fp32).

Outputs (under tools/golden/, gitignored):
  arcface_iresnet100.safetensors  -- converted weights, canonical keys
  arcface_goldens.safetensors     -- inputs [K,112,112,3] f32 (normalized) + embeddings [K,512] f32

Run with the dwpose-spike venv (onnx + onnxruntime + numpy):
  ~/.dwpose-spike/venv/bin/python tools/convert_glintr100.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

LAYERS = [3, 13, 30, 3]  # iresnet100
BN_EPS = 1e-5
GLINTR = os.path.expanduser("~/.insightface/models/antelopev2/glintr100.onnx")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "golden")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = onnx.load(GLINTR)
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    nodes = list(g.node)

    out: dict[str, np.ndarray] = {}

    def f32(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

    def conv_w(name):  # onnx OIHW -> MLX OHWI [out,kH,kW,in]
        return f32(np.transpose(init[name], (0, 2, 3, 1)))

    def fold_bn(prefix, node):
        # inputs: [x, weight, bias, running_mean, running_var]
        w = init[node.input[1]].astype(np.float64)
        b = init[node.input[2]].astype(np.float64)
        mean = init[node.input[3]].astype(np.float64)
        var = init[node.input[4]].astype(np.float64)
        scale = w / np.sqrt(var + BN_EPS)
        shift = b - mean * scale
        out[f"{prefix}.scale"] = f32(scale)
        out[f"{prefix}.shift"] = f32(shift)

    i = 0

    def expect(op):
        nonlocal i
        n = nodes[i]
        assert n.op_type == op, f"node {i}: expected {op}, got {n.op_type}"
        i += 1
        return n

    # --- stem: Conv(+folded bn1 bias) -> PRelu
    n = expect("Conv")
    out["stem.conv.weight"] = conv_w(n.input[1])
    out["stem.conv.bias"] = f32(init[n.input[2]])
    n = expect("PRelu")
    out["stem.prelu.weight"] = f32(init[n.input[1]].reshape(-1))

    # --- blocks
    for L, nb in enumerate(LAYERS, start=1):
        for B in range(nb):
            p = f"layer{L}.{B}"
            fold_bn(f"{p}.bn1", expect("BatchNormalization"))
            n = expect("Conv")  # conv1 (+folded bn2)
            out[f"{p}.conv1.weight"] = conv_w(n.input[1])
            out[f"{p}.conv1.bias"] = f32(init[n.input[2]])
            n = expect("PRelu")
            out[f"{p}.prelu.weight"] = f32(init[n.input[1]].reshape(-1))
            n = expect("Conv")  # conv2 (+folded bn3), carries the stride
            out[f"{p}.conv2.weight"] = conv_w(n.input[1])
            out[f"{p}.conv2.bias"] = f32(init[n.input[2]])
            if B == 0:
                n = expect("Conv")  # downsample 1x1 (+folded bn)
                out[f"{p}.downsample.weight"] = conv_w(n.input[1])
                out[f"{p}.downsample.bias"] = f32(init[n.input[2]])
            expect("Add")

    # --- head: BN(bn2) -> Flatten -> Gemm(fc) -> BN(features)
    fold_bn("bn2", expect("BatchNormalization"))
    expect("Flatten")
    n = expect("Gemm")
    transB = next((a.i for a in n.attribute if a.name == "transB"), 0)
    fcw = init[n.input[1]]
    # core nn::linear does addmm(b, x, w.t()) expecting w = [out, in]. onnx Gemm with
    # transB=1 stores w as [out, in] already; transB=0 would be [in, out] (transpose).
    if not transB:
        fcw = fcw.T
    out["fc.weight"] = f32(fcw)
    out["fc.bias"] = f32(init[n.input[2]])
    fold_bn("features", expect("BatchNormalization"))

    assert i == len(nodes), f"walked {i} of {len(nodes)} nodes"

    # report a few shapes for sanity
    print(f"converted {len(out)} tensors from {len(nodes)} nodes")
    for k in ("stem.conv.weight", "stem.prelu.weight", "layer1.0.bn1.scale",
              "layer1.0.conv1.weight", "layer1.0.downsample.weight",
              "layer4.2.conv2.weight", "bn2.scale", "fc.weight", "fc.bias",
              "features.scale"):
        print(f"  {k:30} {tuple(out[k].shape)}")

    # --- write weights via safetensors (numpy backend)
    from safetensors.numpy import save_file
    wpath = os.path.join(OUT_DIR, "arcface_iresnet100.safetensors")
    save_file(out, wpath)
    print("wrote", wpath)

    # --- goldens: deterministic inputs + onnx embeddings
    import onnxruntime as ort
    rng = np.random.default_rng(3081)
    K = 4
    imgs_u8 = rng.integers(0, 256, size=(K, 112, 112, 3), dtype=np.uint8)
    inputs_nhwc = ((imgs_u8.astype(np.float32) - 127.5) / 127.5)  # [K,112,112,3]
    sess = ort.InferenceSession(GLINTR, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    nchw = np.ascontiguousarray(np.transpose(inputs_nhwc, (0, 3, 1, 2)))  # [K,3,112,112]
    embs = sess.run(None, {in_name: nchw})[0].astype(np.float32)  # [K,512]
    print("golden embeddings shape", embs.shape, "norm[0]", float(np.linalg.norm(embs[0])))

    from safetensors.numpy import save_file as save_file2
    gpath = os.path.join(OUT_DIR, "arcface_goldens.safetensors")
    save_file2({"inputs": np.ascontiguousarray(inputs_nhwc), "embeddings": np.ascontiguousarray(embs)}, gpath)
    print("wrote", gpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
