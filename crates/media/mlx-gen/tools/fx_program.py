"""Lower a small torch CNN (a `torch.fx.GraphModule`, e.g. the onnx2torch-converted MediaPipe
FaceMesh-v2 `face_landmarks_detector`) to the **SceneWorks fx-program** format that the native
`mlx-gen-face` / `candle-gen-face` program executors run (epic 2123, sc-24831).

Format (`sceneworks-fx-program/1`): one safetensors file whose tensors are the parameters (torch
layout — conv kernels OIHW) and whose `__metadata__` carries

- `format`  = `"sceneworks-fx-program/1"`
- `program` = JSON `{"inputs": [name], "outputs": [name], "nodes": [node, ...]}`

Every node is `{"op": <op>, "out": <name>, "inputs": [<operand>, ...], ...attrs}`; an operand is a
value name, `"param:<key>"` (a stored tensor) or `"scalar:<float>"`. All tensor semantics are torch's
**NCHW**. Ops:

    conv2d    weight, bias|null, stride [sh,sw], padding [ph,pw], dilation [dh,dw], groups
    prelu     slope (param key, [C])
    relu, sigmoid
    add, sub, mul, div        (two operands, broadcasting)
    maxpool2d kernel [kh,kw], stride [sh,sw], padding [ph,pw]
    pad       pads [[before, after] x 4] over N,C,H,W; value
    reshape   shape [..] (torch semantics, -1 allowed)
    concat    axis
    getitem   index (select one output of a multi-output node; produced only by the tracer)

Anything else raises — the converter never guesses at an op it does not know, so a new op in a real
checkpoint surfaces as a named error here, not as a silently wrong native forward.
"""

from __future__ import annotations

import json
import operator
from typing import Any

import torch
import torch.nn.functional as F

FORMAT = "sceneworks-fx-program/1"


def _pair(v) -> list[int]:
    if isinstance(v, (tuple, list)):
        assert len(v) == 2, v
        return [int(v[0]), int(v[1])]
    return [int(v), int(v)]


def _torch_pads_to_nchw(pads, ndim: int = 4) -> list[list[int]]:
    """torch `F.pad` order (last dim first: [w_b, w_a, h_b, h_a, c_b, c_a, ...]) → per-axis pairs."""
    pads = [int(p) for p in pads]
    out = [[0, 0] for _ in range(ndim)]
    for i in range(len(pads) // 2):
        axis = ndim - 1 - i
        out[axis] = [pads[2 * i], pads[2 * i + 1]]
    return out


def _onnx_pads_to_nchw(pads) -> list[list[int]]:
    """ONNX `Pad` order ([x1_begin, ..., xn_begin, x1_end, ..., xn_end]) → per-axis pairs."""
    pads = [int(p) for p in pads]
    n = len(pads) // 2
    out = [[0, 0] for _ in range(4)]
    for i in range(n):
        out[4 - n + i] = [pads[i], pads[n + i]]
    return out


class _Lowerer:
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm
        self.modules = dict(gm.named_modules())
        self.params: dict[str, torch.Tensor] = {}
        self.nodes: list[dict[str, Any]] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.const_values: dict[str, torch.Tensor] = {}

    # -- helpers ---------------------------------------------------------------------------
    def param(self, key: str, t: torch.Tensor) -> str:
        key = key.replace("/", ".")
        if key not in self.params:
            self.params[key] = t.detach().to(torch.float32).contiguous().clone()
        return f"param:{key}"

    def operand(self, a) -> str:
        if isinstance(a, torch.fx.Node):
            if a.name in self.const_values:
                return self.param(a.name, self.const_values[a.name])
            return a.name
        if isinstance(a, (int, float)):
            return f"scalar:{float(a)!r}"
        raise ValueError(f"unsupported operand {a!r}")

    def const_of(self, a):
        """The concrete value of a constant operand (a get_attr buffer or a python literal)."""
        if isinstance(a, torch.fx.Node):
            if a.name in self.const_values:
                return self.const_values[a.name]
            raise ValueError(f"operand {a.name} is not a constant")
        return a

    def emit(self, op: str, node, inputs: list[str], **attrs):
        out = node if isinstance(node, str) else node.name
        self.nodes.append({"op": op, "out": out, "inputs": inputs, **attrs})

    # -- per-node lowering -----------------------------------------------------------------
    def lower(self):
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.inputs.append(node.name)
            elif node.op == "get_attr":
                t = self.gm
                for part in node.target.split("."):
                    t = getattr(t, part)
                self.const_values[node.name] = t
            elif node.op == "call_module":
                self.call_module(node)
            elif node.op == "call_function":
                self.call_function(node, node.target)
            elif node.op == "call_method":
                self.call_method(node)
            elif node.op == "output":
                outs = node.args[0]
                outs = outs if isinstance(outs, (tuple, list)) else [outs]
                self.outputs = [self.operand(o) for o in outs]
            else:
                raise ValueError(f"unsupported fx node kind {node.op}")

    def call_module(self, node):
        self.lower_module(self.modules[node.target], node.target, node.name, list(node.args))

    def lower_module(self, m, target: str, out: str, args: list):
        """Lower module `m` applied to fx `args` (a value operand may also be a plain value-name
        string — a `Sequential`'s intermediate), producing value `out`."""
        cls = type(m).__name__
        operand = lambda a: a if isinstance(a, str) else self.operand(a)  # noqa: E731
        x = args[0] if args else None
        if isinstance(m, torch.nn.Sequential):
            # onnx2torch wraps fused blocks (e.g. a static pad + conv) in a Sequential: chain the
            # children through intermediate values.
            children = list(m.named_children())
            if not children:
                raise ValueError(f"{target}: empty Sequential")
            cur = x
            for i, (name, child) in enumerate(children):
                step_out = out if i == len(children) - 1 else f"{out}__{i}"
                self.lower_module(child, f"{target}.{name}", step_out, [cur])
                cur = step_out
        elif isinstance(m, torch.nn.Conv2d):
            if m.padding_mode != "zeros":
                raise ValueError(f"{target}: conv padding_mode {m.padding_mode}")
            pad = m.padding
            if isinstance(pad, str):
                raise ValueError(f"{target}: string conv padding {pad!r}")
            self.emit(
                "conv2d",
                out,
                [operand(x)],
                weight=self.param(f"{target}.weight", m.weight)[6:],
                bias=(self.param(f"{target}.bias", m.bias)[6:] if m.bias is not None else None),
                stride=_pair(m.stride),
                padding=_pair(pad),
                dilation=_pair(m.dilation),
                groups=int(m.groups),
            )
        elif isinstance(m, torch.nn.PReLU) or "PReLU" in cls or "Prelu" in cls:
            slope = getattr(m, "weight", None)
            if slope is None and len(args) > 1:
                # onnx2torch `OnnxPReLU(x, slope)`: the slope is a constant operand.
                slope = self.const_of(args[1])
            if slope is None:
                tensors = list(m.parameters()) + list(m.buffers())
                if len(tensors) != 1:
                    raise ValueError(f"{target}: cannot find the PReLU slope ({cls})")
                slope = tensors[0]
            # Only a scalar or per-channel slope ([C], [C,1,1], [1,C,1,1]) reduces to torch `prelu`.
            dims = [int(d) for d in slope.shape]
            channel_axis = {1: 0, 3: 0, 4: 1}.get(len(dims))
            per_channel = channel_axis is not None and all(
                d == 1 for i, d in enumerate(dims) if i != channel_axis
            )
            if slope.nelement() != 1 and not per_channel:
                raise ValueError(f"{target}: non-per-channel PReLU slope {tuple(dims)}")
            self.emit(
                "prelu",
                out,
                [operand(x)],
                slope=self.param(f"{target}.slope", slope.reshape(-1))[6:],
            )
        elif isinstance(m, torch.nn.ReLU):
            self.emit("relu", out, [operand(x)])
        elif isinstance(m, torch.nn.Sigmoid) or cls == "OnnxSigmoid":
            self.emit("sigmoid", out, [operand(x)])
        elif isinstance(m, torch.nn.MaxPool2d):
            if m.dilation not in (1, (1, 1), [1, 1]) or m.ceil_mode:
                raise ValueError(f"{target}: unsupported maxpool dilation/ceil_mode")
            self.emit(
                "maxpool2d",
                out,
                [operand(x)],
                kernel=_pair(m.kernel_size),
                stride=_pair(m.stride if m.stride is not None else m.kernel_size),
                padding=_pair(m.padding),
            )
        elif cls == "OnnxPadDynamic":
            # forward(x, pads, constant_value=0): `pads` is in ONNX order
            # [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
            if getattr(m, "mode", "constant") != "constant":
                raise ValueError(f"{target}: pad mode {m.mode}")
            pads = [int(v) for v in self.const_of(args[1]).tolist()]
            value = float(self.const_of(args[2])) if len(args) > 2 and args[2] is not None else 0.0
            self.emit("pad", out, [operand(x)], pads=_onnx_pads_to_nchw(pads), value=value)
        elif "Pad" in cls:
            # `OnnxPadStatic` / torch-style modules: `pads` already in torch `F.pad` order.
            mode = getattr(m, "mode", "constant")
            if mode != "constant":
                raise ValueError(f"{target}: pad mode {mode}")
            pads = getattr(m, "pads", None)
            if pads is None:
                raise ValueError(f"{target}: cannot find the pads of {cls}")
            value = float(getattr(m, "constant_value", 0.0) or 0.0)
            self.emit("pad", out, [operand(x)], pads=_torch_pads_to_nchw(pads), value=value)
        elif "Reshape" in cls:
            shape = self.const_of(args[1])
            shape = shape.tolist() if torch.is_tensor(shape) else list(shape)
            self.emit("reshape", out, [operand(x)], shape=[int(s) for s in shape])
        elif "BinaryMath" in cls:
            fn = getattr(m, "math_op_function", None)
            name = getattr(fn, "__name__", str(fn))
            op = {"add": "add", "sub": "sub", "mul": "mul", "div": "div", "true_divide": "div"}.get(name)
            if op is None:
                raise ValueError(f"{target}: unsupported binary op {name}")
            self.emit(op, out, [operand(a) for a in args[:2]])
        elif "Concat" in cls:
            self.emit("concat", out, [operand(a) for a in args], axis=int(m.axis))
        else:
            raise ValueError(f"{target}: unsupported module {cls}")

    def call_function(self, node, fn):
        args = node.args
        binary = {
            operator.add: "add", torch.add: "add",
            operator.sub: "sub", torch.sub: "sub",
            operator.mul: "mul", torch.mul: "mul",
            operator.truediv: "div", torch.div: "div",
        }
        if fn in binary:
            self.emit(binary[fn], node, [self.operand(a) for a in args[:2]])
        elif fn in (torch.sigmoid, F.sigmoid):
            self.emit("sigmoid", node, [self.operand(args[0])])
        elif fn in (torch.relu, F.relu):
            self.emit("relu", node, [self.operand(args[0])])
        elif fn is F.prelu:
            self.emit(
                "prelu", node, [self.operand(args[0])],
                slope=self.param(f"{node.name}.slope", self.const_of(args[1]).reshape(-1))[6:],
            )
        elif fn is F.pad:
            mode = node.kwargs.get("mode", args[2] if len(args) > 2 else "constant")
            if mode != "constant":
                raise ValueError(f"{node.name}: pad mode {mode}")
            value = node.kwargs.get("value", args[3] if len(args) > 3 else 0.0) or 0.0
            self.emit("pad", node, [self.operand(args[0])],
                      pads=_torch_pads_to_nchw(args[1]), value=float(value))
        elif fn is torch.reshape:
            self.emit("reshape", node, [self.operand(args[0])], shape=[int(s) for s in args[1]])
        elif fn is torch.cat:
            axis = node.kwargs.get("dim", args[1] if len(args) > 1 else 0)
            self.emit("concat", node, [self.operand(a) for a in args[0]], axis=int(axis))
        elif fn is F.max_pool2d or getattr(fn, "__name__", "") in ("max_pool2d", "fn") and "max_pool2d" in node.name:
            kernel = node.kwargs.get("kernel_size", args[1] if len(args) > 1 else None)
            stride = node.kwargs.get("stride", args[2] if len(args) > 2 else None) or kernel
            padding = node.kwargs.get("padding", args[3] if len(args) > 3 else 0)
            if node.kwargs.get("ceil_mode", False) or node.kwargs.get("dilation", 1) not in (1, (1, 1)):
                raise ValueError(f"{node.name}: unsupported maxpool dilation/ceil_mode")
            self.emit("maxpool2d", node, [self.operand(args[0])],
                      kernel=_pair(kernel), stride=_pair(stride), padding=_pair(padding))
        elif fn is operator.getitem:
            self.emit("getitem", node, [self.operand(args[0])], index=int(args[1]))
        else:
            raise ValueError(f"{node.name}: unsupported function {fn}")

    def call_method(self, node):
        name, args = node.target, node.args
        if name in ("reshape", "view"):
            shape = args[1:] if len(args) > 2 or isinstance(args[1], int) else args[1]
            self.emit("reshape", node, [self.operand(args[0])], shape=[int(s) for s in shape])
        elif name == "sigmoid":
            self.emit("sigmoid", node, [self.operand(args[0])])
        elif name in ("add", "sub", "mul", "div"):
            self.emit(name, node, [self.operand(a) for a in args[:2]])
        else:
            raise ValueError(f"{node.name}: unsupported method {name}")


def lower(module: torch.nn.Module) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Lower `module` (an fx GraphModule, or any symbolically traceable module) to
    `(program, params)`."""
    gm = module if isinstance(module, torch.fx.GraphModule) else torch.fx.symbolic_trace(module)
    lw = _Lowerer(gm)
    lw.lower()
    program = {"inputs": lw.inputs, "outputs": lw.outputs, "nodes": lw.nodes}
    return program, lw.params


def save(module: torch.nn.Module, path: str) -> dict[str, Any]:
    """Lower `module` and write the program checkpoint to `path`; returns the program."""
    from safetensors.torch import save_file

    program, params = lower(module)
    save_file(params, path, metadata={"format": FORMAT, "program": json.dumps(program)})
    return program
