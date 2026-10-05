//! Native executor for **SceneWorks fx-program** checkpoints (`sceneworks-fx-program/1`, epic 2123
//! sc-24831) — the MediaPipe FaceMesh-v2 landmark detector the face-landmark loss runs, lowered from
//! upstream's onnx2torch `GraphModule` by `tools/convert_mp_facemesh_v2.py` (format documented in
//! `tools/fx_program.py`). The program is a straight-line list of NCHW-semantics ops over a small
//! op set (conv incl. depthwise, PReLU, ReLU, sigmoid, binary arithmetic, max-pool, constant pad,
//! reshape, concat); this interpreter runs it with MLX ops, **differentiable in the input** (the
//! parameters are captured constants, so autograd only ever yields input/adapter gradients).
//!
//! Layout: 4-D activations are kept NHWC (MLX's native conv layout) and converted to NCHW only at
//! the ops whose semantics are layout-bound (`reshape`, `concat` axis, `pad` widths), so the program's
//! torch semantics are reproduced exactly. Conv kernels are stored OIHW (torch) and transposed to
//! MLX's OHWI once at load.

use std::collections::HashMap;

use mlx_gen::nn::conv2d_general;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{add, concatenate_axis, divide, maximum, minimum, multiply, pad, subtract};
use mlx_rs::Array;
use serde_json::Value;

/// The format tag a program checkpoint's `__metadata__.format` must carry.
pub const FORMAT: &str = "sceneworks-fx-program/1";

#[derive(Clone, Debug)]
enum Operand {
    Value(String),
    Param(String),
    Scalar(f32),
}

#[derive(Clone, Debug)]
enum Op {
    Conv {
        weight: String,
        bias: Option<String>,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
    },
    Prelu {
        slope: String,
    },
    Relu,
    Sigmoid,
    Binary(BinaryOp),
    MaxPool {
        kernel: (i32, i32),
        stride: (i32, i32),
        padding: (i32, i32),
    },
    Pad {
        pads: [(i32, i32); 4],
        value: f32,
    },
    Reshape(Vec<i32>),
    Concat(i32),
}

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
struct Node {
    op: Op,
    out: String,
    inputs: Vec<Operand>,
}

/// A parsed program (structure only — backend-free, shared shape with `candle-gen-face`).
#[derive(Clone, Debug)]
pub struct ProgramSpec {
    inputs: Vec<String>,
    outputs: Vec<Operand>,
    nodes: Vec<Node>,
}

fn pair(v: &Value, what: &str) -> Result<(i32, i32)> {
    let a = v
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| Error::Msg(format!("fx-program: {what} must be a pair, got {v}")))?;
    let g = |x: &Value| {
        x.as_i64()
            .map(|n| n as i32)
            .ok_or_else(|| Error::Msg(format!("fx-program: {what} entries must be integers")))
    };
    Ok((g(&a[0])?, g(&a[1])?))
}

fn string(v: &Value, key: &str) -> Result<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::Msg(format!("fx-program: node field '{key}' missing or not a string")))
}

fn operand(s: &str) -> Result<Operand> {
    Ok(if let Some(k) = s.strip_prefix("param:") {
        Operand::Param(k.to_owned())
    } else if let Some(v) = s.strip_prefix("scalar:") {
        Operand::Scalar(
            v.parse::<f32>()
                .map_err(|e| Error::Msg(format!("fx-program: bad scalar '{v}': {e}")))?,
        )
    } else {
        Operand::Value(s.to_owned())
    })
}

impl ProgramSpec {
    /// Parse the program JSON (the checkpoint's `__metadata__.program`).
    pub fn parse(json: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(json)
            .map_err(|e| Error::Msg(format!("fx-program: invalid program JSON: {e}")))?;
        Self::from_value(&v)
    }

    /// Parse an already-decoded program object.
    pub fn from_value(v: &Value) -> Result<Self> {
        let strings = |key: &str| -> Result<Vec<String>> {
            v.get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Msg(format!("fx-program: '{key}' must be a list")))?
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| Error::Msg(format!("fx-program: '{key}' entries are names")))
                })
                .collect()
        };
        let inputs = strings("inputs")?;
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "fx-program: exactly one input is supported, got {}",
                inputs.len()
            )));
        }
        let outputs = strings("outputs")?
            .iter()
            .map(|s| operand(s))
            .collect::<Result<Vec<_>>>()?;
        let mut nodes = Vec::new();
        for n in v
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Msg("fx-program: 'nodes' must be a list".into()))?
        {
            let kind = string(n, "op")?;
            let op = match kind.as_str() {
                "conv2d" => Op::Conv {
                    weight: string(n, "weight")?,
                    bias: n.get("bias").and_then(Value::as_str).map(str::to_owned),
                    stride: pair(&n["stride"], "stride")?,
                    padding: pair(&n["padding"], "padding")?,
                    dilation: pair(&n["dilation"], "dilation")?,
                    groups: n["groups"]
                        .as_i64()
                        .ok_or_else(|| Error::Msg("fx-program: conv groups".into()))?
                        as i32,
                },
                "prelu" => Op::Prelu {
                    slope: string(n, "slope")?,
                },
                "relu" => Op::Relu,
                "sigmoid" => Op::Sigmoid,
                "add" => Op::Binary(BinaryOp::Add),
                "sub" => Op::Binary(BinaryOp::Sub),
                "mul" => Op::Binary(BinaryOp::Mul),
                "div" => Op::Binary(BinaryOp::Div),
                "maxpool2d" => Op::MaxPool {
                    kernel: pair(&n["kernel"], "kernel")?,
                    stride: pair(&n["stride"], "stride")?,
                    padding: pair(&n["padding"], "padding")?,
                },
                "pad" => {
                    let p = n["pads"]
                        .as_array()
                        .filter(|p| p.len() == 4)
                        .ok_or_else(|| Error::Msg("fx-program: pad needs 4 axis pairs".into()))?;
                    let mut pads = [(0, 0); 4];
                    for (dst, src) in pads.iter_mut().zip(p) {
                        *dst = pair(src, "pads")?;
                    }
                    Op::Pad {
                        pads,
                        value: n["value"].as_f64().unwrap_or(0.0) as f32,
                    }
                }
                "reshape" => Op::Reshape(
                    n["shape"]
                        .as_array()
                        .ok_or_else(|| Error::Msg("fx-program: reshape shape".into()))?
                        .iter()
                        .map(|d| {
                            d.as_i64().map(|d| d as i32).ok_or_else(|| {
                                Error::Msg("fx-program: reshape dims are integers".into())
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                "concat" => Op::Concat(
                    n["axis"]
                        .as_i64()
                        .ok_or_else(|| Error::Msg("fx-program: concat axis".into()))?
                        as i32,
                ),
                other => {
                    return Err(Error::Msg(format!(
                        "fx-program: unsupported op '{other}' (node {})",
                        string(n, "out").unwrap_or_default()
                    )))
                }
            };
            let inputs = n
                .get("inputs")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Msg("fx-program: node inputs".into()))?
                .iter()
                .map(|s| {
                    operand(
                        s.as_str()
                            .ok_or_else(|| Error::Msg("fx-program: operands are strings".into()))?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            nodes.push(Node {
                op,
                out: string(n, "out")?,
                inputs,
            });
        }
        Ok(Self {
            inputs,
            outputs,
            nodes,
        })
    }

    /// Every parameter key the program reads, with whether it is a conv kernel.
    pub fn param_keys(&self) -> Vec<(String, bool)> {
        let mut keys = Vec::new();
        for n in &self.nodes {
            match &n.op {
                Op::Conv { weight, bias, .. } => {
                    keys.push((weight.clone(), true));
                    if let Some(b) = bias {
                        keys.push((b.clone(), false));
                    }
                }
                Op::Prelu { slope } => keys.push((slope.clone(), false)),
                _ => {}
            }
            for i in &n.inputs {
                if let Operand::Param(k) = i {
                    keys.push((k.clone(), false));
                }
            }
        }
        keys.sort();
        keys.dedup();
        keys
    }
}

/// A value during execution: a 4-D NHWC activation, or a plain (torch-layout) tensor.
#[derive(Clone)]
enum Val {
    Nhwc(Array),
    Plain(Array),
}

impl Val {
    /// Torch-layout view (NCHW for a 4-D activation).
    fn nchw(&self) -> Result<Array> {
        Ok(match self {
            Val::Nhwc(a) => a.transpose_axes(&[0, 3, 1, 2])?,
            Val::Plain(a) => a.clone(),
        })
    }

    /// NHWC view (a plain 4-D tensor is read as NCHW).
    fn nhwc(&self) -> Result<Array> {
        Ok(match self {
            Val::Nhwc(a) => a.clone(),
            Val::Plain(a) if a.ndim() == 4 => a.transpose_axes(&[0, 2, 3, 1])?,
            Val::Plain(a) => {
                return Err(Error::Msg(format!(
                    "fx-program: a spatial op needs a 4-D tensor, got shape {:?}",
                    a.shape()
                )))
            }
        })
    }
}

/// A loaded program: its structure plus frozen parameters (conv kernels pre-transposed to OHWI,
/// PReLU slopes flattened to `[C]`).
pub struct Program {
    spec: ProgramSpec,
    params: HashMap<String, Array>,
}

impl Program {
    /// Load a `sceneworks-fx-program/1` checkpoint (program in `__metadata__`, params as tensors).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let w = Weights::from_file(path)?;
        match w.metadata("format") {
            Some(FORMAT) => {}
            other => {
                return Err(Error::Msg(format!(
                    "{}: not a {FORMAT} checkpoint (format = {other:?})",
                    path.display()
                )))
            }
        }
        let json = w.metadata("program").ok_or_else(|| {
            Error::Msg(format!("{}: missing the 'program' metadata", path.display()))
        })?;
        Self::new(ProgramSpec::parse(json)?, &w)
    }

    /// Bind `spec` to the parameters in `w` (torch layout).
    pub fn new(spec: ProgramSpec, w: &Weights) -> Result<Self> {
        let mut params = HashMap::new();
        for (key, conv) in spec.param_keys() {
            let t = mlx_gen::weights::to_f32(w.require(&key)?)?;
            let t = if conv {
                if t.ndim() != 4 {
                    return Err(Error::Msg(format!(
                        "fx-program: conv weight '{key}' must be 4-D, got {:?}",
                        t.shape()
                    )));
                }
                t.transpose_axes(&[0, 2, 3, 1])?
            } else {
                t
            };
            t.eval()?;
            params.insert(key, t);
        }
        Ok(Self { spec, params })
    }

    fn param(&self, key: &str) -> Result<&Array> {
        self.params
            .get(key)
            .ok_or_else(|| Error::Msg(format!("fx-program: unbound parameter '{key}'")))
    }

    /// Run the program on an NHWC `[N, H, W, C]` input (the program's NCHW input, channels-last).
    /// Returns every program output in **torch layout** (e.g. FaceMesh's `[N, 1, 1, 1434]`).
    /// Differentiable in `x`.
    pub fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        let mut env: HashMap<&str, Val> = HashMap::new();
        env.insert(self.spec.inputs[0].as_str(), Val::Nhwc(x.clone()));
        for node in &self.spec.nodes {
            let get = |o: &Operand| -> Result<Val> {
                Ok(match o {
                    Operand::Value(n) => env
                        .get(n.as_str())
                        .cloned()
                        .ok_or_else(|| Error::Msg(format!("fx-program: undefined value '{n}'")))?,
                    Operand::Param(k) => Val::Plain(self.param(k)?.clone()),
                    Operand::Scalar(s) => Val::Plain(Array::from_f32(*s)),
                })
            };
            let one = || -> Result<Val> {
                node.inputs.first().map(get).ok_or_else(|| {
                    Error::Msg(format!("fx-program: node '{}' has no input", node.out))
                })?
            };
            let out = match &node.op {
                Op::Conv {
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let b = bias.as_ref().map(|b| self.param(b)).transpose()?;
                    Val::Nhwc(conv2d_general(
                        &one()?.nhwc()?,
                        self.param(weight)?,
                        b,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?)
                }
                Op::Prelu { slope } => {
                    let s = self.param(slope)?;
                    let x = one()?.nhwc()?;
                    let zero = Array::from_f32(0.0);
                    let pos = maximum(&x, &zero)?;
                    let neg = minimum(&x, &zero)?;
                    Val::Nhwc(add(&pos, &multiply(&neg, s)?)?)
                }
                Op::Relu => map_val(one()?, |a| Ok(maximum(a, Array::from_f32(0.0))?))?,
                Op::Sigmoid => map_val(one()?, |a| Ok(mlx_rs::ops::sigmoid(a)?))?,
                Op::Binary(op) => {
                    let (a, b) = (get(&node.inputs[0])?, get(&node.inputs[1])?);
                    binary(*op, &a, &b)?
                }
                Op::MaxPool {
                    kernel,
                    stride,
                    padding,
                } => Val::Nhwc(max_pool_nhwc(&one()?.nhwc()?, *kernel, *stride, *padding)?),
                Op::Pad { pads, value } => {
                    let v = one()?;
                    let fill = Array::from_f32(*value);
                    match v {
                        Val::Nhwc(a) => {
                            // NCHW pairs → NHWC axis order.
                            let w = [pads[0], pads[2], pads[3], pads[1]];
                            Val::Nhwc(pad(&a, &w[..], fill, None)?)
                        }
                        Val::Plain(a) => {
                            let w: Vec<(i32, i32)> = pads[4 - a.ndim().min(4)..].to_vec();
                            Val::Plain(pad(&a, &w[..], fill, None)?)
                        }
                    }
                }
                Op::Reshape(shape) => Val::Plain(one()?.nchw()?.reshape(shape)?),
                Op::Concat(axis) => {
                    let vals = node
                        .inputs
                        .iter()
                        .map(&get)
                        .collect::<Result<Vec<_>>>()?;
                    if vals.iter().all(|v| matches!(v, Val::Nhwc(_))) {
                        let ax = match axis.rem_euclid(4) {
                            0 => 0,
                            1 => 3,
                            2 => 1,
                            _ => 2,
                        };
                        let arrs = vals.iter().map(Val::nhwc).collect::<Result<Vec<_>>>()?;
                        Val::Nhwc(concatenate_axis(&arrs, ax)?)
                    } else {
                        let arrs = vals.iter().map(Val::nchw).collect::<Result<Vec<_>>>()?;
                        Val::Plain(concatenate_axis(&arrs, *axis)?)
                    }
                }
            };
            env.insert(node.out.as_str(), out);
        }
        self.spec
            .outputs
            .iter()
            .map(|o| match o {
                Operand::Value(n) => env
                    .get(n.as_str())
                    .ok_or_else(|| Error::Msg(format!("fx-program: undefined output '{n}'")))?
                    .nchw(),
                Operand::Param(k) => Ok(self.param(k)?.clone()),
                Operand::Scalar(s) => Ok(Array::from_f32(*s)),
            })
            .collect()
    }

    /// Total parameter count (for the E7 footprint).
    pub fn param_count(&self) -> usize {
        self.params.values().map(|a| a.size()).sum()
    }
}

fn map_val(v: Val, f: impl Fn(&Array) -> Result<Array>) -> Result<Val> {
    Ok(match v {
        Val::Nhwc(a) => Val::Nhwc(f(&a)?),
        Val::Plain(a) => Val::Plain(f(&a)?),
    })
}

fn binary(op: BinaryOp, a: &Val, b: &Val) -> Result<Val> {
    let f = |x: &Array, y: &Array| -> Result<Array> {
        Ok(match op {
            BinaryOp::Add => add(x, y)?,
            BinaryOp::Sub => subtract(x, y)?,
            BinaryOp::Mul => multiply(x, y)?,
            BinaryOp::Div => divide(x, y)?,
        })
    };
    let scalar_like = |v: &Val| matches!(v, Val::Plain(p) if p.ndim() == 0);
    Ok(match (a, b) {
        (Val::Nhwc(x), Val::Nhwc(y)) => Val::Nhwc(f(x, y)?),
        (Val::Nhwc(x), s) if scalar_like(s) => Val::Nhwc(f(x, &s.nchw()?)?),
        (s, Val::Nhwc(y)) if scalar_like(s) => Val::Nhwc(f(&s.nchw()?, y)?),
        _ => Val::Plain(f(&a.nchw()?, &b.nchw()?)?),
    })
}

/// Max-pool over NHWC with `-inf` padding (torch `max_pool2d` semantics, floor mode): the max over
/// the `kh·kw` strided window views — differentiable (gradient to the arg-max like torch).
fn max_pool_nhwc(
    x: &Array,
    kernel: (i32, i32),
    stride: (i32, i32),
    padding: (i32, i32),
) -> Result<Array> {
    use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
    let x = if padding != (0, 0) {
        pad(
            x,
            &[(0, 0), padding.into(), padding.into(), (0, 0)][..],
            Array::from_f32(f32::NEG_INFINITY),
            None,
        )?
    } else {
        x.clone()
    };
    let s = x.shape();
    let (h, w) = (s[1], s[2]);
    let oh = (h - kernel.0) / stride.0 + 1;
    let ow = (w - kernel.1) / stride.1 + 1;
    let mut out: Option<Array> = None;
    for dy in 0..kernel.0 {
        for dx in 0..kernel.1 {
            let view = x.index((
                ..,
                (dy..dy + (oh - 1) * stride.0 + 1).stride_by(stride.0),
                (dx..dx + (ow - 1) * stride.1 + 1).stride_by(stride.1),
                ..,
            ));
            out = Some(match out {
                Some(m) => maximum(&m, &view)?,
                None => view,
            });
        }
    }
    out.ok_or_else(|| Error::Msg("fx-program: empty max-pool kernel".into()))
}
