//! MLX executor for **SceneWorks fx-program** checkpoints (`sceneworks-fx-program/1`, epic 2123
//! sc-24831) — the MediaPipe FaceMesh-v2 landmark detector the face-landmark loss runs, lowered from
//! upstream's onnx2torch `GraphModule` by `tools/convert_mp_facemesh_v2.py`. The program structure
//! is parsed once, backend-neutrally, by [`gen_core::fx_program`] (shared with `candle-gen-face`);
//! this interpreter runs it with MLX ops, **differentiable in the input** (the parameters are
//! captured constants, so autograd only ever yields input/adapter gradients).
//!
//! Layout: 4-D activations are kept NHWC (MLX's native conv layout) and converted to NCHW only at
//! the ops whose semantics are layout-bound (`reshape`, `concat` axis), so the program's torch
//! semantics are reproduced exactly. Conv kernels are stored OIHW (torch) and transposed to MLX's
//! OHWI once at load.

use std::collections::HashMap;

use mlx_gen::gen_core::fx_program::{BinaryOp, Op, Operand, ProgramSpec, FORMAT};
use mlx_gen::nn::conv2d_general;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{add, concatenate_axis, divide, maximum, minimum, multiply, pad, subtract};
use mlx_rs::Array;

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

fn i(v: (usize, usize)) -> (i32, i32) {
    (v.0 as i32, v.1 as i32)
}

/// A loaded program: its structure plus frozen parameters (conv kernels pre-transposed to OHWI).
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

    fn operand(&self, env: &HashMap<&str, Val>, o: &Operand) -> Result<Val> {
        Ok(match o {
            Operand::Value(n) => env
                .get(n.as_str())
                .cloned()
                .ok_or_else(|| Error::Msg(format!("fx-program: undefined value '{n}'")))?,
            Operand::Param(k) => Val::Plain(self.param(k)?.clone()),
            Operand::Scalar(s) => Val::Plain(Array::from_f32(*s)),
        })
    }

    /// Run the program on an NHWC `[N, H, W, C]` input (the program's NCHW input, channels-last).
    /// Returns every program output in **torch layout** (e.g. FaceMesh's `[N, 1, 1, 1434]`).
    /// Differentiable in `x`.
    pub fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        let mut env: HashMap<&str, Val> = HashMap::new();
        env.insert(self.spec.input.as_str(), Val::Nhwc(x.clone()));
        for node in &self.spec.nodes {
            let ins = node
                .inputs
                .iter()
                .map(|o| self.operand(&env, o))
                .collect::<Result<Vec<_>>>()?;
            let first = || {
                ins.first().cloned().ok_or_else(|| {
                    Error::Msg(format!("fx-program: node '{}' has no input", node.out))
                })
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
                        &first()?.nhwc()?,
                        self.param(weight)?,
                        b,
                        i(*stride),
                        i(*padding),
                        i(*dilation),
                        *groups as i32,
                    )?)
                }
                Op::Prelu { slope } => {
                    let x = first()?.nhwc()?;
                    let zero = Array::from_f32(0.0);
                    let neg = multiply(&minimum(&x, &zero)?, self.param(slope)?)?;
                    Val::Nhwc(add(&maximum(&x, &zero)?, &neg)?)
                }
                Op::Relu => map_val(first()?, |a| Ok(maximum(a, Array::from_f32(0.0))?))?,
                Op::Sigmoid => map_val(first()?, |a| Ok(mlx_rs::ops::sigmoid(a)?))?,
                Op::Binary(op) => {
                    if ins.len() != 2 {
                        return Err(Error::Msg(format!(
                            "fx-program: binary node '{}' needs 2 operands",
                            node.out
                        )));
                    }
                    binary(*op, &ins[0], &ins[1])?
                }
                Op::MaxPool {
                    kernel,
                    stride,
                    padding,
                } => Val::Nhwc(max_pool_nhwc(
                    &first()?.nhwc()?,
                    i(*kernel),
                    i(*stride),
                    i(*padding),
                )?),
                Op::Pad { pads, value } => {
                    let fill = Array::from_f32(*value);
                    let p = |a: (usize, usize)| (a.0 as i32, a.1 as i32);
                    match first()? {
                        // NCHW pairs → NHWC axis order.
                        Val::Nhwc(a) => {
                            let w = [p(pads[0]), p(pads[2]), p(pads[3]), p(pads[1])];
                            Val::Nhwc(pad(&a, &w[..], fill, None)?)
                        }
                        Val::Plain(a) => {
                            let w: Vec<(i32, i32)> =
                                pads[4 - a.ndim().min(4)..].iter().map(|&x| p(x)).collect();
                            Val::Plain(pad(&a, &w[..], fill, None)?)
                        }
                    }
                }
                Op::Reshape(shape) => {
                    let v = first()?.nchw()?;
                    let dims = ProgramSpec::resolve_shape(shape, v.size())?;
                    let dims: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
                    Val::Plain(v.reshape(&dims)?)
                }
                Op::Concat(axis) => {
                    if ins.iter().all(|v| matches!(v, Val::Nhwc(_))) {
                        let ax = match axis.rem_euclid(4) {
                            0 => 0,
                            1 => 3,
                            2 => 1,
                            _ => 2,
                        };
                        let arrs = ins.iter().map(Val::nhwc).collect::<Result<Vec<_>>>()?;
                        Val::Nhwc(concatenate_axis(&arrs, ax)?)
                    } else {
                        let arrs = ins.iter().map(Val::nchw).collect::<Result<Vec<_>>>()?;
                        Val::Plain(concatenate_axis(&arrs, *axis as i32)?)
                    }
                }
            };
            env.insert(node.out.as_str(), out);
        }
        self.spec
            .outputs
            .iter()
            .map(|o| self.operand(&env, o)?.nchw())
            .collect()
    }

    /// Total parameter count.
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

/// Max-pool over NHWC with `-inf` padding (torch `max_pool2d`, floor mode): the max over the
/// `kh·kw` strided window views — differentiable.
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
            &[(0, 0), padding, padding, (0, 0)][..],
            Array::from_f32(f32::NEG_INFINITY),
            None,
        )?
    } else {
        x.clone()
    };
    let s = x.shape();
    let oh = (s[1] - kernel.0) / stride.0 + 1;
    let ow = (s[2] - kernel.1) / stride.1 + 1;
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
